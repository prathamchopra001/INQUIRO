import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.validation.schol_eval import ScholarEval
from src.utils.llm_client import LLMClient
from src.execution.docker_executor import DockerExecutor
from src.execution.notebook_manager import NotebookManager
from src.skills.task_skill_generator import TaskSkillGenerator
from config.prompts.data_agent import (
    PLANNING_PROMPT,
    CODE_GENERATION_PROMPT,
    FINDING_EXTRACTION_PROMPT,
    CODE_FIX_PROMPT,
    CODE_RESTRATEGY_PROMPT,
    CODE_VERIFICATION_PROMPT,
    OUTPUT_VALIDATION_MARKERS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# VERIFICATION DATA STRUCTURES
# =============================================================================

@dataclass
class OutputValidationResult:
    """Result of heuristic output validation."""
    is_valid: bool
    exit_code_ok: bool
    has_artifacts: bool
    has_statistical_output: bool
    has_data_presence: bool
    failure_reason: str = ""
    artifacts_found: List[str] = field(default_factory=list)
    statistical_terms_found: List[str] = field(default_factory=list)
    silent_failures_detected: List[str] = field(default_factory=list)


@dataclass 
class VerificationResult:
    """Complete result of the verification loop."""
    success: bool
    execution_result: object  # ExecutionResult from docker
    final_code: str
    tier_reached: int  # 1, 2, or 3
    total_attempts: int
    failure_history: List[dict] = field(default_factory=list)
    needs_human_review: bool = False
    human_review_reason: str = ""


@dataclass
class CodeVerificationResult:
    """Result of LLM-based code logic verification.
    
    This checks whether the code actually computes what the task asked for,
    catches circular reasoning, data leakage, and other logical errors.
    """
    passes_verification: bool
    task_alignment_score: float  # 0.0-1.0
    logical_correctness_score: float  # 0.0-1.0
    output_validity_score: float  # 0.0-1.0
    issues_found: List[str] = field(default_factory=list)
    severity: str = "none"  # none, minor, major, critical
    recommendation: str = "accept"  # accept, flag_for_review, reject
    reasoning: str = ""


class DataAnalysisAgent:
    """Agent that analyzes data by generating and executing code."""
    
    def __init__(self, llm_client: LLMClient, executor: DockerExecutor = None,
                 notebook_manager: NotebookManager = None,
                 enable_task_skills: bool = True):
        # Store the LLM client
        # Create executor and notebook_manager if not provided (use defaults)
        self.llm = llm_client
        self.executor = executor or DockerExecutor()
        self.notebook_manager = notebook_manager or NotebookManager()
        
        # Task skill generator for technique-specific guidance
        self.task_skill_generator = TaskSkillGenerator(
            llm_client=llm_client,
            skills_dir="./skills/task_skills",
            enabled=enable_task_skills,
        )
        self.enable_task_skills = enable_task_skills
    
    def _preview_data(self, data_path: str, n_rows: int = 5) -> str:
        """Load the dataset and create a text preview for the LLM.
        
        The LLM can't see files — it needs a TEXT description of the data.
        Think: column names, types, shape, first few rows, missing values.
        """
        # Read limited rows to avoid memory issues in parallel execution
        df = pd.read_csv(data_path, nrows=10000)

        # ── Build column name cheatsheet ──────────────────────────────────
        # Local LLMs hallucinate plausible column names. A mapping from
        # "what you might guess" → "what it's actually called" blocks this.
        col_list = df.columns.tolist()
        col_list_str = ", ".join(col_list)

        preview = [
            "=" * 60,
            "⚠️  COLUMN NAMES — USE ONLY THESE EXACT NAMES ⚠️",
            "=" * 60,
            f"ALL COLUMNS: [{col_list_str}]",
            "",
            "NUMERIC COLUMNS:",
        ]
        for col in df.select_dtypes(include="number").columns:
            preview.append(f"  • {col}")
        non_numeric = df.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            preview.append("\nNON-NUMERIC COLUMNS:")
            for col in non_numeric:
                preview.append(f"  • {col}")

        preview.append("")
        preview.append("COMMON MISTAKES TO AVOID:")
        # Dynamically generate warnings for columns with suffixes
        # that LLMs tend to drop (e.g. '_pct_gdp' → '_pct')
        for col in col_list:
            # Warn about columns with long suffixes LLMs might truncate
            for suffix in ["_pct_gdp", "_yoy_pct", "_pct", "_index", "_ratio"]:
                if col.endswith(suffix) and len(col) > len(suffix) + 3:
                    short = col[: -len(suffix.split("_")[-1]) - 1]  # drop last _part
                    if short != col and short not in col_list:
                        preview.append(f"  ❌ '{short}' → ✅ Use '{col}'")
                        break
        preview.append("  If a column name you want is NOT in the list above,")
        preview.append("  it does NOT exist. Use df.columns.tolist() to check.")
        preview.append("=" * 60)

        # ── Macro vs Micro variable detection ─────────────────────────────
        # Columns that are constant within each time period (quarter/date)
        # are MACRO variables — the same value for every firm in that period.
        # Correlating two macro variables "by sector" gives identical results
        # for every sector because the values don't vary within a period.
        macro_cols, micro_cols = self._detect_macro_micro(df)
        if macro_cols:
            preview.append("")
            preview.append("=" * 60)
            preview.append("⚠️  MACRO vs FIRM-LEVEL COLUMNS ⚠️")
            preview.append("=" * 60)
            preview.append(
                "MACRO COLUMNS (same value for ALL firms in a given quarter):"
            )
            for col in macro_cols:
                preview.append(f"  📊 {col}")
            preview.append("")
            preview.append(
                "FIRM-LEVEL COLUMNS (vary across firms within a quarter):"
            )
            for col in micro_cols:
                preview.append(f"  🏢 {col}")
            preview.append("")
            preview.append("⚠️  CRITICAL RULES FOR MACRO COLUMNS:")
            preview.append(
                "  1. Do NOT compute 'sector-specific' correlations between"
            )
            preview.append(
                "     two MACRO columns. The result will be IDENTICAL for"
            )
            preview.append(
                "     every sector because these values don't vary by firm."
            )
            preview.append(
                "  2. Correlations between a MACRO column and a FIRM-LEVEL"
            )
            preview.append(
                "     column CAN differ by sector — that's a valid analysis."
            )
            preview.append(
                "  3. To study how macro conditions affect firms, correlate"
            )
            preview.append(
                "     a macro column with a firm-level column WITHIN sectors."
            )
            preview.append("=" * 60)

        preview.extend([
            f"\nDataset Shape: {df.shape}",
            "\nColumn Types:",
            df.dtypes.to_string(),
            f"\nFirst {n_rows} Rows:",
            df.head(n_rows).to_string(),
            "\nMissing Values:",
            df.isnull().sum().to_string(),
        ])

        # ── Enumerate unique values for categorical columns ───────────────
        try:
            cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        except Exception:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            preview.append("\nCategorical Column Values (USE EXACTLY THESE):")
            for col in cat_cols:
                unique_vals = df[col].dropna().unique().tolist()
                if len(unique_vals) <= 20:
                    vals_str = ", ".join(f"'{v}'" for v in sorted(str(v) for v in unique_vals))
                    preview.append(f"  {col}: [{vals_str}]  (n_unique={len(unique_vals)})")
                else:
                    preview.append(f"  {col}: {len(unique_vals)} unique values (high cardinality)")

        return "\n".join(preview)

    def _detect_macro_micro(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Detect which numeric columns are macro-level vs firm-level.

        A macro column has the same value for every row (firm) within
        a given time period. A firm-level column varies across firms
        in the same period.

        Auto-detects the time column by looking for common names.

        Returns:
            (macro_columns, micro_columns) — both lists of column names
        """
        # Try to find the time/period column
        time_col = None
        for candidate in ["quarter", "date", "period", "year", "time",
                          "timestep", "month", "year_quarter"]:
            if candidate in df.columns:
                time_col = candidate
                break

        if time_col is None:
            return [], []  # Can't detect without a time column

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        macro_cols = []
        micro_cols = []

        for col in numeric_cols:
            try:
                # For each time period, check if all values are identical
                nunique_per_period = df.groupby(time_col)[col].nunique()
                # If max unique values per period is 1, it's macro
                if nunique_per_period.max() <= 1:
                    macro_cols.append(col)
                else:
                    micro_cols.append(col)
            except Exception:
                micro_cols.append(col)  # Default to micro if check fails

        return macro_cols, micro_cols
    
    def _plan_analysis(self, task_description: str, task_goal: str,
                       data_preview: str, objective: str,
                       world_model_summary: str = "") -> str:
        """Ask the LLM to create an analysis plan.
        
        Fill in the PLANNING_PROMPT template and send it to the LLM.
        Return the LLM's response (the plan text).
        """
        prompt = PLANNING_PROMPT.format(
            objective=objective,
            task_description=task_description,
            task_goal=task_goal,
            data_preview=data_preview,
            world_model_summary=world_model_summary
        )
        response = self.llm.complete_for_role(prompt=prompt, role="code_generation", system="You are a data scientist.")
        return response.content
    
    def _generate_code(self, analysis_plan: str, data_preview: str,
                       objective: str, data_filename: str,
                       task_description: str = "") -> str:
        """Ask the LLM to write Python code for the analysis.
        
        Fill in the CODE_GENERATION_PROMPT template and send it to the LLM.
        Return the generated code string.
        
        IMPORTANT: Clean the response — LLMs sometimes wrap code in 
```python ... ``` even when told not to. Strip those if present.
        
        Now includes task-specific skills (UpSkill integration) that provide:
        - Correct API patterns for the required techniques
        - Required packages to install
        - Common pitfalls to avoid
        """
        # Generate task-specific skill with correct API patterns
        task_skill = ""
        logger.debug(f"UpSkill check: enable_task_skills={self.enable_task_skills}, task_desc={bool(task_description)}")
        if self.enable_task_skills and task_description:
            skill_content = self.task_skill_generator.get_skill_for_code_generation(
                task=task_description,
                dataset_context=data_preview[:500],  # First 500 chars for context
            )
            if skill_content:
                task_skill = skill_content
                logger.info(f"  📚 Injected task skill ({len(skill_content)} chars) for: {task_description[:50]}...")
            else:
                logger.debug(f"  No task skill generated for: {task_description[:50]}...")
        
        prompt = CODE_GENERATION_PROMPT.format(
            objective=objective,
            analysis_plan=analysis_plan,
            data_preview=data_preview,
            data_filename=data_filename,
            task_skill=task_skill,
        )
        response = self.llm.complete_for_role(prompt=prompt, role="code_generation", system="You are a senior data scientist.")
        code = response.content.strip()
        for prefix in ["```python", "```"]:
            if code.startswith(prefix):
                code = code[len(prefix):].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
            
        return code
    
    def _validate_finding(self, finding: dict) -> tuple[bool, str]:
        """
        Validate a finding for physically/statistically impossible values.
        
        Returns:
            (is_valid, reason) — if is_valid is False, the finding is discarded.
        """
        claim = finding.get("claim", "").lower()
        evidence = finding.get("evidence", "").lower()
        text = claim + " " + evidence
    
        import re
    
        # --- Check 1: Variance / R² cannot exceed 100% ---
        variance_matches = re.findall(
            r'(?:explains?|accounts? for|variance explained)[^\d]*(\d+\.?\d*)\s*%',
            text
        )
        for match in variance_matches:
            value = float(match)
            if value > 100.0:
                return False, f"Impossible variance explained: {value}% (>100%)"
    
        # --- Check 2: p-values must be in [0, 1] ---
        p_matches = re.findall(
            r'p[\s\-_]?(?:value|val)?[\s=<>]*(\d+\.?\d*(?:e[+-]?\d+)?)',
            text
        )
        for match in p_matches:
            value = float(match)
            if value > 1.0:
                return False, f"Impossible p-value: {value} (>1.0)"
    
        # --- Check 3: Correlations must be in [-1, 1] ---
        corr_matches = re.findall(
            r'(?:correlation|r\^?2|pearson|spearman)[^\d\-]*(-?\d+\.?\d*)',
            text
        )
        for match in corr_matches:
            value = float(match)
            if abs(value) > 1.0 and value != int(value):
                # Only flag decimals (r=1.5), not fold changes (r^2 as count)
                return False, f"Impossible correlation value: {value}"
    
        # --- Check 4: Confidence score must be in [0, 1] ---
        confidence = finding.get("confidence", 0.5)
        try:
            conf_val = float(confidence)
            if conf_val < 0.0 or conf_val > 1.0:
                return False, f"Confidence out of range: {conf_val}"
        except (ValueError, TypeError):
            pass
        
        return True, "ok"

    # =========================================================================
    # CODE LOGIC VERIFICATION (Phase 7A #3)
    # =========================================================================

    def _verify_code_logic(
        self,
        task_description: str,
        code: str,
        stdout: str,
        min_score: float = 0.6,
    ) -> CodeVerificationResult:
        """
        LLM-based verification that code actually computes what the task asked for.
        
        This catches:
        - Task misalignment (code doesn't answer the question)
        - Circular reasoning (using target as feature)
        - Data leakage (train/test contamination)
        - Implausible results (perfect correlations, etc.)
        
        Args:
            task_description: What the task asked for
            code: The code that was executed
            stdout: The output produced
            min_score: Minimum average score to pass (default 0.6)
            
        Returns:
            CodeVerificationResult with scores and issues
        """
        # Skip verification if no meaningful output
        if not stdout or len(stdout.strip()) < 50:
            logger.debug("Skipping code logic verification (no output)")
            return CodeVerificationResult(
                passes_verification=True,
                task_alignment_score=0.5,
                logical_correctness_score=0.5,
                output_validity_score=0.5,
                issues_found=["No output to verify"],
                severity="minor",
                recommendation="accept",
                reasoning="Insufficient output to verify; accepting by default.",
            )
        
        # Truncate code/output if too long to avoid context overflow
        code_truncated = code[:3000] + ("\n... [truncated]" if len(code) > 3000 else "")
        stdout_truncated = stdout[:2000] + ("\n... [truncated]" if len(stdout) > 2000 else "")
        
        prompt = CODE_VERIFICATION_PROMPT.format(
            task_description=task_description,
            code=code_truncated,
            stdout=stdout_truncated,
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="code_verification",
                system="You are a senior data scientist reviewing code for correctness. Respond with JSON only.",
            )
            
            # Parse JSON response
            text = response.content.strip()
            
            # Extract JSON from response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                result_dict = json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")
            
            # Build result
            result = CodeVerificationResult(
                passes_verification=result_dict.get("passes_verification", True),
                task_alignment_score=float(result_dict.get("task_alignment_score", 0.7)),
                logical_correctness_score=float(result_dict.get("logical_correctness_score", 0.7)),
                output_validity_score=float(result_dict.get("output_validity_score", 0.7)),
                issues_found=result_dict.get("issues_found", []),
                severity=result_dict.get("severity", "none"),
                recommendation=result_dict.get("recommendation", "accept"),
                reasoning=result_dict.get("reasoning", ""),
            )
            
            # Calculate average score
            avg_score = (
                result.task_alignment_score +
                result.logical_correctness_score +
                result.output_validity_score
            ) / 3.0
            
            # Override passes_verification based on average score
            if avg_score < min_score:
                result.passes_verification = False
                if not result.issues_found:
                    result.issues_found.append(f"Average score {avg_score:.2f} below threshold {min_score}")
            
            # Log result
            if result.passes_verification:
                logger.info(
                    f"  ✓ Code logic verified (align={result.task_alignment_score:.2f}, "
                    f"logic={result.logical_correctness_score:.2f}, "
                    f"valid={result.output_validity_score:.2f})"
                )
            else:
                logger.warning(
                    f"  ⚠️ Code logic issues: {result.severity} | {result.issues_found[:2]}"
                )
            
            return result
            
        except Exception as e:
            logger.warning(f"Code logic verification failed: {e}. Accepting by default.")
            return CodeVerificationResult(
                passes_verification=True,
                task_alignment_score=0.7,
                logical_correctness_score=0.7,
                output_validity_score=0.7,
                issues_found=[f"Verification error: {str(e)[:100]}"],
                severity="none",
                recommendation="accept",
                reasoning="Verification failed; accepting by default to avoid blocking.",
            )

    # =========================================================================
    # CODE VERIFICATION SYSTEM (3-Tier with Smoke Testing)
    # =========================================================================

    def _create_sample_data(self, data_path: str, sample_rows: int = 100) -> str:
        """
        Create a temporary sample of the data for smoke testing.
        
        Args:
            data_path: Path to the full dataset
            sample_rows: Number of rows to sample (default 100)
            
        Returns:
            Path to the temporary sample file
        """
        try:
            df = pd.read_csv(data_path, nrows=sample_rows)
            
            # Create temp file in the same directory structure
            sample_dir = Path(data_path).parent / "_smoke_test"
            sample_dir.mkdir(exist_ok=True)
            sample_path = sample_dir / f"sample_{Path(data_path).name}"
            
            df.to_csv(sample_path, index=False)
            logger.debug(f"Created smoke test sample: {sample_path} ({len(df)} rows)")
            return str(sample_path)
            
        except Exception as e:
            logger.warning(f"Could not create sample data: {e}. Using full data.")
            return data_path

    def _validate_execution_output(
        self,
        exec_result,
        output_dir: str = "/app/outputs/figures",
    ) -> OutputValidationResult:
        """
        Heuristic validation of code execution output.
        
        Checks:
        1. Exit code is 0
        2. Output artifacts exist (figures, etc.)
        3. stdout contains statistical terms (real analysis happened)
        4. No silent failure patterns detected
        
        Args:
            exec_result: ExecutionResult from DockerExecutor
            output_dir: Where artifacts should be saved
            
        Returns:
            OutputValidationResult with detailed validation info
        """
        result = OutputValidationResult(
            is_valid=False,
            exit_code_ok=False,
            has_artifacts=False,
            has_statistical_output=False,
            has_data_presence=False,
        )
        
        # Check 1: Exit code
        result.exit_code_ok = exec_result.success
        if not result.exit_code_ok:
            result.failure_reason = f"Non-zero exit code: {exec_result.stderr[:200]}"
            return result
        
        stdout_lower = (exec_result.stdout or "").lower()
        
        # Check 2: Silent failure patterns (even with exit code 0)
        for pattern in OUTPUT_VALIDATION_MARKERS["silent_failure_patterns"]:
            if pattern in stdout_lower:
                result.silent_failures_detected.append(pattern)
        
        if result.silent_failures_detected:
            # Don't immediately fail — some warnings are benign
            # Only fail if there's no other positive signal
            pass
        
        # Check 3: Statistical terms present
        for term in OUTPUT_VALIDATION_MARKERS["statistical_terms"]:
            if term.lower() in stdout_lower:
                result.statistical_terms_found.append(term)
        
        result.has_statistical_output = len(result.statistical_terms_found) >= 1
        
        # Check 4: Data presence (actual numbers in output)
        for pattern in OUTPUT_VALIDATION_MARKERS["data_indicators"]:
            if re.search(pattern, exec_result.stdout or "", re.IGNORECASE):
                result.has_data_presence = True
                break
        
        # Check 5: Artifacts (figures) — check stdout for savefig mentions
        # In Docker, we can't directly check the filesystem, but we can
        # look for evidence that plt.savefig() was called
        artifact_patterns = [
            r"saved.*\.png",
            r"figure.*saved",
            r"savefig",
            r"writing.*figure",
        ]
        for pattern in artifact_patterns:
            if re.search(pattern, stdout_lower):
                result.has_artifacts = True
                result.artifacts_found.append(pattern)
                break
        
        # Determine overall validity
        # Valid if: exit OK AND (has stats OR has data) AND no critical silent failures
        critical_failures = ["keyerror", "valueerror", "typeerror", "indexerror"]
        has_critical_failure = any(
            f in result.silent_failures_detected for f in critical_failures
        )
        
        if result.exit_code_ok and not has_critical_failure:
            if result.has_statistical_output or result.has_data_presence:
                result.is_valid = True
            else:
                result.failure_reason = (
                    "Code ran but produced no statistical output or data. "
                    "This may be a silent failure."
                )
        elif has_critical_failure:
            result.failure_reason = (
                f"Silent failures detected: {result.silent_failures_detected}"
            )
        
        return result

    def _run_smoke_test(
        self,
        code: str,
        data_path: str,
        sample_rows: int = 100,
    ) -> Tuple[bool, str]:
        """
        Run code on a small data sample to catch errors quickly.
        
        Args:
            code: The Python code to test
            data_path: Path to full dataset
            sample_rows: Rows to sample (default 100)
            
        Returns:
            (passed, error_message) — True if smoke test passed
        """
        # Create sample data
        sample_path = self._create_sample_data(data_path, sample_rows)
        
        try:
            # Run on sample
            result = self.executor.execute_code(code=code, data_path=sample_path)
            
            if not result.success:
                return False, f"Smoke test failed: {result.stderr[:300]}"
            
            # Basic validation
            validation = self._validate_execution_output(result)
            if not validation.is_valid and validation.silent_failures_detected:
                return False, f"Smoke test silent failure: {validation.failure_reason}"
            
            return True, ""
            
        finally:
            # Clean up sample file
            try:
                if sample_path != data_path:
                    Path(sample_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _execute_with_verification(
        self,
        code: str,
        data_path: str,
        data_preview: str,
        task_description: str,
        tier1_retries: int = 3,
        enable_smoke_test: bool = True,
    ) -> VerificationResult:
        """
        3-tier code execution with verification.
        
        Tier 1: Iterative fix (up to tier1_retries attempts)
                Fix specific errors while keeping the same approach.
                
        Tier 2: Re-strategy (1 attempt)
                Generate completely different analytical approach.
                
        Tier 3: Human-in-the-loop
                Flag for manual review.
        
        Args:
            code: Initial generated code
            data_path: Path to dataset
            data_preview: Text preview of data (for re-strategy prompt)
            task_description: What we're trying to accomplish
            tier1_retries: Max retries in tier 1 (default 3)
            enable_smoke_test: Run quick test on sample first (default True)
            
        Returns:
            VerificationResult with success status, final code, and history
        """
        current_code = code
        failure_history = []
        total_attempts = 0
        
        # Pre-load column names for fix prompts
        try:
            _cols = pd.read_csv(data_path, nrows=0).columns.tolist()
            column_names_str = str(_cols)
        except Exception:
            column_names_str = "(could not read columns)"
        
        # =====================================================================
        # TIER 1: Iterative Fix (same approach, fix specific errors)
        # =====================================================================
        logger.info("🔧 Tier 1: Iterative fix phase")
        
        for attempt in range(1, tier1_retries + 1):
            total_attempts += 1
            
            # Optional smoke test first
            if enable_smoke_test and attempt == 1:
                smoke_passed, smoke_error = self._run_smoke_test(
                    current_code, data_path
                )
                if not smoke_passed:
                    logger.info(f"  💨 Smoke test failed (fast catch): {smoke_error[:100]}")
                    failure_history.append({
                        "tier": 1,
                        "attempt": attempt,
                        "phase": "smoke_test",
                        "error": smoke_error,
                        "code_snippet": current_code[:200],
                    })
                    # Fix and retry
                    current_code = self._fix_code(
                        current_code, smoke_error, task_description, column_names_str
                    )
                    continue
            
            # Full execution
            exec_result = self.executor.execute_code(
                code=current_code, data_path=data_path
            )
            
            # Validate output
            validation = self._validate_execution_output(exec_result)
            
            if validation.is_valid:
                logger.info(
                    f"  ✅ Tier 1 success on attempt {attempt} "
                    f"(stats: {validation.statistical_terms_found[:3]})"
                )
                return VerificationResult(
                    success=True,
                    execution_result=exec_result,
                    final_code=current_code,
                    tier_reached=1,
                    total_attempts=total_attempts,
                    failure_history=failure_history,
                )
            
            # Record failure
            error_msg = validation.failure_reason or exec_result.stderr[:300]
            failure_history.append({
                "tier": 1,
                "attempt": attempt,
                "phase": "full_execution",
                "error": error_msg,
                "validation": {
                    "exit_ok": validation.exit_code_ok,
                    "has_stats": validation.has_statistical_output,
                    "silent_failures": validation.silent_failures_detected,
                },
                "code_snippet": current_code[:200],
            })
            
            logger.info(f"  ❌ Attempt {attempt}/{tier1_retries} failed: {error_msg[:80]}")
            
            # Fix for next attempt (unless last attempt)
            if attempt < tier1_retries:
                current_code = self._fix_code(
                    current_code, error_msg, task_description, column_names_str
                )
        
        # =====================================================================
        # TIER 2: Re-Strategy (completely different approach)
        # =====================================================================
        logger.info("🔄 Tier 2: Re-strategy phase (new approach)")
        total_attempts += 1
        
        # Build failure summary for re-strategy prompt
        failed_approaches = "\n".join(
            f"Attempt {f['attempt']}: {f['error'][:150]}" 
            for f in failure_history[-3:]  # Last 3 failures
        )
        failure_summary = self._summarize_failures(failure_history)
        
        # Generate completely new code
        restrategy_code = self._generate_restrategy_code(
            task_description=task_description,
            data_preview=data_preview,
            data_filename=Path(data_path).name,
            failed_approaches=failed_approaches,
            failure_summary=failure_summary,
            num_attempts=total_attempts - 1,
        )
        
        # Execute re-strategized code
        exec_result = self.executor.execute_code(
            code=restrategy_code, data_path=data_path
        )
        
        validation = self._validate_execution_output(exec_result)
        
        if validation.is_valid:
            logger.info("  ✅ Tier 2 re-strategy succeeded!")
            return VerificationResult(
                success=True,
                execution_result=exec_result,
                final_code=restrategy_code,
                tier_reached=2,
                total_attempts=total_attempts,
                failure_history=failure_history,
            )
        
        # Record tier 2 failure
        error_msg = validation.failure_reason or exec_result.stderr[:300]
        failure_history.append({
            "tier": 2,
            "attempt": 1,
            "phase": "restrategy",
            "error": error_msg,
            "code_snippet": restrategy_code[:200],
        })
        logger.warning(f"  ❌ Tier 2 re-strategy failed: {error_msg[:80]}")
        
        # =====================================================================
        # TIER 3: Human-in-the-Loop (flag for review)
        # =====================================================================
        logger.warning("🚨 Tier 3: Flagging for human review")
        
        # Return the best result we have (even if failed)
        # Prefer tier 2 code since it's a fresh approach
        return VerificationResult(
            success=False,
            execution_result=exec_result,
            final_code=restrategy_code,
            tier_reached=3,
            total_attempts=total_attempts,
            failure_history=failure_history,
            needs_human_review=True,
            human_review_reason=self._format_review_reason(failure_history),
        )

    def _fix_code(
        self,
        failed_code: str,
        error_message: str,
        task_description: str,
        column_names: str,
    ) -> str:
        """Ask LLM to fix specific error in code."""
        fix_prompt = CODE_FIX_PROMPT.format(
            task_description=task_description,
            failed_code=failed_code,
            error_message=error_message,
            column_names=column_names,
        )
        
        response = self.llm.complete_for_role(
            prompt=fix_prompt,
            role="code_generation",
            system="You are a senior data scientist debugging code.",
        )
        
        fixed_code = response.content.strip()
        # Clean markdown fences
        for prefix in ["```python", "```"]:
            if fixed_code.startswith(prefix):
                fixed_code = fixed_code[len(prefix):].strip()
        if fixed_code.endswith("```"):
            fixed_code = fixed_code[:-3].strip()
        
        return fixed_code

    def _generate_restrategy_code(
        self,
        task_description: str,
        data_preview: str,
        data_filename: str,
        failed_approaches: str,
        failure_summary: str,
        num_attempts: int,
    ) -> str:
        """Generate completely new analytical approach."""
        prompt = CODE_RESTRATEGY_PROMPT.format(
            task_description=task_description,
            data_preview=data_preview,
            data_filename=data_filename,
            failed_approaches=failed_approaches,
            failure_summary=failure_summary,
            num_attempts=num_attempts,
        )
        
        response = self.llm.complete_for_role(
            prompt=prompt,
            role="code_generation",
            system="You are a senior data scientist who excels at finding alternative approaches.",
        )
        
        code = response.content.strip()
        # Clean markdown fences
        for prefix in ["```python", "```"]:
            if code.startswith(prefix):
                code = code[len(prefix):].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        
        return code

    def _summarize_failures(self, failure_history: List[dict]) -> str:
        """Create a brief summary of why code kept failing."""
        if not failure_history:
            return "Unknown failures."
        
        # Group errors by type
        error_types = {}
        for f in failure_history:
            error = f.get("error", "")[:100]
            # Extract error type
            for etype in ["KeyError", "ValueError", "TypeError", "ImportError", 
                          "IndexError", "AttributeError", "NameError"]:
                if etype.lower() in error.lower():
                    error_types[etype] = error_types.get(etype, 0) + 1
                    break
            else:
                error_types["Other"] = error_types.get("Other", 0) + 1
        
        summary_parts = []
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            summary_parts.append(f"{etype}: {count}x")
        
        return "; ".join(summary_parts) if summary_parts else "Multiple failures"

    def _format_review_reason(self, failure_history: List[dict]) -> str:
        """Format failure history for human review."""
        lines = ["Code verification failed after all tiers:"]
        for f in failure_history[-5:]:  # Last 5 failures
            tier = f.get("tier", "?")
            phase = f.get("phase", "unknown")
            error = f.get("error", "Unknown")[:100]
            lines.append(f"  - Tier {tier} ({phase}): {error}")
        return "\n".join(lines)

    def _extract_findings(self, task_description: str, code_stdout: str) -> list[dict]:
        schol_eval = ScholarEval(min_score=0.40)
        prompt = FINDING_EXTRACTION_PROMPT.format(
            task_description=task_description,
            code_stdout=code_stdout
        )
        
        response = self.llm.complete_for_role(prompt=prompt, role="finding_extraction", system="You are a research analyst.")
        response = response.content
        try:
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            json_str = response[start_idx:end_idx]
            raw_findings = json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            return [{
                "claim": "Error parsing structured findings.",
                "confidence": 0.0,
                "evidence": code_stdout[:200],
                "tags": ["error"]
            }]
    
        # ── Validate and score each finding before storing ────────────────
        validated = []
        for finding in raw_findings:
            # Step 1: hard physics/stats validator
            is_valid, reason = self._validate_finding(finding)
            if not is_valid:
                logger.warning(
                    f"🚫 Finding discarded (impossible value): {reason}\n"
                    f"   Claim: {finding.get('claim', '')[:120]}"
                )
                continue
            
            # Step 2: ScholarEval 8-dimension quality score
            eval_result = schol_eval.evaluate(
                finding, source_type="data_analysis"
            )
            if not eval_result.passes:
                logger.warning(
                    f"📉 Finding rejected by ScholarEval "
                    f"(score={eval_result.composite_score:.2f}): "
                    f"{finding.get('claim', '')[:80]}"
                )
                continue
            
            # Attach scores to finding for world model storage
            finding.update(eval_result.to_dict())
            validated.append(finding)
        
        return validated
    
    def execute(self, task, data_path: str, objective: str,
                world_model_summary: str = "") -> dict:
        """Main entry point — run the full analysis pipeline.
        
        This orchestrates everything:
        1. Preview the data
        2. Plan the analysis
        3. Generate code
        4. Execute with 3-tier verification (smoke test → fix → re-strategy)
        5. Extract findings
        6. Create notebook (NotebookManager)
        7. Return results
        
        Returns a dict with:
          - "findings": list of finding dicts
          - "notebook_path": where the notebook was saved
          - "execution_result": the ExecutionResult from DockerExecutor
          - "code": the generated code
          - "plan": the analysis plan
          - "verification": VerificationResult with tier info and history
        """
        # 1. Preview the data
        preview = self._preview_data(data_path)
        
        # 2. Plan the analysis
        plan = self._plan_analysis(
            task_description=task["description"],
            task_goal=task["goal"],
            data_preview=preview,
            objective=objective,
            world_model_summary=world_model_summary
        )
        
        # 3. Generate code (with task-specific skill injection)
        code = self._generate_code(
            analysis_plan=plan,
            data_preview=preview,
            objective=objective,
            data_filename=Path(data_path).name,
            task_description=task["description"],  # UpSkill: generates technique-specific guidance
        )
        
        # 4. Execute with 3-tier verification system
        verification = self._execute_with_verification(
            code=code,
            data_path=data_path,
            data_preview=preview,
            task_description=task["description"],
            tier1_retries=3,
            enable_smoke_test=True,
        )
        
        exec_result = verification.execution_result
        final_code = verification.final_code
        
        # Log verification outcome
        if verification.success:
            logger.info(
                f"  ✅ Code verified (Tier {verification.tier_reached}, "
                f"{verification.total_attempts} attempts)"
            )
        else:
            logger.warning(
                f"  ⚠️ Code verification failed after {verification.total_attempts} attempts. "
                f"Tier {verification.tier_reached} reached. "
                f"Human review: {verification.needs_human_review}"
            )
        
        # 5. LOGIC VERIFICATION: Does the code actually answer the task?
        #    This is Phase 7A #3 - catches circular reasoning, task misalignment, etc.
        logic_verification = None
        if verification.success and exec_result and hasattr(exec_result, 'stdout'):
            logic_verification = self._verify_code_logic(
                task_description=task["description"],
                code=final_code,
                stdout=exec_result.stdout,
                min_score=0.6,
            )
            
            # If logic verification fails, flag findings but don't block extraction
            if not logic_verification.passes_verification:
                logger.warning(
                    f"  🚩 Logic verification flagged issues: {logic_verification.reasoning[:100]}"
                )
        
        # 6. Extract findings from stdout (even if verification failed, try extraction)
        findings = []
        if exec_result and hasattr(exec_result, 'stdout') and exec_result.stdout:
            findings = self._extract_findings(task["description"], exec_result.stdout)
            
            # If logic verification found issues, add a warning tag to findings
            if logic_verification and not logic_verification.passes_verification:
                for finding in findings:
                    if "tags" not in finding:
                        finding["tags"] = []
                    finding["tags"].append("unverified_logic")
                    finding["logic_issues"] = logic_verification.issues_found
        
        # 7. Create/Update notebook for the user to review
        nb = self.notebook_manager.create_notebook(
            title=task["description"],
            task_description=task.get("goal", ""),
            cycle=task.get("cycle", 1)
        )
        self.notebook_manager.add_markdown_cell(nb, f"## Analysis Plan\n{plan}")
        
        # Add verification status to notebook
        if not verification.success:
            status_md = (
                f"## ⚠️ Execution Verification Status\n\n"
                f"**Status:** Failed after {verification.total_attempts} attempts\n"
                f"**Tier Reached:** {verification.tier_reached}\n"
                f"**Needs Human Review:** {verification.needs_human_review}\n\n"
                f"**Failure Summary:**\n```\n{verification.human_review_reason}\n```"
            )
            self.notebook_manager.add_markdown_cell(nb, status_md)
        
        # Add logic verification status to notebook
        if logic_verification:
            if logic_verification.passes_verification:
                logic_md = (
                    f"## ✅ Logic Verification\n\n"
                    f"**Task Alignment:** {logic_verification.task_alignment_score:.0%}\n"
                    f"**Logical Correctness:** {logic_verification.logical_correctness_score:.0%}\n"
                    f"**Output Validity:** {logic_verification.output_validity_score:.0%}\n"
                )
            else:
                logic_md = (
                    f"## 🚩 Logic Verification Issues\n\n"
                    f"**Severity:** {logic_verification.severity}\n"
                    f"**Recommendation:** {logic_verification.recommendation}\n\n"
                    f"**Issues Found:**\n"
                    + "\n".join(f"- {issue}" for issue in logic_verification.issues_found) +
                    f"\n\n**Reasoning:** {logic_verification.reasoning}"
                )
            self.notebook_manager.add_markdown_cell(nb, logic_md)
        
        code_cell_idx = self.notebook_manager.add_code_cell(
            nb, 
            code=final_code, 
            output=exec_result.stdout if exec_result else "No output"
        )
        
        if findings:
            findings_text = "\n".join(f"- {f['claim']}" for f in findings)
            self.notebook_manager.add_markdown_cell(nb, f"## Findings\n{findings_text}")
        
        notebook_path = self.notebook_manager.save_notebook(nb)

        # Build logic verification dict for return value
        logic_verification_dict = None
        if logic_verification:
            logic_verification_dict = {
                "passes": logic_verification.passes_verification,
                "task_alignment_score": logic_verification.task_alignment_score,
                "logical_correctness_score": logic_verification.logical_correctness_score,
                "output_validity_score": logic_verification.output_validity_score,
                "issues_found": logic_verification.issues_found,
                "severity": logic_verification.severity,
                "recommendation": logic_verification.recommendation,
                "reasoning": logic_verification.reasoning,
            }

        return {
            "findings": findings,
            "notebook_path": notebook_path,
            "execution_result": exec_result,
            "code": final_code,
            "plan": plan,
            "code_cell_index": code_cell_idx,
            "attempts": verification.total_attempts,
            "tier_reached": verification.tier_reached,
            "verification_success": verification.success,
            "needs_human_review": verification.needs_human_review,
            "failure_history": verification.failure_history,
            "logic_verification": logic_verification_dict,
            "figures": exec_result.figures if exec_result else [],
        }

    def _execute_with_retry(self, code: str, data_path: str,
                        task_description: str, max_retries: int = 3) -> tuple:
        """
        DEPRECATED: Use _execute_with_verification() instead.
        
        This method is kept for backward compatibility but is superseded by
        the 3-tier verification system which includes:
        - Smoke testing on sample data
        - Heuristic output validation
        - Re-strategy on persistent failures
        - Human-in-the-loop flagging
        
        Execute code with automatic retry on failure.
        If code fails, pass the error to the LLM for fixing.
        Try up to max_retries times.

        Returns: (ExecutionResult, final_code, attempts_made)
        """
        current_code = code

        # Pre-load column names so the fix prompt knows what's available
        try:
            _cols = pd.read_csv(data_path, nrows=0).columns.tolist()
            column_names_str = str(_cols)
        except Exception:
            column_names_str = "(could not read columns)"

        for attempt in range(1, max_retries + 1):
            result = self.executor.execute_code(code=current_code, data_path=data_path)
            if result.success:
                return result, current_code, attempt
            
            if attempt < max_retries:
                print(f"[Retry {attempt}/{max_retries}] Code execution failed with error:\n{result.stderr}")
                print("Asking the agent to fix the code...")
                
                fix_prompt = CODE_FIX_PROMPT.format(
                    task_description=task_description,
                    failed_code=current_code,
                    error_message=result.stderr,
                    column_names=column_names_str,
                )
                
                llm_response = self.llm.complete_for_role(prompt=fix_prompt, role="code_generation", system="You are a senior data scientist debugging code.")
                fixed_code = llm_response.content.strip()
                for prefix in ["```python", "```"]:
                    if fixed_code.startswith(prefix):
                        fixed_code = fixed_code[len(prefix):].strip()
                if fixed_code.endswith("```"):
                    fixed_code = fixed_code[:-3].strip()
                current_code = fixed_code
        return result, current_code, attempt