# -*- coding: utf-8 -*-
"""
Synthetic Dataset Generator — creates research-appropriate datasets
when no user dataset is provided.

Problem it solves:
  INQUIRO's value comes from the interplay between data analysis and
  literature search. Without a dataset, the system can only do
  literature reviews. This generator creates a realistic synthetic
  dataset so the full research pipeline can run.

Design:
  1. LLM reads the research objective and designs a dataset schema
  2. LLM generates Python code to create the dataset with numpy/pandas
  3. Code executes in the Docker sandbox (safe, isolated)
  4. CSV is saved to the run's data directory
  5. Everything is flagged as synthetic in the report

Usage:
    generator = SyntheticDatasetGenerator(llm_client, executor)
    result = generator.generate(
        objective="Study the effect of X on Y...",
        output_dir="./data/run_123/"
    )
    # result.path = "./data/run_123/synthetic_dataset.csv"
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from src.utils.llm_client import LLMClient
from src.execution.docker_executor import DockerExecutor
from config.prompts.synthetic_data import (
    SCHEMA_DESIGN_PROMPT,
    CODE_GENERATION_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataResult:
    """Result of synthetic dataset generation."""
    success: bool
    path: str                     # Path to generated CSV
    schema: dict                  # The designed schema
    description: str              # Human-readable description
    row_count: int
    column_count: int
    generation_code: str          # The code that created it


class SyntheticDatasetGenerator:
    """
    Generates a research-appropriate synthetic dataset from a research objective.

    Two-step process:
      1. Schema design (LLM → JSON schema)
      2. Code generation + execution (LLM → Python → Docker → CSV)
    """

    def __init__(self, llm_client: LLMClient, executor: DockerExecutor):
        self.llm = llm_client
        self.executor = executor

    def generate(
        self,
        objective: str,
        output_dir: str,
        max_retries: int = 3,
    ) -> SyntheticDataResult:
        """
        Generate a synthetic dataset for the given research objective.

        Args:
            objective:   The research question to design data for
            output_dir:  Directory to save the CSV (will be created)
            max_retries: How many times to retry code execution

        Returns:
            SyntheticDataResult with path to the CSV and metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "synthetic_dataset.csv"

        logger.info("🧪 Generating synthetic dataset for research objective...")

        # Step 1: Design the schema
        logger.info("  Step 1: Designing dataset schema...")
        schema = self._design_schema(objective)
        if not schema:
            logger.error("  Schema design failed")
            return SyntheticDataResult(
                success=False, path="", schema={},
                description="Schema design failed",
                row_count=0, column_count=0, generation_code="",
            )

        description = schema.get("description", "Synthetic dataset")
        logger.info(
            f"  Schema: {description} | "
            f"{schema.get('rows', '?')} rows, "
            f"{len(schema.get('columns', []))} columns"
        )

        # Step 2: Generate and execute code
        logger.info("  Step 2: Generating dataset creation code...")
        code = self._generate_code(schema, objective)
        if not code:
            logger.error("  Code generation failed")
            return SyntheticDataResult(
                success=False, path="", schema=schema,
                description=description,
                row_count=0, column_count=0, generation_code="",
            )

        # Step 3: Execute with retry
        logger.info("  Step 3: Executing in sandbox...")
        logger.info(f"  Output path: {output_path.absolute()}")
        result = None
        current_code = code

        for attempt in range(1, max_retries + 1):
            result = self.executor.execute_code(
                code=current_code,
                output_path=str(output_path),
            )

            # Debug: what did the executor see?
            logger.debug(f"  Executor stdout (last 300 chars): {(result.stdout or '')[-300:]}")
            logger.debug(f"  Executor figures/files found: {getattr(result, 'figures', [])}")

            if result.success:
                # Verify the CSV was actually created by checking stdout
                if "SYNTHETIC_SAVE_OK" in (result.stdout or ""):
                    logger.info(f"  ✅ Dataset generated and saved on attempt {attempt}")

                    # Give Docker Desktop a moment to flush the volume mount
                    import time
                    time.sleep(2)

                    break
                else:
                    # Code ran but may not have saved — treat as failure
                    logger.warning(
                        f"  Attempt {attempt}: code ran but CSV save not confirmed. "
                        f"Retrying with explicit save..."
                    )
                    # Append a forced save to the code
                    current_code = current_code.rstrip() + "\n".join([
                        "",
                        "",
                        "# Force save — auto-appended by INQUIRO",
                        "import os",
                        "os.makedirs('/app/outputs/', exist_ok=True)",
                        "try:",
                        "    df.to_csv('/app/outputs/synthetic_dataset.csv', index=False)",
                        "    print('SYNTHETIC_SAVE_OK')",
                        "    print('Saved', len(df), 'rows to /app/outputs/synthetic_dataset.csv')",
                        "    # Backup: print CSV to stdout in case Docker mount fails",
                        "    print('===SYNTHETIC_CSV_START===')",
                        "    print(df.to_csv(index=False))",
                        "    print('===SYNTHETIC_CSV_END===')",
                        "except Exception as e:",
                        "    print('SAVE_FAILED:', e)",
                        "",
                    ])
                    if attempt < max_retries:
                        continue
                    else:
                        # Last attempt — run the force-save version
                        result = self.executor.execute_code(
                            code=current_code,
                            output_path=str(output_path),
                        )
                        if result.success:
                            logger.info(f"  ✅ Dataset generated with forced save")
                        break

            if attempt < max_retries:
                logger.warning(
                    f"  Attempt {attempt} failed: {result.stderr[:200]}..."
                )
                current_code = self._fix_code(
                    current_code, result.stderr, objective
                )
            else:
                logger.error(
                    f"  ❌ All {max_retries} attempts failed"
                )
                
        # If code succeeded but file might not be on disk (Docker mount issue),
        # ensure the stdout backup markers exist for extraction
        if (result and result.success 
            and "SYNTHETIC_SAVE_OK" in (result.stdout or "")
            and "===SYNTHETIC_CSV_START===" not in (result.stdout or "")):
            logger.info("  Adding stdout backup for CSV extraction...")
            backup_code = current_code.rstrip() + "\n".join([
                "",
                "",
                "# Stdout backup — auto-appended by INQUIRO",
                "try:",
                "    print('===SYNTHETIC_CSV_START===')",
                "    print(df.to_csv(index=False))",
                "    print('===SYNTHETIC_CSV_END===')",
                "except Exception as e:",
                "    print('STDOUT_BACKUP_FAILED:', e)",
                "",
            ])
            backup_result = self.executor.execute_code(
                code=backup_code,
                output_path=str(output_path),
            )
            if backup_result.success:
                result = backup_result  # Use this result for stdout extraction

        # Step 4: Find the generated CSV
        # Search broadly — the LLM might have saved it with a different name
        # or in a subdirectory.
        found_csv = None
        if result and result.success:
            # Debug: list what's actually in the output directory
            logger.info(f"  Checking for CSV at: {csv_path}")
            try:
                dir_contents = list(output_path.iterdir())
                logger.info(f"  Directory contents: {[str(f.name) for f in dir_contents]}")
            except Exception as e:
                logger.warning(f"  Could not list directory: {e}")

            for candidate in [
                csv_path,                                    # expected path
                output_path / "figures" / "synthetic_dataset.csv",  # common mistake
            ]:
                if candidate.exists():
                    found_csv = candidate
                    break

            # Fallback: find any CSV in the output tree
            if not found_csv:
                for f in output_path.rglob("*.csv"):
                    found_csv = f
                    break

            # Last resort: extract CSV from stdout if Docker mount failed
            # The LLM code prints the dataframe — we can reconstruct from stdout
            if not found_csv and result.stdout:
                logger.warning(
                    "  ⚠️ Docker volume mount failed — extracting CSV from stdout"
                )
                found_csv = self._extract_csv_from_stdout(
                    result.stdout, csv_path
                )
                # Nuclear option: if Docker won't save files, run the code
            # natively in Python (no sandbox, but better than no data)
            if not found_csv and result.stdout:
                logger.warning(
                    "  ⚠️ Docker mount and stdout extraction both failed — "
                    "attempting native Python execution"
                )
                found_csv = self._execute_natively(current_code, csv_path)

        if found_csv:
            # Move to canonical location if needed
            if found_csv != csv_path:
                found_csv.rename(csv_path)
                logger.info(f"  Moved dataset from {found_csv} to {csv_path}")

            import pandas as pd
            try:
                df = pd.read_csv(csv_path)
                row_count = len(df)
                col_count = len(df.columns)
            except Exception:
                row_count = 0
                col_count = 0

            logger.info(
                f"  📊 Synthetic dataset: {csv_path} "
                f"({row_count} rows × {col_count} columns)"
            )

            return SyntheticDataResult(
                success=True,
                path=str(csv_path),
                schema=schema,
                description=description,
                row_count=row_count,
                column_count=col_count,
                generation_code=current_code,
            )

        logger.error("  Dataset CSV not found after execution")
        return SyntheticDataResult(
            success=False, path="", schema=schema,
            description=description,
            row_count=0, column_count=0,
            generation_code=current_code,
        )

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _extract_csv_from_stdout(self, stdout: str, csv_path: Path) -> Optional[Path]:
        """
        Extract CSV data from stdout when Docker volume mount fails.

        The generated code prints the CSV between markers:
          ===SYNTHETIC_CSV_START===
          col1,col2,...
          val1,val2,...
          ===SYNTHETIC_CSV_END===

        This method extracts that text and saves it to disk.
        """
        start_marker = "===SYNTHETIC_CSV_START==="
        end_marker = "===SYNTHETIC_CSV_END==="

        start = stdout.find(start_marker)
        end = stdout.find(end_marker)

        if start < 0 or end <= start:
            logger.warning("  No CSV markers found in stdout")
            return None

        csv_text = stdout[start + len(start_marker):end].strip()
        if not csv_text or "," not in csv_text:
            logger.warning("  CSV marker found but content is empty or invalid")
            return None

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(csv_text, encoding="utf-8")
        logger.info(
            f"  📦 Extracted CSV from stdout → {csv_path} "
            f"({len(csv_text)} bytes)"
        )
        return csv_path

    def _design_schema(self, objective: str) -> Optional[dict]:
        """Ask the LLM to design a dataset schema."""
        prompt = SCHEMA_DESIGN_PROMPT.format(objective=objective)
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="schema_design",
                system=(
                    "You are an expert research data scientist. "
                    "Respond only with valid JSON."
                ),
            )
            text = response.content.strip()

            # Extract JSON from response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])

            logger.warning("No valid JSON found in schema response")
            return None

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Schema design failed: {e}")
            return None

    def _generate_code(self, schema: dict, objective: str) -> Optional[str]:
        """Ask the LLM to write dataset generation code."""
        prompt = CODE_GENERATION_PROMPT.format(
            schema_json=json.dumps(schema, indent=2),
            objective=objective,
        )
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="code_generation",
                system="You are a senior data scientist. Write only Python code.",
            )
            code = response.content.strip()

            # Strip markdown fences
            for prefix in ["```python", "```"]:
                if code.startswith(prefix):
                    code = code[len(prefix):].strip()
            if code.endswith("```"):
                code = code[:-3].strip()

            return code

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None

    def _fix_code(self, failed_code: str, error: str, objective: str) -> str:
        """Ask the LLM to fix broken generation code."""
        prompt = (
            f"Fix this Python code that was supposed to generate a "
            f"synthetic dataset.\n\n"
            f"## Failed Code\n{failed_code}\n\n"
            f"## Error\n{error[:500]}\n\n"
            f"## Rules\n"
            f"- Use ONLY pandas, numpy, scipy.stats\n"
            f"- Save to /app/outputs/synthetic_dataset.csv\n"
            f"- os.makedirs('/app/outputs/', exist_ok=True)\n"
            f"- Use np.random.seed(42)\n"
            f"- First line: # -*- coding: utf-8 -*-\n"
            f"- NEVER use 'lambda' as a variable name — it's a Python reserved word. Use 'lam' or 'rate' instead.\n"
            f"- After saving the CSV, ALWAYS print: print('SYNTHETIC_SAVE_OK')\n"
            f"- The df.to_csv() call MUST be present. Do not remove it.\n\n"
            f"Return ONLY corrected Python code. No markdown."
        )
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="code_fix",
                system="You are a senior data scientist debugging code.",
            )
            code = response.content.strip()
            for prefix in ["```python", "```"]:
                if code.startswith(prefix):
                    code = code[len(prefix):].strip()
            if code.endswith("```"):
                code = code[:-3].strip()
            return code
        except Exception:
            return failed_code  # Return original if fix fails
        
    def _execute_natively(self, code: str, csv_path: Path) -> Optional[Path]:
        """
        Last resort: run the dataset generation code directly in Python.
        No Docker sandbox, but the code only uses numpy/pandas/scipy.
        """
        import subprocess
        import sys
        import tempfile

        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Rewrite /app/outputs paths to actual target
            target_dir = str(csv_path.parent).replace("\\", "/")
            target_file = str(csv_path).replace("\\", "/")

            modified_code = code.replace(
                "/app/outputs/synthetic_dataset.csv", target_file
            )
            modified_code = modified_code.replace("/app/outputs/", target_dir + "/")
            modified_code = modified_code.replace("/app/outputs", target_dir)

            logger.info(f"  Native: target CSV = {target_file}")

            # Write to temp file and run
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(modified_code)
                script_path = f.name

            logger.info(f"  Native: running {script_path}...")
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=target_dir,
            )

            Path(script_path).unlink(missing_ok=True)

            # Log what happened
            logger.info(f"  Native: exit code = {result.returncode}")
            if result.stdout:
                logger.info(f"  Native stdout (last 200): {result.stdout[-200:]}")
            if result.stderr:
                logger.warning(f"  Native stderr (last 300): {result.stderr[-300:]}")

            # Check exact path first
            if csv_path.exists() and csv_path.stat().st_size > 100:
                logger.info(f"  ✅ Native execution saved CSV: {csv_path}")
                return csv_path

            # Search for any CSV the script might have created
            for f in csv_path.parent.rglob("*.csv"):
                if f.stat().st_size > 100:
                    logger.info(f"  ✅ Native execution: found CSV at {f}")
                    return f

            logger.warning(f"  Native execution: no CSV found in {target_dir}")
            return None

        except subprocess.TimeoutExpired:
            logger.error("  Native execution: timed out after 60s")
            return None
        except Exception as e:
            logger.error(f"  Native execution failed: {e}")
            return None
