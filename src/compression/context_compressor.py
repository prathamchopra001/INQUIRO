# -*- coding: utf-8 -*-
"""
Hierarchical 3-tier context compression for INQUIRO.

Prevents context overflow when passing world model history
to the orchestrator across many research cycles.

Tier 1: Task-level compression
    Raw execution output → compact summary with key statistics

Tier 2: Cycle-level compression  
    Multiple task summaries → single cycle overview

Tier 3: Run-level compression
    All cycle overviews → final synthesis for orchestrator
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskSummary:
    """Compressed summary of a single task execution."""
    task_id: str
    task_description: str
    task_type: str          # "data_analysis" or "literature"
    cycle: int
    findings_count: int
    key_findings: list      # top 2-3 finding claims, truncated
    key_stats: dict         # p-values, fold changes, correlations found
    schol_eval_avg: float   # average ScholarEval score if available
    compressed_text: str    # final 1-2 sentence summary


@dataclass
class CycleSummary:
    """Compressed summary of one full research cycle."""
    cycle: int
    tasks_run: int
    findings_count: int
    relationships_count: int
    top_findings: list      # top 3 findings by confidence
    themes: list            # key topics/pathways that emerged
    compressed_text: str    # 2-3 sentence cycle overview


@dataclass 
class RunContext:
    """
    The final compressed context passed to the orchestrator.
    
    Contains the full hierarchy:
    - High-level run summary (always included)
    - Per-cycle overviews (compressed)
    - Top findings across all cycles (lazy-loaded on demand)
    """
    objective: str
    cycles_completed: int
    total_findings: int
    total_relationships: int
    cycle_summaries: list           # list of CycleSummary
    top_findings_global: list       # top 5 findings across all cycles
    open_questions: list            # unanswered questions from orchestrator
    compressed_summary: str         # final ~500 token context string


class ContextCompressor:
    """
    3-tier hierarchical context compressor.
    
    Usage:
        compressor = ContextCompressor(llm_client)
        
        # After each task completes:
        task_summary = compressor.compress_task(task, result)
        
        # After each cycle completes:
        cycle_summary = compressor.compress_cycle(cycle_num, task_summaries, findings)
        
        # Before passing to orchestrator:
        run_context = compressor.build_run_context(objective, cycle_summaries, world_model)
        orchestrator_prompt = run_context.compressed_summary
    """

    # Maximum tokens for each tier output
    TIER1_MAX_CHARS = 300    # per task summary
    TIER2_MAX_CHARS = 500    # per cycle summary
    TIER3_MAX_CHARS = 1500   # final run context

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional LLM client for intelligent compression.
                        If None, falls back to rule-based compression.
        """
        self.llm = llm_client
        self._task_summaries: dict[int, list] = {}   # cycle → [TaskSummary]
        self._cycle_summaries: list = []

    # =========================================================================
    # TIER 1: Task-level compression
    # =========================================================================

    def compress_task(
        self,
        task: dict,
        result: Optional[dict],
    ) -> TaskSummary:
        """
        Compress a single task execution into a compact summary.

        Args:
            task:   Task dict with description, goal, cycle, type
            result: Agent result dict with findings, notebook_path, etc.
        
        Returns:
            TaskSummary with key statistics extracted
        """
        task_id = task.get("id", "unknown")
        description = task.get("description", "")
        task_type = task.get("type", "data_analysis")
        cycle = task.get("cycle", 0)

        if result is None:
            return TaskSummary(
                task_id=task_id,
                task_description=description[:80],
                task_type=task_type,
                cycle=cycle,
                findings_count=0,
                key_findings=[],
                key_stats={},
                schol_eval_avg=0.0,
                compressed_text=f"Task failed: {description[:60]}"
            )

        findings = result.get("findings", [])
        findings_count = len(findings)

        # Extract key findings (top 3 by confidence)
        sorted_findings = sorted(
            findings,
            key=lambda f: f.get("confidence", 0),
            reverse=True
        )
        key_findings = [
            f.get("claim", "")[:100]
            for f in sorted_findings[:3]
        ]

        # Extract key statistics from findings
        key_stats = self._extract_key_stats(findings)

        # Compute average ScholarEval score
        schol_scores = [
            f.get("schol_eval", {}).get("composite_score", 0)
            for f in findings
            if "schol_eval" in f
        ]
        schol_eval_avg = sum(schol_scores) / len(schol_scores) if schol_scores else 0.0

        # Build compressed text
        compressed_text = self._compress_task_text(
            description, findings_count, key_findings, key_stats, task_type
        )

        summary = TaskSummary(
            task_id=task_id,
            task_description=description[:80],
            task_type=task_type,
            cycle=cycle,
            findings_count=findings_count,
            key_findings=key_findings,
            key_stats=key_stats,
            schol_eval_avg=round(schol_eval_avg, 2),
            compressed_text=compressed_text,
        )

        # Store for cycle-level compression
        if cycle not in self._task_summaries:
            self._task_summaries[cycle] = []
        self._task_summaries[cycle].append(summary)

        return summary

    def _extract_key_stats(self, findings: list) -> dict:
        """Extract the most important statistics from findings."""
        stats = {
            "min_pvalue": None,
            "max_fold_change": None,
            "max_correlation": None,
            "significant_count": 0,
        }

        for f in findings:
            text = f.get("claim", "") + " " + f.get("evidence", "")

            # Extract p-values
            p_matches = re.findall(
                r'p[\s=<>]*([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)',
                text.lower()
            )
            for p in p_matches:
                try:
                    val = float(p)
                    if 0 < val <= 1:
                        if stats["min_pvalue"] is None or val < stats["min_pvalue"]:
                            stats["min_pvalue"] = round(val, 6)
                        if val < 0.05:
                            stats["significant_count"] += 1
                except ValueError:
                    pass

            # Extract fold changes
            fc_matches = re.findall(r'(\d+\.?\d*)[- ]fold', text.lower())
            for fc in fc_matches:
                try:
                    val = float(fc)
                    if stats["max_fold_change"] is None or val > stats["max_fold_change"]:
                        stats["max_fold_change"] = round(val, 2)
                except ValueError:
                    pass

            # Extract correlations
            r_matches = re.findall(r'r\s*=\s*([0-9.]+)', text.lower())
            for r in r_matches:
                try:
                    val = abs(float(r))
                    if val <= 1.0:
                        if stats["max_correlation"] is None or val > stats["max_correlation"]:
                            stats["max_correlation"] = round(val, 3)
                except ValueError:
                    pass

        # Remove None values for cleaner output
        return {k: v for k, v in stats.items() if v is not None}

    def _compress_task_text(
        self,
        description: str,
        findings_count: int,
        key_findings: list,
        key_stats: dict,
        task_type: str,
    ) -> str:
        """Build a compact 1-2 sentence task summary."""
        if findings_count == 0:
            return f"{task_type.replace('_', ' ').title()}: '{description[:60]}' → no findings."

        stats_str = ""
        if key_stats.get("min_pvalue"):
            stats_str += f" Best p={key_stats['min_pvalue']:.2e}."
        if key_stats.get("max_fold_change"):
            stats_str += f" Max fold-change={key_stats['max_fold_change']}x."
        if key_stats.get("max_correlation"):
            stats_str += f" Max r={key_stats['max_correlation']}."

        top = key_findings[0] if key_findings else ""
        return (
            f"{task_type.replace('_', ' ').title()} ({findings_count} findings):"
            f" '{description[:50]}...'  Top: {top[:80]}.{stats_str}"
        )[:self.TIER1_MAX_CHARS]

    # =========================================================================
    # TIER 2: Cycle-level compression
    # =========================================================================

    def compress_cycle(
        self,
        cycle: int,
        findings: list,
        relationships_count: int,
    ) -> CycleSummary:
        """
        Compress all task summaries from a cycle into one overview.

        Args:
            cycle:                Cycle number
            findings:             All findings added this cycle
            relationships_count:  Relationships created this cycle
        
        Returns:
            CycleSummary with key themes and top findings
        """
        task_summaries = self._task_summaries.get(cycle, [])
        tasks_run = len(task_summaries)

        # Get top findings by confidence
        sorted_findings = sorted(
            findings,
            key=lambda f: f.get("confidence", 0),
            reverse=True
        )
        top_findings = [
            {
                "claim": f.get("claim", "")[:100],
                "confidence": f.get("confidence", 0),
                "source": f.get("finding_type", "unknown"),
            }
            for f in sorted_findings[:3]
        ]

        # Extract themes (key terms that appear frequently)
        themes = self._extract_themes(findings)

        # Build compressed cycle text
        compressed_text = self._compress_cycle_text(
            cycle, tasks_run, findings, relationships_count,
            top_findings, themes
        )

        summary = CycleSummary(
            cycle=cycle,
            tasks_run=tasks_run,
            findings_count=len(findings),
            relationships_count=relationships_count,
            top_findings=top_findings,
            themes=themes,
            compressed_text=compressed_text,
        )

        self._cycle_summaries.append(summary)
        return summary

    def _extract_themes(self, findings: list) -> list:
        """Extract recurring themes from findings using term frequency."""
        term_counts = {}
        stopwords = {
            "the", "a", "an", "and", "or", "in", "of", "to", "is",
            "was", "are", "were", "with", "between", "that", "this",
            "for", "by", "from", "has", "have", "been", "on", "at",
            "group", "groups", "sample", "samples", "analysis", "data"
        }

        for f in findings:
            text = f.get("claim", "") + " " + f.get("evidence", "")
            words = re.findall(r'\b[a-z][a-z_]{3,}\b', text.lower())
            for word in words:
                if word not in stopwords:
                    term_counts[word] = term_counts.get(word, 0) + 1

        # Return top 5 themes
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        return [term for term, count in sorted_terms[:5] if count >= 2]

    def _compress_cycle_text(
        self,
        cycle: int,
        tasks_run: int,
        findings: list,
        relationships_count: int,
        top_findings: list,
        themes: list,
    ) -> str:
        """Build a compact 2-3 sentence cycle summary."""
        themes_str = ", ".join(themes[:3]) if themes else "no clear themes"
        top_claim = top_findings[0]["claim"] if top_findings else "no significant findings"

        return (
            f"Cycle {cycle}: {tasks_run} tasks, {len(findings)} findings, "
            f"{relationships_count} relationships. "
            f"Key themes: {themes_str}. "
            f"Top finding: {top_claim[:100]}."
        )[:self.TIER2_MAX_CHARS]

    # =========================================================================
    # TIER 3: Run-level synthesis
    # =========================================================================

    def build_run_context(
        self,
        objective: str,
        total_findings: int,
        total_relationships: int,
        top_findings_global: list,
        open_questions: Optional[list] = None,
    ) -> RunContext:
        """
        Build the final compressed context for the orchestrator.

        This is what gets passed to the orchestrator instead of the
        raw world model dump — typically 500-1500 chars vs 50,000+.

        Args:
            objective:            Research objective
            total_findings:       Total findings in world model
            total_relationships:  Total relationships in world model
            top_findings_global:  Top findings across all cycles
            open_questions:       Unanswered questions if any
        
        Returns:
            RunContext with compressed_summary ready for orchestrator
        """
        compressed_summary = self._build_final_summary(
            objective=objective,
            cycle_summaries=self._cycle_summaries,
            total_findings=total_findings,
            total_relationships=total_relationships,
            top_findings_global=top_findings_global[:5],
            open_questions=open_questions or [],
        )

        return RunContext(
            objective=objective,
            cycles_completed=len(self._cycle_summaries),
            total_findings=total_findings,
            total_relationships=total_relationships,
            cycle_summaries=self._cycle_summaries,
            top_findings_global=top_findings_global[:5],
            open_questions=open_questions or [],
            compressed_summary=compressed_summary,
        )

    def _build_final_summary(
        self,
        objective: str,
        cycle_summaries: list,
        total_findings: int,
        total_relationships: int,
        top_findings_global: list,
        open_questions: list,
    ) -> str:
        """Build the final ~1500 char context string."""
        lines = [
            f"RESEARCH OBJECTIVE: {objective[:120]}",
            f"PROGRESS: {len(cycle_summaries)} cycles | "
            f"{total_findings} findings | "
            f"{total_relationships} relationships",
            "",
        ]

        # Cycle history (compressed)
        if cycle_summaries:
            lines.append("CYCLE HISTORY:")
            for cs in cycle_summaries:
                lines.append(f"  {cs.compressed_text}")
            lines.append("")

        # Top findings globally
        if top_findings_global:
            lines.append("TOP FINDINGS:")
            for i, f in enumerate(top_findings_global[:5], 1):
                claim = f.get("claim", f.get("compressed_text", ""))[:100]
                conf = f.get("confidence", 0)
                lines.append(f"  {i}. [{conf:.0%}] {claim}")
            lines.append("")

        # Open questions
        if open_questions:
            lines.append("OPEN QUESTIONS:")
            for q in open_questions[:3]:
                lines.append(f"  • {q[:80]}")

        result = "\n".join(lines)

        # Hard truncation at TIER3_MAX_CHARS
        if len(result) > self.TIER3_MAX_CHARS:
            result = result[:self.TIER3_MAX_CHARS] + "\n[...truncated]"

        return result

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_compression_stats(self) -> dict:
        """Return compression statistics for logging."""
        total_tasks = sum(
            len(tasks) for tasks in self._task_summaries.values()
        )
        return {
            "cycles_compressed": len(self._cycle_summaries),
            "tasks_compressed": total_tasks,
            "cycle_summaries": [
                {
                    "cycle": cs.cycle,
                    "findings": cs.findings_count,
                    "themes": cs.themes,
                }
                for cs in self._cycle_summaries
            ],
        }