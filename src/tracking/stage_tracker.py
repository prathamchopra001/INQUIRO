# -*- coding: utf-8 -*-
"""
Structured JSONL stage tracker for INQUIRO.

Writes one JSON line per workflow stage to a .jsonl file alongside
the standard log. Enables programmatic analysis of research runs.

Each entry:
{
    "timestamp": "2026-02-19T18:32:45.123",
    "run_id": "run_20260219_...",
    "stage": "cycle",
    "substage": "task_execution",
    "status": "completed",
    "cycle": 2,
    "duration_ms": 4521,
    "metadata": {...}
}
"""

import json
import time
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class StageTracker:
    """
    Writes structured JSONL entries for each workflow stage.

    Usage:
        tracker = StageTracker(run_id="run_20260219_...", output_dir="./outputs/run_...")

        # As context manager (auto-records start/end/duration)
        with tracker.track("cycle", cycle=1):
            execute_cycle()

        # With metadata
        with tracker.track("task", substage="data_analysis", task_id="t_123"):
            run_task()

        # Manual event
        tracker.event("finding_added", metadata={"claim": "...", "confidence": 0.9})
    """

    def __init__(self, run_id: str, output_dir: str):
        self.run_id = run_id
        self._output_path = Path(output_dir) / "stages.jsonl"
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stage_stack: list = []

        # Write header entry
        self._write({
            "stage": "run_start",
            "status": "started",
            "metadata": {"run_id": run_id, "output_dir": str(output_dir)},
        })

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    @contextmanager
    def track(self, stage: str, substage: str = None, cycle: int = None, **metadata):
        """
        Context manager that records start, end, and duration of a stage.

        Args:
            stage:    Stage name (e.g., "cycle", "task", "literature_search")
            substage: Optional sub-classification
            cycle:    Cycle number if applicable
            **metadata: Any additional key-value pairs to record
        """
        start_time = time.time()
        parent = self._stage_stack[-1] if self._stage_stack else None
        self._stage_stack.append(stage)

        self._write({
            "stage":    stage,
            "substage": substage,
            "status":   "started",
            "cycle":    cycle,
            "parent":   parent,
            "metadata": metadata or None,
        })

        error_info = None
        try:
            yield
        except Exception as e:
            error_info = {"type": type(e).__name__, "message": str(e)[:200]}
            raise
        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            self._stage_stack.pop() if self._stage_stack else None

            self._write({
                "stage":       stage,
                "substage":    substage,
                "status":      "failed" if error_info else "completed",
                "cycle":       cycle,
                "parent":      parent,
                "duration_ms": duration_ms,
                "error":       error_info,
                "metadata":    metadata or None,
            })

    # =========================================================================
    # MANUAL EVENT
    # =========================================================================

    def event(self, stage: str, cycle: int = None, metadata: dict = None):
        """Record a one-shot event (no duration tracking)."""
        self._write({
            "stage":    stage,
            "status":   "event",
            "cycle":    cycle,
            "parent":   self._stage_stack[-1] if self._stage_stack else None,
            "metadata": metadata,
        })

    def finding_added(self, claim: str, confidence: float,
                      source_type: str, cycle: int):
        """Convenience method for finding events."""
        self.event("finding_added", cycle=cycle, metadata={
            "claim_preview": claim[:80],
            "confidence":    round(confidence, 3),
            "source_type":   source_type,
        })

    def cycle_summary(self, cycle: int, findings: int,
                      relationships: int, tasks: int):
        """Record end-of-cycle summary."""
        self.event("cycle_summary", cycle=cycle, metadata={
            "findings_added":      findings,
            "relationships_added": relationships,
            "tasks_run":           tasks,
        })

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _write(self, data: dict):
        """Write one JSONL entry."""
        entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "run_id":    self.run_id,
        }
        # Remove None values for cleaner output
        entry.update({k: v for k, v in data.items() if v is not None})

        try:
            with open(self._output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"StageTracker write error: {e}")

    def get_output_path(self) -> str:
        return str(self._output_path)