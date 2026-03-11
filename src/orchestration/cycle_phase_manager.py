# -*- coding: utf-8 -*-
"""
Cycle Phase Manager — enforces research progression.

Problem it solves:
  Without enforcement, the LLM orchestrator tends to repeat
  the same type of analysis every cycle (e.g., correlations).
  The cycle plan in the objective is just a suggestion it ignores.

Solution:
  Define phases with required task characteristics. Each phase
  specifies mandatory keywords that at least one task must contain.
  If the LLM doesn't generate matching tasks, a mandatory task
  is injected automatically.

Usage:
    phases = [
        CyclePhase(
            name="literature_review",
            cycles=(1, 4),
            description="Review literature on ABM pricing and RL",
            required_task_types=["literature"],
            required_keywords=["paper", "literature", "review"],
            mandatory_task={
                "type": "literature",
                "description": "Search for papers on ...",
                "goal": "Find relevant prior work",
                "priority": "high",
            },
        ),
        ...
    ]
    manager = CyclePhaseManager(phases)
    instruction = manager.get_phase_instruction(cycle=8)
    validated = manager.enforce_phase(cycle=8, tasks=generated_tasks)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CyclePhase:
    """Definition of a research phase spanning one or more cycles."""

    name: str
    cycles: tuple[int, int]          # (start_cycle, end_cycle) inclusive
    description: str                  # What this phase is about
    required_task_types: list[str]    # At least one task must be this type
    required_keywords: list[str]      # At least one task must contain one of these
    mandatory_task: dict              # Injected if no task matches requirements
    prompt_instruction: str = ""      # Extra instruction for the LLM prompt

    def covers_cycle(self, cycle: int) -> bool:
        return self.cycles[0] <= cycle <= self.cycles[1]


class CyclePhaseManager:
    """
    Manages research phase progression and enforces task requirements.

    Think of it as a research advisor who says:
    "We're in the implementation phase now — I need to see at least
     one task that actually builds something, not more correlations."
    """

    def __init__(self, phases: list[CyclePhase] = None, has_dataset: bool = True):
        self.phases = phases or []
        self.has_dataset = has_dataset
        self._phase_map: dict[int, CyclePhase] = {}
        self._build_phase_map()

    def _build_phase_map(self):
        """Pre-compute which phase each cycle belongs to."""
        for phase in self.phases:
            for c in range(phase.cycles[0], phase.cycles[1] + 1):
                self._phase_map[c] = phase

    def get_current_phase(self, cycle: int) -> Optional[CyclePhase]:
        """Get the phase definition for a given cycle."""
        return self._phase_map.get(cycle)

    def get_phase_instruction(self, cycle: int) -> str:
        """
        Get a prompt instruction for the current phase.

        This is injected into the task generation prompt to guide
        the LLM toward appropriate task types. Adapts automatically
        when no dataset is available.
        """
        phase = self.get_current_phase(cycle)
        if not phase:
            return ""

        # Adapt task types when no dataset is available
        if self.has_dataset:
            task_types = phase.required_task_types
            extra_note = ""
        else:
            # Convert data_analysis requirements to literature
            task_types = [
                "literature_search" if t == "data_analysis" else t
                for t in phase.required_task_types
            ]
            extra_note = (
                "\n⚠️ NO DATASET AVAILABLE — all tasks must be literature searches.\n"
                "Instead of running data analysis, search for papers that REPORT "
                "the results this phase is looking for (empirical findings, "
                "benchmark comparisons, methodology descriptions).\n"
            )

        instruction = (
            f"\n[RESEARCH PHASE: {phase.name.upper()} — "
            f"Cycles {phase.cycles[0]}-{phase.cycles[1]}]\n"
            f"{phase.description}\n"
        )

        if phase.prompt_instruction:
            instruction += f"\nMANDATORY: {phase.prompt_instruction}\n"

        instruction += extra_note

        instruction += (
            f"\nAt least one task in this cycle MUST be of type "
            f"{'/'.join(task_types)} and MUST relate to: "
            f"{', '.join(phase.required_keywords)}.\n"
            f"Do NOT generate only correlation or descriptive statistics tasks "
            f"during this phase.\n"
        )

        return instruction

    def enforce_phase(self, cycle: int, tasks: list) -> list:
        """
        Validate generated tasks against phase requirements.

        If no task matches the phase requirements, inject the
        mandatory task. This is the hard enforcement layer.

        Args:
            cycle: Current cycle number
            tasks: List of Task objects from the orchestrator

        Returns:
            List of Task objects, possibly with a mandatory task added
        """
        phase = self.get_current_phase(cycle)
        if not phase:
            return tasks  # No phase defined for this cycle

        # Adapt required types when no dataset
        if self.has_dataset:
            effective_types = phase.required_task_types
        else:
            # In literature-only mode, literature tasks satisfy data requirements
            effective_types = list(set(
                "literature_search" if t == "data_analysis" else t
                for t in phase.required_task_types
            )) + phase.required_task_types  # accept either

        # Check if any task matches the requirements
        has_matching_task = False
        for task in tasks:
            desc_lower = task.description.lower() if hasattr(task, 'description') else ""
            goal_lower = task.goal.lower() if hasattr(task, 'goal') else ""
            text = desc_lower + " " + goal_lower

            # Check task type
            task_type_str = (
                task.task_type.value
                if hasattr(task, 'task_type') and hasattr(task.task_type, 'value')
                else str(getattr(task, 'task_type', ''))
            )

            type_match = any(
                req_type in task_type_str
                for req_type in effective_types
            )

            # Check keywords
            keyword_match = any(
                kw.lower() in text
                for kw in phase.required_keywords
            )

            # Using OR here intentionally — LLM task descriptions don't always
            # contain exact keywords, so matching either type OR keyword is
            # sufficient to confirm the task is phase-appropriate.
            if type_match or keyword_match:
                has_matching_task = True
                break

        if has_matching_task:
            logger.info(
                f"  ✅ Phase '{phase.name}': task requirements met"
            )
            return tasks

        # No matching task found — inject mandatory task
        logger.warning(
            f"  ⚠️ Phase '{phase.name}': no matching task found. "
            f"Injecting mandatory task."
        )

        from src.world_model.models import Task, TaskType, TaskStatus

        mandatory = phase.mandatory_task
        task_type_str = mandatory.get("type", "data_analysis").lower()

        # Convert data tasks to literature when no dataset
        if not self.has_dataset and "literature" not in task_type_str:
            task_type = TaskType.LITERATURE_SEARCH
            description = (
                f"Search for papers reporting results on: "
                f"{mandatory.get('description', '')}"
            )
            logger.info(
                f"  📚 No dataset — mandatory task converted to literature search"
            )
        elif "literature" in task_type_str:
            task_type = TaskType.LITERATURE_SEARCH
            description = mandatory.get("description", "")
        else:
            task_type = TaskType.DATA_ANALYSIS
            description = mandatory.get("description", "")

        mandatory_task = Task(
            task_type=task_type,
            description=description,
            goal=mandatory.get("goal", ""),
            priority=mandatory.get("priority", "high"),
            status=TaskStatus.PENDING,
            cycle=cycle,
        )

        # Replace the lowest-priority existing task
        if len(tasks) >= 3:
            # Find lowest priority task to replace
            priority_order = {"high": 3, "medium": 2, "low": 1}
            tasks.sort(
                key=lambda t: priority_order.get(
                    getattr(t, 'priority', 'medium'), 2
                )
            )
            replaced = tasks[0]
            tasks[0] = mandatory_task
            logger.info(
                f"  🔄 Replaced task: '{replaced.description[:50]}...' "
                f"with mandatory phase task"
            )
        else:
            tasks.append(mandatory_task)
            logger.info(f"  ➕ Added mandatory phase task")

        return tasks
