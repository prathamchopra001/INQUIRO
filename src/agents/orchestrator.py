"""
Orchestrator Agent for Inquiro.

The central coordinator that manages the research process.
It decides WHAT to investigate, monitors progress, and determines
when the objective has been met.

Think of it as the research director who:
- Reads the shared notebook (WorldModel) each morning
- Assigns tasks to the data scientist and librarian
- Decides when enough evidence exists to write the paper
"""

import json
import logging
from typing import List, Optional

from src.utils.llm_client import LLMClient
from src.world_model.world_model import WorldModel
from src.world_model.models import Task, TaskType, TaskStatus, generate_id
from src.novelty.novelty_detector import NoveltyDetector
from src.orchestration.plan_reviewer import PlanReviewer
from src.orchestration.cycle_phase_manager import CyclePhaseManager
from config.prompts.orchestrator import (
    TASK_GENERATION_PROMPT,
    COMPLETION_CHECK_PROMPT,
    DISCOVERY_RANKING_PROMPT,
)
from config.prompts.gap_analysis import GAP_ANALYSIS_PROMPT
from src.orchestration.research_plan import ResearchPlanner, format_task_generation_context, QuestionManager


logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Manages the research cycle: plan → execute → evaluate → repeat.

    This agent doesn't run code or search papers itself.
    It reads the world model and decides what others should do.

    Usage:
        orchestrator = OrchestratorAgent(llm_client, world_model)
        tasks = orchestrator.generate_tasks(objective, cycle=1, num_tasks=5)
        is_done = orchestrator.check_completion(objective, cycles_completed=3)
        top = orchestrator.rank_discoveries(objective)
    """

    def __init__(self, llm_client: LLMClient, world_model: WorldModel,
                 cycle_phase_manager: CyclePhaseManager = None,
                 has_dataset: bool = True,
                 use_adaptive_decomposition: bool = False):
        self.llm = llm_client
        self.world_model = world_model
        self.novelty_detector = NoveltyDetector(threshold=0.65)
        self.plan_reviewer = PlanReviewer(min_score=0.45)
        self.phase_manager = cycle_phase_manager
        self.has_dataset = has_dataset
        self.use_adaptive_decomposition = use_adaptive_decomposition
        
        # Research planning for multi-area objectives
        self.research_planner = ResearchPlanner(llm_client)
        self._research_plan: Optional[dict] = None
        self._last_coverage: Optional[dict] = None
        
        # Question-driven research
        self.question_manager = QuestionManager(
            llm_client, world_model, 
            use_adaptive_decomposition=use_adaptive_decomposition
        )
        self._questions_initialized = False

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _parse_json_response(self, response_text: str, fallback=None):
        """
        Safely parse JSON from LLM response.
        Handles markdown code blocks and leading/trailing text.
        """
        text = response_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON array or object from within the text
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                start = text.find(start_char)
                end = text.rfind(end_char)
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        continue

        logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
        return fallback if fallback is not None else []

    def _build_findings_text(self, findings) -> str:
        """
        Format a list of Finding objects into text for LLM prompts.
        """
        if not findings:
            return "No findings yet."

        lines = []
        for f in findings:
            lines.append(
                f"ID: {f.id}\n"
                f"Claim: {f.claim}\n"
                f"Type: {f.finding_type}\n"
                f"Confidence: {f.confidence:.2f}\n"
                f"Cycle: {f.cycle}\n"
                f"Tags: {', '.join(f.tags) if f.tags else 'none'}\n"
            )
        return "\n".join(lines)

    # =========================================================================
    # GAP ANALYSIS
    # =========================================================================

    def _build_gap_analysis(self, objective: str, cycle: int) -> str:
        """
        Analyze what's been answered vs what's still open.

        Queries the world model for all findings and completed tasks,
        then asks the LLM to identify gaps. The result is a structured
        text block that gets injected into the task generation prompt.

        Args:
            objective: The research goal
            cycle: Current cycle number

        Returns:
            A formatted text block describing gaps, or empty string
            if too early (cycle 1 has no findings to analyze).
        """
        # No gap analysis on cycle 1 — nothing to analyze yet
        if cycle <= 1:
            return ""

        all_findings = self.world_model.get_all_findings()
        if not all_findings:
            return ""

        # Group findings by cycle for a clearer picture
        findings_by_cycle = {}
        for f in all_findings:
            c = f.cycle
            if c not in findings_by_cycle:
                findings_by_cycle[c] = []
            findings_by_cycle[c].append(f)

        cycle_text_parts = []
        for c in sorted(findings_by_cycle.keys()):
            cycle_findings = findings_by_cycle[c]
            lines = [f"\n--- Cycle {c} ({len(cycle_findings)} findings) ---"]
            for f in cycle_findings:
                conf_str = f"{f.confidence:.2f}" if f.confidence else "?"
                lines.append(f"  [{f.finding_type}] (conf={conf_str}) {f.claim[:120]}")
            cycle_text_parts.append("\n".join(lines))

        findings_text = "\n".join(cycle_text_parts)

        # Build completed tasks summary
        completed_tasks_lines = []
        for c in range(1, cycle):
            tasks = self.world_model.get_tasks_by_cycle(c)
            for t in tasks:
                status = t.status.value if hasattr(t.status, 'value') else str(t.status)
                completed_tasks_lines.append(
                    f"  [Cycle {c}] ({status}) {t.description[:120]}"
                )

        completed_text = "\n".join(completed_tasks_lines) if completed_tasks_lines else "No tasks completed yet."

        # Ask LLM for gap analysis
        prompt = GAP_ANALYSIS_PROMPT.format(
            objective=objective,
            findings_by_cycle=findings_text,
            completed_tasks=completed_text,
        )

        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="gap_analysis",
                system="You are a research gap analyst. Respond only with valid JSON.",
                )
            result = self._parse_json_response(response.content, fallback={})

            # Handle case where LLM returns a list instead of dict
            if isinstance(result, list):
                logger.warning(f"Gap analysis: LLM returned list instead of dict, using fallback")
                result = {}

            if not result:
                return ""

            # Format gap analysis into readable text for the task generation prompt
            gap_lines = []
            gap_lines.append("\n[GAP ANALYSIS — What's been answered vs what's still open]")

            # Answered
            answered = result.get("answered", [])
            if answered:
                gap_lines.append("\n  ANSWERED:")
                for item in answered[:5]:
                    q = item.get("question", "")
                    a = item.get("answered_by", "")
                    gap_lines.append(f"    ✓ {q} → {a}")

            # Open questions
            open_qs = result.get("open_questions", [])
            if open_qs:
                gap_lines.append("\n  STILL OPEN (prioritize these):")
                for item in open_qs[:5]:
                    q = item.get("question", "")
                    imp = item.get("importance", "medium")
                    approach = item.get("suggested_approach", "")
                    gap_lines.append(f"    ✗ [{imp.upper()}] {q} (suggested: {approach})")

            # Weak areas
            weak = result.get("weak_areas", [])
            if weak:
                gap_lines.append("\n  NEEDS VALIDATION:")
                for item in weak[:3]:
                    claim = item.get("claim", "")[:80]
                    weakness = item.get("weakness", "")
                    gap_lines.append(f"    ⚠ {claim} — {weakness}")

            # Recommended focus
            focus = result.get("recommended_focus", "")
            if focus:
                gap_lines.append(f"\n  RECOMMENDED FOCUS FOR THIS CYCLE: {focus}")

            gap_text = "\n".join(gap_lines)
            logger.info(f"Gap analysis: {len(answered)} answered, {len(open_qs)} open, {len(weak)} weak")
            return gap_text

        except Exception as e:
            logger.warning(f"Gap analysis failed: {e} — proceeding without it")
            return ""

    # =========================================================================
    # EXPLORATION / EXPLOITATION STRATEGY
    # =========================================================================

    def _get_exploration_strategy(self, cycle: int, max_cycles: int) -> str:
        """
        Calculate exploration vs exploitation strategy based on cycle progress.
        
        Early cycles: 70% exploration (broad search), 30% exploitation (targeted)
        Late cycles: 30% exploration, 70% exploitation (filling gaps)
        
        This dynamic ratio helps the system:
        - Discover diverse findings early (avoid tunnel vision)
        - Focus on gaps and validation later (converge to completion)
        
        Args:
            cycle: Current cycle number (1-indexed)
            max_cycles: Maximum number of cycles
            
        Returns:
            Strategy instruction string for the task generation prompt
        """
        if max_cycles <= 1:
            # Single cycle: balanced approach
            explore_pct = 50
            exploit_pct = 50
            phase = "single-cycle"
        else:
            # Calculate progress through research (0.0 to 1.0)
            progress = (cycle - 1) / (max_cycles - 1) if max_cycles > 1 else 0.5
            
            # Linear interpolation from exploration-heavy to exploitation-heavy
            # Early (progress=0): 70% explore, 30% exploit
            # Late (progress=1): 30% explore, 70% exploit
            explore_pct = int(70 - 40 * progress)  # 70 → 30
            exploit_pct = int(30 + 40 * progress)  # 30 → 70
            
            # Determine phase name
            if progress < 0.33:
                phase = "exploration"
            elif progress < 0.67:
                phase = "balanced"
            else:
                phase = "exploitation"
        
        # Build strategy instruction
        strategy_lines = [
            f"**Current Phase**: {phase.upper()} (Cycle {cycle}/{max_cycles})",
            f"**Task Strategy**: {explore_pct}% exploratory / {exploit_pct}% targeted",
            "",
        ]
        
        if phase == "exploration":
            strategy_lines.extend([
                "🔍 EXPLORATION PHASE — Cast a wide net:",
                "  - Search diverse literature to discover unexpected connections",
                "  - Run exploratory data analysis to find patterns",
                "  - Don't overcommit to a single hypothesis yet",
                "  - Prioritize breadth over depth",
            ])
        elif phase == "balanced":
            strategy_lines.extend([
                "⚖️ BALANCED PHASE — Mix exploration and validation:",
                "  - Follow up on promising findings from earlier cycles",
                "  - Still explore new angles, but start focusing",
                "  - Validate key claims with additional evidence",
            ])
        else:  # exploitation
            strategy_lines.extend([
                "🎯 EXPLOITATION PHASE — Target gaps and validate:",
                "  - Focus on unanswered research questions",
                "  - Validate weak findings with additional evidence",
                "  - Fill specific gaps identified in gap analysis",
                "  - Prioritize depth over breadth",
            ])
        
        logger.info(f"Research strategy: {phase} phase ({explore_pct}% explore / {exploit_pct}% exploit)")
        return "\n".join(strategy_lines)

    # =========================================================================
    # RESEARCH PLANNING (for multi-area objectives)
    # =========================================================================

    def initialize_research_plan(self, objective: str) -> Optional[dict]:
        """
        Initialize research plan for multi-area objectives.
        Call this BEFORE the first research cycle.
        
        For complex objectives that cover multiple research areas,
        this creates a structured plan and enables coverage tracking.
        
        Args:
            objective: The research objective string
            
        Returns:
            Research plan dict if multi-area, None otherwise
        """
        if self.research_planner.is_multi_area_objective(objective):
            logger.info("📋 Detected multi-area objective - creating research plan")
            self._research_plan = self.research_planner.create_research_plan(objective)
            
            # Log the plan
            for area in self._research_plan.get("areas", []):
                keywords_preview = area.get('keywords', [])[:3]
                logger.info(f"  📋 Area: {area['name']} (keywords: {keywords_preview})")
            
            return self._research_plan
        else:
            logger.info("Single-area objective - no research plan needed")
            return None
    
    def get_coverage_context(self) -> str:
        """
        Get coverage tracking context for task generation.
        
        Returns:
            Formatted string showing which areas are covered/uncovered
        """
        if not self._research_plan:
            return ""
        
        # Get current findings
        all_findings = self.world_model.get_all_findings()
        finding_dicts = [{"claim": f.claim} for f in all_findings]
        
        # Check coverage
        self._last_coverage = self.research_planner.check_coverage(
            self._research_plan,
            finding_dicts
        )
        
        logger.info(f"Research coverage: {self._last_coverage.get('overall_coverage', 0)}%")
        
        return self.research_planner.format_coverage_for_prompt(self._last_coverage)

    # =========================================================================
    # QUESTION-DRIVEN RESEARCH
    # =========================================================================

    def initialize_questions(self, objective: str, domain_context: str = "") -> List[dict]:
        """
        Initialize research questions before cycle 1.
        
        Decomposes the objective into specific, trackable research questions
        that guide task generation and completion evaluation.
        
        Args:
            objective: The research objective
            domain_context: Optional domain information from DomainAnchorExtractor
            
        Returns:
            List of question dicts that were created
        """
        if self._questions_initialized:
            logger.info("Questions already initialized - skipping")
            return []
        
        logger.info("📋 Decomposing objective into research questions...")
        
        questions = self.question_manager.decompose_objective(
            objective=objective,
            domain_context=domain_context,
            store_in_world_model=True
        )
        
        self._questions_initialized = True
        
        # Log the questions
        for q in questions:
            priority = q.get('priority', 'medium')
            priority_emoji = '🔴' if priority == 'high' else '🟡' if priority == 'medium' else '🟢'
            logger.info(f"  {priority_emoji} Q: {q.get('question_text', '')[:80]}...")
        
        logger.info(f"📋 Created {len(questions)} research questions")
        return questions
    
    def validate_research_questions(self, cycle: int) -> dict:
        """
        Validate which research questions have been answered.
        
        Call this after each cycle to update question status based on
        the findings collected so far.
        
        Args:
            cycle: Current cycle number
            
        Returns:
            Validation report with status for each question
        """
        if not self._questions_initialized:
            return {"evaluations": [], "overall_progress": {}}
        
        # Get all findings as dicts
        all_findings = self.world_model.get_all_findings()
        finding_dicts = [
            {"id": f.id, "claim": f.claim, "confidence": f.confidence}
            for f in all_findings
        ]
        
        validation = self.question_manager.validate_questions(
            findings=finding_dicts,
            current_cycle=cycle
        )
        
        # Log progress
        progress = validation.get("overall_progress", {})
        answered = progress.get("answered", 0)
        partial = progress.get("partial", 0)
        unanswered = progress.get("unanswered", 0)
        completion = progress.get("completion_percentage", 0)
        
        logger.info(
            f"📊 Question progress: {answered} answered, {partial} partial, "
            f"{unanswered} unanswered ({completion}% complete)"
        )
        
        return validation
    
    def get_question_context_for_tasks(self) -> str:
        """
        Get unanswered questions formatted for task generation prompt.
        
        Returns:
            Formatted string showing questions that need research
        """
        if not self._questions_initialized:
            return ""
        
        return self.question_manager.format_questions_for_task_generation()
    
    def get_research_completion_status(self) -> dict:
        """
        Get overall research completion status based on questions.
        
        Returns:
            Dict with completion metrics:
            - complete: bool - all questions answered
            - percentage: float - completion percentage
            - answered/partial/unanswered: counts
            - should_continue: bool - whether to continue research
        """
        if not self._questions_initialized:
            return {"complete": False, "percentage": 0, "should_continue": True}
        
        return self.question_manager.get_completion_status()
    
    def get_research_gaps(self, min_unanswered: int = 2) -> dict:
        """
        Check if significant research gaps remain.
        
        Used after report generation to decide whether to continue research.
        Returns gap information if high-priority questions remain unanswered.
        
        Args:
            min_unanswered: Minimum unanswered questions to consider as "gaps"
            
        Returns:
            Dict with:
            - has_gaps: bool - whether significant gaps exist
            - gap_count: int - number of unanswered questions
            - high_priority_gaps: list - high-priority unanswered questions
            - gap_summary: str - human-readable summary
        """
        if not self._questions_initialized:
            return {
                "has_gaps": False,
                "gap_count": 0,
                "high_priority_gaps": [],
                "gap_summary": "No questions initialized"
            }
        
        # Get all questions from world model
        all_questions = self.world_model.get_all_questions()
        
        if not all_questions:
            return {
                "has_gaps": False,
                "gap_count": 0,
                "high_priority_gaps": [],
                "gap_summary": "No questions to evaluate"
            }
        
        # Filter unanswered questions
        # Handle both enum and string values for status
        def get_status_str(q):
            if hasattr(q.status, 'value'):
                return q.status.value
            return str(q.status)
        
        def get_priority_str(q):
            if hasattr(q.priority, 'value'):
                return q.priority.value
            return str(q.priority)
        
        unanswered = [
            q for q in all_questions
            if get_status_str(q) in ("unanswered", "partial")
        ]
        
        # Find high-priority gaps
        high_priority = [
            q for q in unanswered
            if get_priority_str(q) == "high"
        ]
        
        gap_count = len(unanswered)
        has_gaps = gap_count >= min_unanswered or len(high_priority) >= 1
        
        # Build summary
        if not unanswered:
            gap_summary = "All research questions have been addressed."
        elif high_priority:
            gap_summary = (
                f"{gap_count} questions remain ({len(high_priority)} high-priority). "
                f"Key gaps: {'; '.join(q.question_text[:60] for q in high_priority[:3])}"
            )
        else:
            gap_summary = f"{gap_count} lower-priority questions remain unanswered."
        
        return {
            "has_gaps": has_gaps,
            "gap_count": gap_count,
            "high_priority_gaps": [
                {
                    "id": q.id,
                    "question": q.question_text,
                    "priority": get_priority_str(q),
                    "status": get_status_str(q),
                }
                for q in high_priority
            ],
            "all_gaps": [
                {
                    "id": q.id,
                    "question": q.question_text,
                    "priority": get_priority_str(q),
                    "status": get_status_str(q),
                }
                for q in unanswered
            ],
            "gap_summary": gap_summary,
        }

    # =========================================================================
    # CORE METHODS
    # =========================================================================

    def generate_tasks(
        self,
        objective: str,
        cycle: int,
        num_tasks: int = 5,
        max_cycles: int = 10,
    ) -> List[Task]:
        """
        Generate research tasks for the next cycle.

        Reads the current world model state and asks the LLM:
        "Given what we know, what should we investigate next?"

        Args:
            objective: The overall research goal
            cycle: Current cycle number (used for context)
            num_tasks: How many tasks to generate (default 5)

        Returns:
            List of Task objects ready to be executed

        Design notes:
            - Pure LLM output for task descriptions (flexible, adaptive)
            - Rule-based fallback if LLM fails (reliable)
            - Tasks are NOT saved to world model here — Inquiro does that
        """
        logger.info(f"Generating tasks for cycle {cycle}...")

        # Get explore/exploit mode instruction for this cycle
        mode_instruction = self.plan_reviewer.get_mode_instruction(
            cycle=cycle, max_cycles=max_cycles
        )

        # Get phase-specific instruction (if phases are defined)
        phase_instruction = ""
        if self.phase_manager:
            phase_instruction = self.phase_manager.get_phase_instruction(cycle)

        # Literature-only mode instruction
        dataset_instruction = ""
        if not self.has_dataset:
            dataset_instruction = (
                "\n[LITERATURE-ONLY MODE — NO DATASET AVAILABLE]\n"
                "There is NO dataset for this research run. "
                "Generate ONLY 'literature' type tasks. "
                "Do NOT generate any 'data_analysis' tasks — there is no data to analyze.\n"
                "Focus on: searching papers, reviewing methods, comparing approaches, "
                "synthesizing knowledge from existing literature.\n"
            )

        # Get current knowledge state
        world_model_summary = self.world_model.get_summary()

        # Build gap analysis (from cycle 2 onward)
        gap_analysis = self._build_gap_analysis(objective, cycle)
        if gap_analysis:
            logger.info(f"\U0001f50d Gap analysis injected into task generation context")

        # Build coverage context for multi-area objectives
        coverage_context = self.get_coverage_context()
        if coverage_context:
            logger.info(f"\U0001f4ca Coverage tracking injected into task generation context")

        # Build question context for question-driven research
        question_context = self.get_question_context_for_tasks()
        if question_context:
            logger.info(f"\u2753 Research questions injected into task generation context")

        # Calculate exploration/exploitation strategy based on cycle progress
        exploration_strategy = self._get_exploration_strategy(cycle, max_cycles)

        # Inject mode + phase + dataset + gap analysis + coverage + questions into world model summary
        world_model_summary = (
            mode_instruction + phase_instruction + dataset_instruction
            + gap_analysis
            + "\n" + coverage_context
            + "\n" + question_context
            + "\n" + world_model_summary
        )

        # Ask LLM to generate tasks
        prompt = TASK_GENERATION_PROMPT.format(
            objective=objective,
            world_model_summary=world_model_summary,
            cycle_number=cycle,
            num_tasks=num_tasks,
            exploration_strategy=exploration_strategy,
        )

        response = self.llm.complete_for_role(
            prompt=prompt,
            role="task_generation",
            system="You are a scientific research orchestrator. Respond only with valid JSON."
            )

        raw_tasks = self._parse_json_response(response.content, fallback=[])

        # Convert raw dicts to Task objects
        tasks = []
        for raw in raw_tasks:
            # Validate required fields — skip malformed tasks
            if not isinstance(raw, dict):
                continue
            if "type" not in raw or "description" not in raw:
                logger.warning(f"Skipping malformed task: {raw}")
                continue

            # Map string type to TaskType enum safely
            task_type_str = raw.get("type", "data_analysis").lower()
            if "literature" in task_type_str:
                task_type = TaskType.LITERATURE_SEARCH
            else:
                task_type = TaskType.DATA_ANALYSIS

            task = Task(
                task_type=task_type,
                description=raw.get("description", ""),
                goal=raw.get("goal", ""),
                priority=raw.get("priority", "medium"),
                status=TaskStatus.PENDING,
                cycle=cycle,
            )
            tasks.append(task)

        # Fallback: if LLM completely failed, generate sensible default tasks
        if not tasks:
            logger.warning("LLM task generation failed — using rule-based fallback")
            tasks = self._fallback_task_generation(objective, cycle)

        # ── Phase enforcement ─────────────────────────────────────────────
        # Ensure at least one task matches the current phase requirements.
        # If not, inject a mandatory task to keep research progressing.
        if self.phase_manager:
            tasks = self.phase_manager.enforce_phase(cycle, tasks)

        # ── Novelty filter ────────────────────────────────────────────────
        # Sync world model findings into detector before checking
        existing_findings = self.world_model.get_recent_findings(limit=50)
        for f in existing_findings:
            self.novelty_detector.register_finding(
                f.claim if hasattr(f, 'claim') else str(f)
            )

        novel_tasks = []
        for task in tasks:
            result = self.novelty_detector.check(task.description)
            if result.is_novel:
                novel_tasks.append(task)
                # Register so future tasks in this batch don't duplicate it
                self.novelty_detector.register_task(task.description)
            else:
                logger.info(
                    f"  🔁 Task skipped (redundant): {task.description[:60]}..."
                    f" | {result.reason}"
                )

        # Always keep at least 1 task even if all flagged redundant
        if not novel_tasks and tasks:
            logger.warning("All tasks flagged redundant — keeping top task")
            novel_tasks = [tasks[0]]

        if len(novel_tasks) < len(tasks):
            logger.info(
                f"Novelty filter: kept {len(novel_tasks)}/{len(tasks)} tasks"
            )

        stats = self.novelty_detector.get_stats()
        logger.debug(f"Novelty detector: {stats}")

        # ── Plan Reviewer ────────────────────────────────────────────────
        # Score each novel task — reject weak ones
        reviewed = self.plan_reviewer.review_tasks(
            tasks=novel_tasks,
            objective=objective,
            cycle=cycle,
            max_cycles=max_cycles,
        )
        approved_tasks = [task for task, score in reviewed if score.passes]

        # Always keep at least 1 task even if all scored low
        if not approved_tasks and novel_tasks:
            logger.warning("All tasks scored below threshold — keeping top task")
            approved_tasks = [novel_tasks[0]]

        if len(approved_tasks) < len(novel_tasks):
            logger.info(
                f"Plan review: kept {len(approved_tasks)}/{len(novel_tasks)} tasks"
            )

        logger.info(f"Generated {len(approved_tasks)} tasks for cycle {cycle}")
        return approved_tasks

    def check_completion(
        self,
        objective: str,
        cycles_completed: int,
        max_cycles: int,
    ) -> bool:
        """
        Decide whether research has gathered enough evidence to stop.

        Uses a hybrid approach:
        - Hard rules first (too few findings = always continue)
        - Quality metrics (not just count-based)
        - LLM judgment for nuanced cases

        Args:
            objective: The research goal
            cycles_completed: How many cycles have run so far
            max_cycles: The hard upper limit on cycles

        Returns:
            True if research is complete, False to continue
        """
        stats = self.world_model.get_statistics()
        total_findings = stats.get("total_findings", 0)
        total_relationships = stats.get("total_relationships", 0)

        # --- Hard rules (always apply regardless of LLM) ---

        # Never stop on the first cycle — always do at least 2
        if cycles_completed < 2:
            logger.info("Completion check: too early (< 2 cycles)")
            return False

        # Always stop at max cycles
        if cycles_completed >= max_cycles:
            logger.info(f"Completion check: reached max cycles ({max_cycles})")
            return True

        # Need at least 3 findings to write any report
        if total_findings < 3:
            logger.info(f"Completion check: too few findings ({total_findings})")
            return False

        # --- Check question-driven completion ---
        question_status = self.get_research_completion_status()
        if self._questions_initialized:
            q_complete = question_status.get("complete", False)
            q_percentage = question_status.get("percentage", 0)
            q_should_continue = question_status.get("should_continue", True)
            
            logger.info(
                f"Question completion: {q_percentage}% "
                f"({question_status.get('answered', 0)} answered, "
                f"{question_status.get('unanswered', 0)} unanswered)"
            )
            
            # If all questions answered, we can complete early
            if q_complete and q_percentage >= 90:
                logger.info("Completion check: all research questions answered!")
                return True
            
            # If questions say continue and we're not at max, keep going
            if q_should_continue and cycles_completed < max_cycles * 0.8:
                # Only force continue if we have significant unanswered questions
                if question_status.get('unanswered', 0) >= 2:
                    logger.info(f"Completion check: {question_status.get('unanswered', 0)} questions still unanswered")
                    return False

        # --- Calculate quality metrics ---
        quality_metrics = self._calculate_quality_metrics(cycles_completed)
        
        # --- LLM judgment with quality-aware prompt ---
        world_model_summary = self.world_model.get_summary()

        # Build question status summary for LLM
        question_summary = ""
        if self._questions_initialized:
            question_summary = f"""

=== RESEARCH QUESTIONS STATUS ===
Total questions: {question_status.get('total', 0)}
  - Answered: {question_status.get('answered', 0)}
  - Partially answered: {question_status.get('partial', 0)}
  - Unanswered: {question_status.get('unanswered', 0)}
Completion: {question_status.get('percentage', 0):.1f}%
"""

        # Enhanced prompt with quality metrics
        quality_section = f"""

=== QUALITY METRICS ===
Total findings: {total_findings}
  - Data-driven findings: {quality_metrics['data_findings']}
  - Literature findings: {quality_metrics['lit_findings']}
  - High-confidence findings (>0.7): {quality_metrics['high_confidence']}
Average confidence: {quality_metrics['avg_confidence']:.2f}
Findings this cycle: {quality_metrics['last_cycle_findings']}
Novelty rate: {quality_metrics['novelty_rate']:.0%} (lower = diminishing returns)
{question_summary}
=== COMPLETION CRITERIA ===
1. Are core aspects of the objective addressed with evidence?
2. Are there at least 3 high-confidence findings?
3. Is the novelty rate below 30%? (suggests saturation)
4. Are most research questions answered (if questions were defined)?
5. Would more cycles likely add significant value?
"""

        prompt = COMPLETION_CHECK_PROMPT.format(
            objective=objective,
            world_model_summary=world_model_summary + quality_section,
            cycles_completed=cycles_completed,
            max_cycles=max_cycles,
            total_findings=total_findings,
            total_relationships=total_relationships,
        )

        response = self.llm.complete_for_role(
            prompt=prompt,
            role="completion_check",
            system="You are a scientific research evaluator. Respond only with valid JSON."
                )

        result = self._parse_json_response(response.content, fallback={})

        # Handle case where LLM returns a list instead of dict
        if isinstance(result, list):
            logger.warning(f"Completion check: LLM returned list instead of dict, using fallback")
            result = {}

        is_complete = result.get("is_complete", False)
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "No reasoning provided")

        logger.info(
            f"Completion check: is_complete={is_complete}, "
            f"confidence={confidence:.2f}, reasoning={reasoning}"
        )
        logger.info(
            f"Quality metrics: high_conf={quality_metrics['high_confidence']}, "
            f"novelty={quality_metrics['novelty_rate']:.0%}"
        )

        # Be conservative: require high confidence to stop early
        # Also check quality metrics as a safety net
        if is_complete and confidence >= 0.7:
            # Extra check: don't stop if novelty is still high
            if quality_metrics['novelty_rate'] > 0.5 and cycles_completed < max_cycles * 0.75:
                logger.info("Completion check: overriding — novelty rate still high")
                return False
            return True

        return False

    def _calculate_quality_metrics(self, current_cycle: int) -> dict:
        """
        Calculate quality-based metrics for completion evaluation.
        
        Returns dict with:
        - data_findings: count of data analysis findings
        - lit_findings: count of literature findings
        - high_confidence: count of findings with confidence > 0.7
        - avg_confidence: average confidence across all findings
        - last_cycle_findings: findings added in current cycle
        - novelty_rate: estimate of how much new info each cycle adds
        """
        all_findings = self.world_model.get_all_findings()
        
        if not all_findings:
            return {
                'data_findings': 0,
                'lit_findings': 0,
                'high_confidence': 0,
                'avg_confidence': 0.0,
                'last_cycle_findings': 0,
                'novelty_rate': 1.0,
            }
        
        # Count by type
        data_findings = sum(1 for f in all_findings 
                           if getattr(f, 'finding_type', '') == 'data_analysis')
        lit_findings = sum(1 for f in all_findings 
                          if getattr(f, 'finding_type', '') == 'literature')
        
        # Confidence analysis
        confidences = [getattr(f, 'confidence', 0.5) for f in all_findings]
        high_conf = sum(1 for c in confidences if c > 0.7)
        avg_conf = sum(confidences) / len(confidences)
        
        # Current cycle findings
        last_cycle = [f for f in all_findings if getattr(f, 'cycle', 0) == current_cycle]
        
        # Novelty rate: compare this cycle to previous
        # If we're adding fewer findings each cycle, we're hitting diminishing returns
        prev_cycle = [f for f in all_findings if getattr(f, 'cycle', 0) == current_cycle - 1]
        
        if prev_cycle:
            novelty_rate = len(last_cycle) / max(len(prev_cycle), 1)
        else:
            novelty_rate = 1.0  # First cycle, assume full novelty
        
        # Cap at 1.0
        novelty_rate = min(novelty_rate, 1.0)
        
        return {
            'data_findings': data_findings,
            'lit_findings': lit_findings,
            'high_confidence': high_conf,
            'avg_confidence': avg_conf,
            'last_cycle_findings': len(last_cycle),
            'novelty_rate': novelty_rate,
        }

    def rank_discoveries(
        self,
        objective: str,
        top_n: int = 10,
    ) -> List[dict]:
        """
        Rank findings by importance for the report generator.

        Uses a hybrid approach:
        - LLM scores findings by relevance, evidence, novelty
        - Falls back to support_count + confidence if LLM fails

        Args:
            objective: Research goal (context for scoring)
            top_n: How many top findings to return

        Returns:
            List of dicts: [{"finding": Finding, "score": float, "reasoning": str}]
        """
        all_findings = self.world_model.get_all_findings()

        if not all_findings:
            logger.warning("No findings to rank")
            return []

        # Build text representation for LLM
        findings_text = self._build_findings_text(all_findings)

        prompt = DISCOVERY_RANKING_PROMPT.format(
            objective=objective,
            findings_text=findings_text,
        )

        response = self.llm.complete_for_role(
            prompt=prompt,
            role="scoring",
            system="You are a scientific editor. Respond only with valid JSON."
        )

        ranked_raw = self._parse_json_response(response.content, fallback=[])

        # Build a lookup map for fast access
        findings_map = {f.id: f for f in all_findings}

        ranked = []
        for item in ranked_raw:
            if not isinstance(item, dict):
                continue
            finding_id = item.get("finding_id")
            if finding_id not in findings_map:
                continue
            ranked.append({
                "finding": findings_map[finding_id],
                "score": item.get("score", 0.0),
                "reasoning": item.get("reasoning", ""),
            })

        # If LLM ranking failed or returned partial results, use graph-based fallback
        if len(ranked) < len(all_findings):
            logger.info("LLM ranking incomplete — supplementing with graph-based scores")
            ranked = self._fallback_ranking(all_findings, ranked)

        return ranked[:top_n]

    # =========================================================================
    # FALLBACK METHODS (rule-based safety nets)
    # =========================================================================

    def _fallback_task_generation(self, objective: str, cycle: int) -> List[Task]:
        """
        Rule-based task generation when LLM fails.
        Generates one data task and one literature task as a safe default.
        In literature-only mode, generates two literature tasks instead.
        """
        tasks = []

        if self.has_dataset:
            tasks.append(Task(
                task_type=TaskType.DATA_ANALYSIS,
                description=f"Perform exploratory data analysis relevant to: {objective[:100]}",
                goal="Identify patterns, correlations, and anomalies in the dataset",
                priority="high",
                status=TaskStatus.PENDING,
                cycle=cycle,
            ))

        tasks.append(Task(
            task_type=TaskType.LITERATURE_SEARCH,
            description=f"Search for papers related to: {objective[:100]}",
            goal="Find relevant prior work and established findings",
            priority="high" if not self.has_dataset else "medium",
            status=TaskStatus.PENDING,
            cycle=cycle,
        ))

        if not self.has_dataset:
            tasks.append(Task(
                task_type=TaskType.LITERATURE_SEARCH,
                description=f"Search for review papers and meta-analyses on: {objective[:100]}",
                goal="Build a comprehensive understanding of the research landscape",
                priority="medium",
                status=TaskStatus.PENDING,
                cycle=cycle,
            ))

        return tasks

    def _fallback_ranking(
        self, all_findings: list, already_ranked: list
    ) -> List[dict]:
        """
        Graph-based ranking fallback using support count + confidence.
        Used when LLM ranking is incomplete or fails entirely.
        """
        ranked_ids = {item["finding"].id for item in already_ranked}

        # Add missing findings using graph-based scores
        top_findings = self.world_model.get_top_findings(n=len(all_findings))

        for item in top_findings:
            f = item["finding"]
            if f.id not in ranked_ids:
                already_ranked.append({
                    "finding": f,
                    "score": min(item["score"] / 10.0, 1.0),  # Normalize to 0-1
                    "reasoning": f"Graph score: {item['support_count']} supporting findings",
                })
                ranked_ids.add(f.id)

        # Sort everything by score
        already_ranked.sort(key=lambda x: x["score"], reverse=True)
        return already_ranked

    def propose_relationships(self, objective: str, cycle: int, max_comparison_findings: int = 30) -> int:
        """
        After each cycle, find connections between findings and create relationships.

        Compares findings from the current cycle against recent/relevant findings
        to avoid context overflow as the world model grows.

        Args:
            objective: Research goal
            cycle: Current cycle number
            max_comparison_findings: Max findings to include in comparison set
                                     (prevents context overflow in later cycles)

        Returns:
            Number of relationships successfully created.
        """
        all_findings = self.world_model.get_all_findings()
        if len(all_findings) < 2:
            return 0

        # Only compare recent findings against everything else
        recent = self.world_model.get_findings_by_cycle(cycle)
        if not recent:
            logger.info(f"  No new findings in cycle {cycle} — skipping relationship proposal")
            return 0

        # OPTIMIZATION: Limit comparison set to avoid context overflow
        # In later cycles with 50+ findings, sending all of them overwhelms the LLM
        # and often results in empty/truncated responses.
        #
        # Strategy: Use top findings by score (most important to connect)
        # plus any findings from recent cycles (most likely to relate)
        if len(all_findings) > max_comparison_findings:
            # Get top findings by score (graph-based ranking)
            top_findings_data = self.world_model.get_top_findings(n=max_comparison_findings // 2)
            top_findings = [item["finding"] for item in top_findings_data]
            top_ids = {f.id for f in top_findings}
            
            # Also include findings from last 2 cycles (temporal relevance)
            recent_cycles = []
            for c in range(max(1, cycle - 2), cycle):
                recent_cycles.extend(self.world_model.get_findings_by_cycle(c))
            
            # Combine: top findings + recent cycles, deduplicated
            comparison_findings = list(top_findings)
            for f in recent_cycles:
                if f.id not in top_ids and len(comparison_findings) < max_comparison_findings:
                    comparison_findings.append(f)
                    top_ids.add(f.id)
            
            # Ensure current cycle's findings are included for self-comparison
            for f in recent:
                if f.id not in top_ids:
                    comparison_findings.append(f)
            
            logger.info(
                f"  Relationship context: {len(comparison_findings)}/{len(all_findings)} findings "
                f"(top {len(top_findings)} + recent cycles + current)"
            )
        else:
            comparison_findings = all_findings

        recent_text = self._build_findings_text(recent)
        comparison_text = self._build_findings_text(comparison_findings)

        prompt = f"""You are a scientific knowledge graph builder.

## Research Objective
{objective}

## New Findings From This Cycle ({len(recent)} findings)
{recent_text}

## Existing Findings to Compare Against ({len(comparison_findings)} findings)
{comparison_text}

## Task
Identify relationships between the NEW findings and the EXISTING findings.
Only propose relationships where there is a clear logical connection.

Relationship types:
- "supports": Finding A provides evidence for Finding B
- "contradicts": Finding A conflicts with Finding B  
- "extends": Finding A adds detail or nuance to Finding B
- "relates_to": Finding A is topically connected to Finding B

IMPORTANT: Use the exact finding IDs shown above. Do not invent IDs.

## Output Format
Respond with ONLY a valid JSON array. If no clear relationships exist, return [].

[
  {{
    "from_id": "<exact finding ID from above>",
    "to_id": "<exact finding ID from above>",
    "relationship_type": "supports",
    "strength": 0.8,
    "reasoning": "One sentence explaining the connection"
  }}
]

Maximum 5 relationships. Only propose high-confidence connections."""

        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="relationship_proposal",
                system="You are a scientific knowledge graph builder. Respond only with valid JSON."
            )
            raw_content = response.content
        except Exception as e:
            logger.warning(f"  Relationship proposal LLM call failed: {e}")
            return 0

        proposed = self._parse_json_response(raw_content, fallback=[])
        
        # Debug logging when LLM returns empty despite having findings
        if not proposed and len(recent) > 0:
            logger.warning(
                f"  LLM returned no relationships despite {len(recent)} new findings. "
                f"Response preview: {raw_content[:200]}..."
            )
        
        valid_ids = {f.id for f in all_findings}
        created = 0
        skipped_invalid = 0

        for rel in proposed:
            if not isinstance(rel, dict):
                continue
            from_id = rel.get("from_id", "")
            to_id = rel.get("to_id", "")

            if from_id not in valid_ids or to_id not in valid_ids:
                logger.debug(f"Skipping relationship — invalid IDs: {from_id} → {to_id}")
                skipped_invalid += 1
                continue
            if from_id == to_id:
                continue

            rel_type = rel.get("relationship_type", "relates_to")
            if rel_type not in {"supports", "contradicts", "extends", "relates_to"}:
                rel_type = "relates_to"

            try:
                self.world_model.add_relationship(
                    from_id=from_id,
                    to_id=to_id,
                    relationship_type=rel_type,
                    strength=float(rel.get("strength", 0.7)),
                    reasoning=rel.get("reasoning", ""),
                )
                created += 1
                logger.info(
                    f"  🔗 Relationship: [{rel_type}] "
                    f"{from_id[:8]}... → {to_id[:8]}..."
                )
            except Exception as e:
                logger.warning(f"Could not create relationship: {e}")

        if skipped_invalid > 0:
            logger.warning(f"  Skipped {skipped_invalid} relationships with invalid finding IDs")
        
        logger.info(f"Proposed {len(proposed)} relationships, created {created}")
        return created
