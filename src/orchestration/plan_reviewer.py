# -*- coding: utf-8 -*-
"""
Plan Reviewer — scores and filters proposed tasks before execution.

Two responsibilities:
  1. Compute exploration/exploitation ratio based on cycle progress
  2. Score each proposed task on 5 dimensions, rejecting weak ones

Exploration/exploitation schedule:
  Cycle 1-2:  70% explore / 30% exploit  (cast wide net)
  Cycle 3-4:  50% explore / 50% exploit  (balanced)
  Cycle 5+:   30% explore / 70% exploit  (deepen best leads)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TaskScore:
    """Score of a single proposed task across 5 dimensions."""
    description: str
    specificity:   float   # Is the task concrete and actionable?
    relevance:     float   # Does it address the research objective?
    novelty:       float   # Does it explore new ground?
    coverage:      float   # Does it fill a gap in current knowledge?
    feasibility:   float   # Can it realistically be executed?
    composite:     float   # Weighted composite
    passes:        bool    # Whether task should proceed
    reason:        str     # Explanation


# Minimum composite score for a task to proceed
DEFAULT_MIN_TASK_SCORE = 0.45

# Dimension weights
TASK_SCORE_WEIGHTS = {
    "specificity": 0.25,
    "relevance":   0.30,
    "novelty":     0.20,
    "coverage":    0.15,
    "feasibility": 0.10,
}


class PlanReviewer:
    """
    Reviews and scores proposed tasks before execution.

    Usage:
        reviewer = PlanReviewer()
        
        # Get the ratio for this cycle
        ratio = reviewer.get_explore_exploit_ratio(cycle=2, max_cycles=10)
        # → {"explore": 0.70, "exploit": 0.30, "mode": "exploratory"}
        
        # Score proposed tasks
        scored = reviewer.review_tasks(tasks, objective, cycle, max_cycles)
        approved = [t for t, s in scored if s.passes]
    """

    def __init__(self, min_score: float = DEFAULT_MIN_TASK_SCORE):
        self.min_score = min_score

    # =========================================================================
    # EXPLORATION / EXPLOITATION RATIO
    # =========================================================================

    def get_explore_exploit_ratio(
        self, cycle: int, max_cycles: int
    ) -> dict:
        """
        Compute the exploration/exploitation ratio for a given cycle.

        Early cycles: explore broadly (high explore ratio)
        Late cycles:  exploit best leads (high exploit ratio)

        Args:
            cycle:      Current cycle number (1-indexed)
            max_cycles: Total cycles planned

        Returns:
            Dict with explore, exploit, mode, and prompt_hint
        """
        if max_cycles <= 0:
            max_cycles = 1

        progress = cycle / max_cycles  # 0.0 → 1.0

        # Linear interpolation from 70/30 to 30/70
        explore = 0.70 - (0.40 * progress)
        exploit = 1.0 - explore

        # Categorize mode
        if explore >= 0.60:
            mode = "exploratory"
            prompt_hint = (
                "FOCUS ON EXPLORATION: Generate broad, diverse tasks that "
                "investigate new angles. Prioritize discovery over validation."
            )
        elif explore >= 0.45:
            mode = "balanced"
            prompt_hint = (
                "BALANCED MODE: Mix broad exploratory tasks with targeted "
                "follow-ups on the most promising existing findings."
            )
        else:
            mode = "exploitative"
            prompt_hint = (
                "FOCUS ON EXPLOITATION: Generate targeted tasks that deepen "
                "and validate the strongest existing findings. Prioritize "
                "evidence strength over new directions."
            )

        ratio = {
            "explore": round(explore, 2),
            "exploit": round(exploit, 2),
            "mode": mode,
            "prompt_hint": prompt_hint,
            "cycle": cycle,
            "max_cycles": max_cycles,
        }

        logger.info(
            f"Cycle {cycle}/{max_cycles} → {mode} "
            f"(explore={explore:.0%}, exploit={exploit:.0%})"
        )

        return ratio

    def get_mode_instruction(self, cycle: int, max_cycles: int) -> str:
        """
        Get a short instruction string to inject into the task generation prompt.
        This tells the LLM whether to explore broadly or exploit deeply.
        """
        ratio = self.get_explore_exploit_ratio(cycle, max_cycles)
        return (
            f"\n[RESEARCH MODE: {ratio['mode'].upper()} — "
            f"Cycle {cycle}/{max_cycles}]\n"
            f"{ratio['prompt_hint']}\n"
        )

    # =========================================================================
    # TASK SCORING
    # =========================================================================

    def review_tasks(
        self,
        tasks: list,
        objective: str,
        cycle: int,
        max_cycles: int,
    ) -> list[tuple]:
        """
        Score a list of proposed tasks and flag weak ones.

        Args:
            tasks:      List of Task objects (or dicts with 'description')
            objective:  Research objective for relevance scoring
            cycle:      Current cycle
            max_cycles: Total cycles planned

        Returns:
            List of (task, TaskScore) tuples for ALL tasks.
            Caller decides what to do with tasks where score.passes=False.
        """
        ratio = self.get_explore_exploit_ratio(cycle, max_cycles)
        results = []

        for task in tasks:
            desc = (
                task.description
                if hasattr(task, "description")
                else task.get("description", "")
            )
            goal = (
                task.goal
                if hasattr(task, "goal")
                else task.get("goal", "")
            )

            score = self._score_task(desc, goal, objective, ratio)
            results.append((task, score))

            if not score.passes:
                logger.info(
                    f"  📋 Task scored low ({score.composite:.2f}): "
                    f"{desc[:60]} | {score.reason}"
                )

        passed = sum(1 for _, s in results if s.passes)
        logger.info(
            f"Plan review: {passed}/{len(tasks)} tasks approved "
            f"(mode={ratio['mode']})"
        )

        return results

    def _score_task(
        self,
        description: str,
        goal: str,
        objective: str,
        ratio: dict,
    ) -> TaskScore:
        """Score a single task on 5 dimensions."""
        text = (description + " " + goal).lower()
        obj_lower = objective.lower()

        # ── 1. Specificity (0-1) ─────────────────────────────────────────
        specificity = self._score_specificity(text)

        # ── 2. Relevance (0-1) ───────────────────────────────────────────
        relevance = self._score_relevance(text, obj_lower)

        # ── 3. Novelty (0-1) ─────────────────────────────────────────────
        # In exploit mode, novelty is less important
        novelty = self._score_novelty(text, ratio["mode"])

        # ── 4. Coverage (0-1) ────────────────────────────────────────────
        coverage = self._score_coverage(text)

        # ── 5. Feasibility (0-1) ─────────────────────────────────────────
        feasibility = self._score_feasibility(text)

        # ── Composite ────────────────────────────────────────────────────
        composite = (
            specificity   * TASK_SCORE_WEIGHTS["specificity"] +
            relevance     * TASK_SCORE_WEIGHTS["relevance"] +
            novelty       * TASK_SCORE_WEIGHTS["novelty"] +
            coverage      * TASK_SCORE_WEIGHTS["coverage"] +
            feasibility   * TASK_SCORE_WEIGHTS["feasibility"]
        )

        passes = composite >= self.min_score

        reason = self._build_reason(
            specificity, relevance, novelty, coverage, feasibility, composite
        )

        return TaskScore(
            description=description[:80],
            specificity=round(specificity, 2),
            relevance=round(relevance, 2),
            novelty=round(novelty, 2),
            coverage=round(coverage, 2),
            feasibility=round(feasibility, 2),
            composite=round(composite, 2),
            passes=passes,
            reason=reason,
        )

    def _score_specificity(self, text: str) -> float:
        """Is the task concrete and actionable?"""
        score = 0.5

        # Specific method names boost score
        specific_methods = [
            "pca", "anova", "t-test", "mann-whitney", "regression",
            "correlation", "fold change", "differential", "clustering",
            "heatmap", "volcano", "pathway", "enrichment", "network",
            "q-learning", "q-table", "rmse", "mae", "forecast",
            "train", "validate", "lambda sweep", "discretize",
            "box plot", "scatter plot", "histogram", "kde",
        ]
        method_count = sum(1 for m in specific_methods if m in text)
        score += min(method_count * 0.10, 0.30)

        # Specific metabolite/variable references
        if any(w in text for w in ["metabolite", "pathway", "group", "cycle"]):
            score += 0.10

        # Vague language penalty
        vague = ["investigate", "explore", "look at", "check", "see if",
                 "maybe", "possibly", "something"]
        vague_count = sum(1 for v in vague if v in text)
        score -= min(vague_count * 0.08, 0.25)

        return max(0.0, min(1.0, score))

    def _score_relevance(self, text: str, objective: str) -> float:
        """Does the task address the research objective?"""
        if not objective:
            return 0.6

        # Tokenize both, removing stop words
        stop_words = {
            "the", "a", "an", "and", "or", "in", "of", "to", "is",
            "for", "with", "that", "this", "be", "by", "as", "on",
            "at", "from", "are", "was", "were", "been", "being",
            "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can",
            "not", "but", "if", "than", "its", "it", "we", "our",
        }
        obj_words = set(objective.lower().split()) - stop_words
        task_words = set(text.split()) - stop_words

        if not obj_words or not task_words:
            return 0.6

        overlap = obj_words & task_words

        # Use the SMALLER set as denominator so long objectives
        # don't crush the score. A task mentioning 5 out of its
        # 15 meaningful words from the objective is very relevant.
        denominator = min(len(obj_words), len(task_words), 30)
        score = len(overlap) / max(denominator, 1)

        # Boost for key scientific terms present in both
        key_terms = [
            "q-learning", "pricing", "agent", "firm", "sector",
            "profit", "price", "heuristic", "forecast", "rmse",
            "lambda", "reinforcement", "heterogeneity", "convergence",
        ]
        key_overlap = sum(
            1 for t in key_terms
            if t in objective.lower() and t in text
        )
        score += key_overlap * 0.05

        return max(0.1, min(1.0, score))

    def _score_novelty(self, text: str, mode: str) -> float:
        """Does the task explore new ground?"""
        # In exploitative mode, novelty matters less — give baseline score
        if mode == "exploitative":
            return 0.65

        score = 0.5

        novel_signals = [
            "novel", "new", "additional", "further", "extend",
            "complement", "validate", "confirm", "beyond"
        ]
        if any(s in text for s in novel_signals):
            score += 0.20

        # Penalty for "repeat" language
        repeat_signals = ["again", "repeat", "same", "re-run", "redo"]
        if any(r in text for r in repeat_signals):
            score -= 0.30

        return max(0.0, min(1.0, score))

    def _score_coverage(self, text: str) -> float:
        """Does the task fill a gap in current knowledge?"""
        score = 0.55

        # Gap-filling language
        gap_signals = [
            "gap", "missing", "unexplored", "not yet", "unclear",
            "understand", "elucidate", "determine", "characterize"
        ]
        if any(g in text for g in gap_signals):
            score += 0.25

        # Follow-up on existing findings
        followup = ["based on", "following", "given", "building on",
                    "extends", "related to finding"]
        if any(f in text for f in followup):
            score += 0.15

        return max(0.0, min(1.0, score))

    def _score_feasibility(self, text: str) -> float:
        """Can this task realistically be executed?"""
        score = 0.7  # most tasks are feasible by default

        # Tasks requiring data not in a typical CSV dataset
        unavailable = [
            "gene expression", "rna-seq", "protein", "imaging",
            "clinical outcome", "survival", "mortality",
            "real-time", "api call", "web scraping", "download",
        ]
        if any(u in text for u in unavailable):
            score -= 0.20

        # Tasks using data/methods that are clearly available
        available = [
            "csv", "dataset", "column", "row", "sector", "firm",
            "correlation", "regression", "t-test", "distribution",
            "q-learning", "q-table", "discretize", "bin",
            "relative_price", "relative_profit", "is_large_firm",
            "profit_margin", "interest_burden", "market_share",
        ]
        if any(a in text for a in available):
            score += 0.15

        return max(0.0, min(1.0, score))

    def _build_reason(
        self,
        specificity: float,
        relevance: float,
        novelty: float,
        coverage: float,
        feasibility: float,
        composite: float,
    ) -> str:
        """Build a human-readable reason string."""
        scores = {
            "specificity": specificity,
            "relevance": relevance,
            "novelty": novelty,
            "coverage": coverage,
            "feasibility": feasibility,
        }
        weakest = min(scores, key=scores.get)
        strongest = max(scores, key=scores.get)
        return (
            f"composite={composite:.2f} | "
            f"strongest={strongest}({scores[strongest]:.2f}) | "
            f"weakest={weakest}({scores[weakest]:.2f})"
        )