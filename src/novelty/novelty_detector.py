# -*- coding: utf-8 -*-
"""
Novelty Detector — prevents redundant tasks across research cycles.

Checks each proposed task against:
  1. Already-executed tasks (description similarity)
  2. Existing findings in the world model (claim overlap)

Tasks that are too similar to existing work are flagged as redundant
and either rejected or rewritten to focus on unexplored angles.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Similarity threshold above which a task is considered redundant
DEFAULT_REDUNDANCY_THRESHOLD = 0.65


@dataclass
class NoveltyResult:
    """Result of novelty check for a single task."""
    is_novel: bool
    similarity_score: float       # 0.0 = completely new, 1.0 = exact duplicate
    most_similar_task: str        # description of the most similar past task
    most_similar_finding: str     # claim of the most similar finding
    reason: str                   # human-readable explanation


class NoveltyDetector:
    """
    Detects redundant tasks before they get executed.

    Uses lightweight term-overlap similarity (no embeddings needed)
    which is fast enough to check every proposed task in real time.

    Usage:
        detector = NoveltyDetector(threshold=0.65)

        # Register completed tasks as they finish
        detector.register_task("Perform PCA on metabolomics dataset")
        detector.register_finding("Metabolite A shows 1.5-fold change")

        # Check new tasks before executing
        result = detector.check(proposed_task_description)
        if not result.is_novel:
            print(f"Redundant: {result.reason}")
    """

    def __init__(self, threshold: float = DEFAULT_REDUNDANCY_THRESHOLD):
        self.threshold = threshold
        self._executed_tasks: list[str] = []     # task descriptions
        self._known_findings: list[str] = []     # finding claims

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register_task(self, task_description: str) -> None:
        """Register a completed task description."""
        normalized = self._normalize(task_description)
        if normalized not in self._executed_tasks:
            self._executed_tasks.append(normalized)

    def register_finding(self, claim: str) -> None:
        """Register a finding claim from the world model."""
        normalized = self._normalize(claim)
        if normalized not in self._known_findings:
            self._known_findings.append(normalized)

    def register_batch(
        self,
        tasks: list[str] = None,
        findings: list[str] = None,
    ) -> None:
        """Register multiple tasks and findings at once."""
        for t in (tasks or []):
            self.register_task(t)
        for f in (findings or []):
            self.register_finding(f)

    # =========================================================================
    # CHECKING
    # =========================================================================

    def check(self, task_description: str) -> NoveltyResult:
        """
        Check if a proposed task is novel or redundant.

        Args:
            task_description: The proposed task to check

        Returns:
            NoveltyResult with is_novel=True if the task should proceed
        """
        normalized = self._normalize(task_description)

        # Check against past tasks
        task_sim, most_similar_task = self._best_match(
            normalized, self._executed_tasks
        )

        # Check against known findings
        finding_sim, most_similar_finding = self._best_match(
            normalized, self._known_findings
        )

        # Overall similarity is the max of both
        overall_sim = max(task_sim, finding_sim)
        is_novel = overall_sim < self.threshold

        if not is_novel:
            if task_sim >= finding_sim:
                reason = (
                    f"Too similar to past task "
                    f"(similarity={task_sim:.2f}): '{most_similar_task[:80]}'"
                )
            else:
                reason = (
                    f"Already covered by existing finding "
                    f"(similarity={finding_sim:.2f}): '{most_similar_finding[:80]}'"
                )
        else:
            reason = f"Novel task (max_similarity={overall_sim:.2f})"

        result = NoveltyResult(
            is_novel=is_novel,
            similarity_score=overall_sim,
            most_similar_task=most_similar_task,
            most_similar_finding=most_similar_finding,
            reason=reason,
        )

        logger.debug(
            f"Novelty check: is_novel={is_novel} "
            f"sim={overall_sim:.2f} | {task_description[:60]}"
        )

        return result

    def check_batch(self, task_descriptions: list[str]) -> list[tuple[str, NoveltyResult]]:
        """
        Check a batch of proposed tasks, returning only the novel ones.

        Args:
            task_descriptions: List of proposed task descriptions

        Returns:
            List of (description, NoveltyResult) for ALL tasks,
            so caller can decide what to do with redundant ones.
        """
        results = []
        for desc in task_descriptions:
            result = self.check(desc)
            results.append((desc, result))

        novel_count = sum(1 for _, r in results if r.is_novel)
        redundant_count = len(results) - novel_count

        if redundant_count > 0:
            logger.info(
                f"🔍 Novelty filter: {novel_count}/{len(results)} tasks novel, "
                f"{redundant_count} redundant"
            )

        return results

    # =========================================================================
    # SIMILARITY (lightweight term overlap — no embeddings needed)
    # =========================================================================

    def _normalize(self, text: str) -> str:
        """Lowercase, remove punctuation, normalize whitespace."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize(self, text: str) -> set:
        """Split into meaningful tokens, removing stopwords."""
        stopwords = {
            "the", "a", "an", "and", "or", "in", "of", "to", "is", "on",
            "for", "with", "this", "that", "be", "by", "as", "at", "it",
            "from", "use", "using", "perform", "conduct", "analyze",
            "analysis", "data", "dataset", "between", "across", "all",
            "each", "any", "new", "different", "result", "results",
            "based", "identify"
            # NOTE: kept "group", "groups", "significant", "sample",
            # "samples" as scientific terms that carry meaning
        }
        tokens = set(text.split())
        return tokens - stopwords

    def _jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute Jaccard similarity between two text strings.

        Jaccard = |intersection| / |union|
        Range: 0.0 (no overlap) to 1.0 (identical)
        """
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union)

    def _best_match(
        self, query: str, candidates: list[str]
    ) -> tuple[float, str]:
        """
        Find the most similar candidate to a query string.

        Returns:
            (best_similarity_score, best_matching_text)
        """
        if not candidates:
            return 0.0, ""

        best_score = 0.0
        best_match = ""

        for candidate in candidates:
            score = self._jaccard_similarity(query, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate

        return best_score, best_match

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_stats(self) -> dict:
        """Return current detector state for logging."""
        return {
            "registered_tasks": len(self._executed_tasks),
            "registered_findings": len(self._known_findings),
            "threshold": self.threshold,
        }

    def reset(self) -> None:
        """Clear all registered tasks and findings (for testing)."""
        self._executed_tasks.clear()
        self._known_findings.clear()