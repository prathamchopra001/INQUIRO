# -*- coding: utf-8 -*-
"""
Finding Deduplicator — prevents duplicate findings from being stored.

Problem it solves:
  The LLM often re-discovers the same statistical result across cycles.
  For example, "govt_consumption correlates with ECB rate at -0.1904"
  appeared as 5 separate findings in a 20-cycle run.

Solution:
  Before storing a new finding, check it against all existing findings
  using two complementary strategies:
    1. Text similarity (Jaccard) — catches similar wording
    2. Numerical fingerprint — extracts key numbers + variable names,
       catching findings that state the same result differently

Usage:
    dedup = FindingDeduplicator(text_threshold=0.60, num_threshold=0.80)

    # Register existing findings
    dedup.register("ECB rate correlates with consumption at r=-0.19")

    # Check a new finding
    result = dedup.check("Consumption shows r=-0.19 correlation with ECB")
    if result.is_duplicate:
        print(f"Skipping: {result.reason}")
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of checking a finding for duplicates."""
    is_duplicate: bool
    similarity_score: float       # highest similarity to any existing finding
    matched_finding: str          # the existing finding it matched
    match_method: str             # "text", "numerical", or "none"
    reason: str


class FindingDeduplicator:
    """
    Detects duplicate findings before they're stored in the world model.
    """

    def __init__(
        self,
        text_threshold: float = 0.50,
        num_threshold: float = 0.80,
    ):
        self.text_threshold = text_threshold
        self.num_threshold = num_threshold
        self._known_findings: list[str] = []
        self._known_fingerprints: list[dict] = []

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(self, claim: str) -> None:
        """Register an existing finding so future checks can compare against it."""
        normalized = self._normalize(claim)
        if normalized and normalized not in self._known_findings:
            self._known_findings.append(normalized)
            self._known_fingerprints.append(self._extract_fingerprint(normalized))

    def register_batch(self, claims: list[str]) -> None:
        """Register multiple findings at once."""
        for claim in claims:
            self.register(claim)

    # =========================================================================
    # CHECKING
    # =========================================================================

    def check(self, claim: str) -> DeduplicationResult:
        """
        Check if a new finding is a duplicate of any existing finding.

        Returns DeduplicationResult with is_duplicate=True if it should be skipped.
        """
        if not self._known_findings:
            return DeduplicationResult(
                is_duplicate=False, similarity_score=0.0,
                matched_finding="", match_method="none",
                reason="No existing findings to compare against",
            )

        normalized = self._normalize(claim)

        # Strategy 1: Text similarity
        text_score, text_match = self._best_text_match(normalized)
        if text_score >= self.text_threshold:
            return DeduplicationResult(
                is_duplicate=True, similarity_score=text_score,
                matched_finding=text_match, match_method="text",
                reason=(
                    f"Text similarity {text_score:.2f} >= {self.text_threshold} "
                    f"with: '{text_match[:80]}...'"
                ),
            )

        # Strategy 2: Numerical fingerprint
        new_fp = self._extract_fingerprint(normalized)
        if new_fp["numbers"] and new_fp["variables"]:
            num_score, num_match = self._best_fingerprint_match(new_fp)
            if num_score >= self.num_threshold:
                return DeduplicationResult(
                    is_duplicate=True, similarity_score=num_score,
                    matched_finding=num_match, match_method="numerical",
                    reason=(
                        f"Numerical fingerprint match {num_score:.2f} >= "
                        f"{self.num_threshold} — same variables and values"
                    ),
                )

        # Not a duplicate
        overall = text_score
        return DeduplicationResult(
            is_duplicate=False, similarity_score=overall,
            matched_finding="", match_method="none",
            reason=f"Novel finding (text_sim={text_score:.2f})",
        )

    # =========================================================================
    # TEXT SIMILARITY
    # =========================================================================

    def _normalize(self, text: str) -> str:
        """Lowercase, remove punctuation, normalize whitespace."""
        text = text.lower()
        text = re.sub(r'[^\w\s\.\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize(self, text: str) -> set:
        """Split into meaningful tokens."""
        stopwords = {
            "the", "a", "an", "and", "or", "in", "of", "to", "is", "on",
            "for", "with", "this", "that", "be", "by", "as", "at", "it",
            "from", "shows", "exhibits", "demonstrates", "reveals",
            "indicates", "suggests", "between", "across", "all", "each",
            "statistically", "significant", "positive", "negative",
        }
        tokens = set(text.split())
        return tokens - stopwords

    def _jaccard(self, a: str, b: str) -> float:
        """Jaccard similarity between two texts."""
        tokens_a = self._tokenize(a)
        tokens_b = self._tokenize(b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def _best_text_match(self, query: str) -> tuple[float, str]:
        """Find the most similar existing finding by text."""
        best_score = 0.0
        best_match = ""
        for known in self._known_findings:
            score = self._jaccard(query, known)
            if score > best_score:
                best_score = score
                best_match = known
        return best_score, best_match

    # =========================================================================
    # NUMERICAL FINGERPRINTING
    # =========================================================================

    def _extract_fingerprint(self, text: str) -> dict:
        """
        Extract a numerical fingerprint from a finding claim.

        A fingerprint consists of:
          - numbers: set of rounded floats found in the text
          - variables: set of column/variable names mentioned

        Two findings with the same variables and same numbers
        are almost certainly stating the same result.
        """
        # Extract all decimal numbers (including negative, scientific notation)
        number_pattern = r'-?\d+\.?\d*(?:e[+-]?\d+)?'
        raw_numbers = re.findall(number_pattern, text)

        # Round to 4 decimal places and deduplicate
        numbers = set()
        for n in raw_numbers:
            try:
                val = round(float(n), 4)
                # Skip trivial numbers (0, 1) and years
                if val in (0.0, 1.0, -1.0) or (1900 < val < 2100):
                    continue
                numbers.add(val)
            except ValueError:
                continue

        # Extract variable/column-like names
        var_pattern = r'[a-z][a-z_]+(?:_[a-z]+)+'  # matches snake_case names
        variables = set(re.findall(var_pattern, text))

        # Also catch common statistical terms paired with numbers
        stat_terms = {"correlation", "pearson", "spearman", "r", "p-value",
                      "t-test", "coefficient", "rmse", "mae"}
        mentioned_stats = stat_terms & set(text.split())

        return {
            "numbers": numbers,
            "variables": variables,
            "stats": mentioned_stats,
        }

    def _fingerprint_similarity(self, fp_a: dict, fp_b: dict) -> float:
        """
        Compare two numerical fingerprints.

        High score means they mention the same variables with the same numbers.
        """
        if not fp_a["numbers"] or not fp_b["numbers"]:
            return 0.0
        if not fp_a["variables"] or not fp_b["variables"]:
            return 0.0

        # Number overlap (Jaccard on rounded values)
        num_inter = fp_a["numbers"] & fp_b["numbers"]
        num_union = fp_a["numbers"] | fp_b["numbers"]
        num_score = len(num_inter) / len(num_union) if num_union else 0.0

        # Variable overlap (Jaccard on variable names)
        var_inter = fp_a["variables"] & fp_b["variables"]
        var_union = fp_a["variables"] | fp_b["variables"]
        var_score = len(var_inter) / len(var_union) if var_union else 0.0

        # Combined: both must match for high confidence
        # Weight numbers slightly more since they're more specific
        return 0.55 * num_score + 0.45 * var_score

    def _best_fingerprint_match(self, query_fp: dict) -> tuple[float, str]:
        """Find the most similar existing finding by numerical fingerprint."""
        best_score = 0.0
        best_match = ""
        for i, known_fp in enumerate(self._known_fingerprints):
            score = self._fingerprint_similarity(query_fp, known_fp)
            if score > best_score:
                best_score = score
                best_match = self._known_findings[i]
        return best_score, best_match

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_stats(self) -> dict:
        return {
            "registered_findings": len(self._known_findings),
            "text_threshold": self.text_threshold,
            "num_threshold": self.num_threshold,
        }

    def reset(self) -> None:
        self._known_findings.clear()
        self._known_fingerprints.clear()
