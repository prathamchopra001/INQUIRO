# -*- coding: utf-8 -*-
"""
Automated Peer Review for INQUIRO Research Reports.

Evaluates completed research reports using criteria similar to academic
conference peer review (NeurIPS/ICML style). Provides structured feedback
with scores and actionable suggestions.

Review Dimensions:
    1. Soundness (1-4)    — Are claims well-supported by evidence?
    2. Significance (1-4) — Is the research impactful and meaningful?
    3. Novelty (1-4)      — Are findings new or non-obvious?
    4. Clarity (1-4)      — Is the report well-organized and readable?
    
Score Scale (NeurIPS-style):
    4 = Excellent: Among the best
    3 = Good: Accept-worthy
    2 = Fair: Borderline, needs improvement
    1 = Poor: Significant issues

Usage:
    reviewer = PaperReviewer(llm_client)
    review = reviewer.review_report(report_text, objective, findings)
    print(review.summary())
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ReviewDimension:
    """Score and feedback for a single review dimension."""
    name: str
    score: int  # 1-4
    confidence: float  # 0-1
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    @property
    def rating(self) -> str:
        """Human-readable rating."""
        ratings = {4: "Excellent", 3: "Good", 2: "Fair", 1: "Poor"}
        return ratings.get(self.score, "Unknown")


@dataclass
class PeerReviewResult:
    """Complete peer review of a research report."""
    
    # Individual dimension scores
    soundness: ReviewDimension = None
    significance: ReviewDimension = None
    novelty: ReviewDimension = None
    clarity: ReviewDimension = None
    
    # Overall assessment
    overall_score: float = 0.0  # Average of dimensions (1-4 scale)
    overall_recommendation: str = ""  # Accept/Revise/Reject
    executive_summary: str = ""
    key_contributions: List[str] = field(default_factory=list)
    major_concerns: List[str] = field(default_factory=list)
    minor_concerns: List[str] = field(default_factory=list)
    
    # Metadata
    review_timestamp: str = ""
    reviewer_confidence: float = 0.0  # How confident is the review?
    
    def __post_init__(self):
        if not self.review_timestamp:
            self.review_timestamp = datetime.now().isoformat()
    
    @property
    def dimensions(self) -> List[ReviewDimension]:
        """All review dimensions as a list."""
        dims = [self.soundness, self.significance, self.novelty, self.clarity]
        return [d for d in dims if d is not None]
    
    def calculate_overall(self) -> None:
        """Calculate overall score from dimensions."""
        dims = self.dimensions
        if dims:
            self.overall_score = sum(d.score for d in dims) / len(dims)
            
            if self.overall_score >= 3.0:
                self.overall_recommendation = "Accept"
            elif self.overall_score >= 2.0:
                self.overall_recommendation = "Revise"
            else:
                self.overall_recommendation = "Reject"
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "AUTOMATED PEER REVIEW",
            "=" * 60,
            "",
            f"Overall Score: {self.overall_score:.1f}/4.0",
            f"Recommendation: {self.overall_recommendation}",
            f"Reviewer Confidence: {self.reviewer_confidence:.0%}",
            "",
        ]
        
        # Dimension scores
        lines.append("DIMENSION SCORES:")
        lines.append("-" * 40)
        for dim in self.dimensions:
            stars = "★" * dim.score + "☆" * (4 - dim.score)
            lines.append(f"  {dim.name:15} {stars}  ({dim.score}/4 - {dim.rating})")
        lines.append("")
        
        # Executive summary
        if self.executive_summary:
            lines.append("SUMMARY:")
            lines.append("-" * 40)
            lines.append(self.executive_summary)
            lines.append("")
        
        # Key contributions
        if self.key_contributions:
            lines.append("KEY CONTRIBUTIONS:")
            lines.append("-" * 40)
            for contrib in self.key_contributions:
                lines.append(f"  ✓ {contrib}")
            lines.append("")
        
        # Major concerns
        if self.major_concerns:
            lines.append("MAJOR CONCERNS:")
            lines.append("-" * 40)
            for concern in self.major_concerns:
                lines.append(f"  ✗ {concern}")
            lines.append("")
        
        # Suggestions
        all_suggestions = []
        for dim in self.dimensions:
            all_suggestions.extend(dim.suggestions)
        
        if all_suggestions:
            lines.append("SUGGESTIONS FOR IMPROVEMENT:")
            lines.append("-" * 40)
            for i, suggestion in enumerate(all_suggestions[:5], 1):
                lines.append(f"  {i}. {suggestion}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for JSON serialization."""
        return {
            "overall_score": round(self.overall_score, 2),
            "overall_recommendation": self.overall_recommendation,
            "reviewer_confidence": round(self.reviewer_confidence, 2),
            "executive_summary": self.executive_summary,
            "key_contributions": self.key_contributions,
            "major_concerns": self.major_concerns,
            "minor_concerns": self.minor_concerns,
            "dimensions": {
                dim.name.lower(): {
                    "score": dim.score,
                    "rating": dim.rating,
                    "confidence": round(dim.confidence, 2),
                    "strengths": dim.strengths,
                    "weaknesses": dim.weaknesses,
                    "suggestions": dim.suggestions,
                }
                for dim in self.dimensions
            },
            "review_timestamp": self.review_timestamp,
        }


class PaperReviewer:
    """
    Automated peer reviewer for INQUIRO research reports.
    
    Uses LLM to evaluate reports on academic review criteria,
    providing scores and actionable feedback.
    
    Usage:
        reviewer = PaperReviewer(llm_client)
        review = reviewer.review_report(
            report_text=report_content,
            objective=research_objective,
            findings=list_of_findings,
        )
        print(review.summary())
    """
    
    def __init__(self, llm_client, min_report_length: int = 500):
        """
        Args:
            llm_client: LLMClient instance for LLM calls
            min_report_length: Minimum chars for review (skip trivial reports)
        """
        self.llm = llm_client
        self.min_report_length = min_report_length
    
    def review_report(
        self,
        report_text: str,
        objective: str,
        findings: List[Dict] = None,
        include_raw_scores: bool = False,
    ) -> PeerReviewResult:
        """
        Perform peer review on a research report.
        
        Args:
            report_text: Full markdown content of the report
            objective: Original research objective
            findings: Optional list of finding dicts for context
            include_raw_scores: Include raw LLM responses in result
            
        Returns:
            PeerReviewResult with scores and feedback
        """
        # Skip trivial reports
        if len(report_text) < self.min_report_length:
            logger.warning(f"Report too short for review ({len(report_text)} chars)")
            return self._empty_review("Report too short for meaningful review")
        
        # Count findings for context
        findings_count = len(findings) if findings else self._count_findings_in_report(report_text)
        
        # Build review context
        context = self._build_review_context(report_text, objective, findings_count)
        
        # Get dimension scores via LLM
        try:
            soundness = self._evaluate_dimension("soundness", context)
            significance = self._evaluate_dimension("significance", context)
            novelty = self._evaluate_dimension("novelty", context)
            clarity = self._evaluate_dimension("clarity", context)
        except Exception as e:
            logger.error(f"Review failed: {e}")
            return self._empty_review(f"Review failed: {e}")
        
        # Build result
        result = PeerReviewResult(
            soundness=soundness,
            significance=significance,
            novelty=novelty,
            clarity=clarity,
        )
        
        # Generate executive summary
        result.executive_summary = self._generate_executive_summary(result, context)
        
        # Extract key contributions and concerns
        result.key_contributions = self._extract_contributions(result)
        result.major_concerns = self._extract_major_concerns(result)
        result.minor_concerns = self._extract_minor_concerns(result)
        
        # Calculate overall score
        result.calculate_overall()
        
        # Estimate reviewer confidence
        result.reviewer_confidence = self._estimate_confidence(result, findings_count)
        
        logger.info(f"Peer review complete: {result.overall_score:.1f}/4 ({result.overall_recommendation})")
        
        return result
    
    def _build_review_context(
        self,
        report_text: str,
        objective: str,
        findings_count: int,
    ) -> Dict[str, Any]:
        """Build context dict for review prompts."""
        # Truncate report if too long (keep intro and conclusions)
        max_chars = 15000
        if len(report_text) > max_chars:
            # Keep first 60% and last 40%
            first_part = report_text[:int(max_chars * 0.6)]
            last_part = report_text[-int(max_chars * 0.4):]
            report_text = first_part + "\n\n[...content truncated...]\n\n" + last_part
        
        return {
            "report_text": report_text,
            "objective": objective,
            "findings_count": findings_count,
            "report_length": len(report_text),
        }
    
    def _evaluate_dimension(
        self,
        dimension: str,
        context: Dict[str, Any],
    ) -> ReviewDimension:
        """Evaluate a single review dimension using LLM."""
        
        prompts = {
            "soundness": self._get_soundness_prompt(context),
            "significance": self._get_significance_prompt(context),
            "novelty": self._get_novelty_prompt(context),
            "clarity": self._get_clarity_prompt(context),
        }
        
        prompt = prompts.get(dimension)
        if not prompt:
            raise ValueError(f"Unknown dimension: {dimension}")
        
        response = self.llm.complete_for_role(
            prompt=prompt,
            role="scoring",
            system="You are an expert academic peer reviewer evaluating research quality.",
            temperature=0.2,  # Low temperature for consistent scoring
        )
        
        return self._parse_dimension_response(dimension, response.content)
    
    def _get_soundness_prompt(self, context: Dict) -> str:
        """Prompt for evaluating soundness."""
        return f"""Evaluate the SOUNDNESS of this research report.

SOUNDNESS measures whether claims are well-supported by evidence:
- Are conclusions justified by the data/findings presented?
- Is the methodology appropriate for the research questions?
- Are limitations acknowledged?
- Is there appropriate citation of sources?

RESEARCH OBJECTIVE: {context['objective']}

REPORT ({context['findings_count']} findings):
{context['report_text'][:8000]}

Score from 1-4:
4 = Excellent: All claims strongly supported, rigorous methodology
3 = Good: Most claims supported, minor gaps in evidence
2 = Fair: Some unsupported claims, methodological concerns
1 = Poor: Major unsupported claims, serious methodology issues

Respond with JSON only:
{{
    "score": <1-4>,
    "confidence": <0.0-1.0>,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""

    def _get_significance_prompt(self, context: Dict) -> str:
        """Prompt for evaluating significance."""
        return f"""Evaluate the SIGNIFICANCE of this research report.

SIGNIFICANCE measures the importance and impact of findings:
- Do findings advance understanding of the topic?
- Are results practically useful or theoretically important?
- Would this work interest the target audience?
- Does it address an important problem?

RESEARCH OBJECTIVE: {context['objective']}

REPORT ({context['findings_count']} findings):
{context['report_text'][:8000]}

Score from 1-4:
4 = Excellent: Highly impactful, addresses important problem
3 = Good: Meaningful contribution, useful findings
2 = Fair: Limited impact, incremental contribution
1 = Poor: Minimal significance, trivial findings

Respond with JSON only:
{{
    "score": <1-4>,
    "confidence": <0.0-1.0>,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""

    def _get_novelty_prompt(self, context: Dict) -> str:
        """Prompt for evaluating novelty."""
        return f"""Evaluate the NOVELTY of this research report.

NOVELTY measures how new or original the findings are:
- Do findings go beyond common knowledge?
- Are there surprising or non-obvious insights?
- Does it synthesize information in new ways?
- Does it identify gaps or contradictions in existing work?

RESEARCH OBJECTIVE: {context['objective']}

REPORT ({context['findings_count']} findings):
{context['report_text'][:8000]}

Score from 1-4:
4 = Excellent: Highly original, surprising insights
3 = Good: Some novel findings, useful synthesis
2 = Fair: Mostly known information, limited new insights
1 = Poor: No novelty, just restates common knowledge

Respond with JSON only:
{{
    "score": <1-4>,
    "confidence": <0.0-1.0>,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""

    def _get_clarity_prompt(self, context: Dict) -> str:
        """Prompt for evaluating clarity."""
        return f"""Evaluate the CLARITY of this research report.

CLARITY measures how well the report communicates:
- Is the structure logical and easy to follow?
- Is the writing clear and professional?
- Are technical terms explained appropriately?
- Are findings presented coherently?

RESEARCH OBJECTIVE: {context['objective']}

REPORT LENGTH: {context['report_length']} characters

REPORT EXCERPT:
{context['report_text'][:6000]}

Score from 1-4:
4 = Excellent: Exceptionally clear, well-organized
3 = Good: Clear with minor issues
2 = Fair: Some unclear sections, organization issues
1 = Poor: Confusing, poorly organized

Respond with JSON only:
{{
    "score": <1-4>,
    "confidence": <0.0-1.0>,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""


    def _parse_dimension_response(
        self,
        dimension: str,
        response_text: str,
    ) -> ReviewDimension:
        """Parse LLM response into ReviewDimension."""
        try:
            # Find JSON in response
            text = response_text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                
                return ReviewDimension(
                    name=dimension.capitalize(),
                    score=max(1, min(4, int(data.get("score", 2)))),
                    confidence=float(data.get("confidence", 0.5)),
                    strengths=data.get("strengths", [])[:3],
                    weaknesses=data.get("weaknesses", [])[:3],
                    suggestions=data.get("suggestions", [])[:3],
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse {dimension} response: {e}")
        
        # Default fallback
        return ReviewDimension(
            name=dimension.capitalize(),
            score=2,
            confidence=0.3,
            strengths=[],
            weaknesses=["Could not evaluate dimension"],
            suggestions=[],
        )
    
    def _generate_executive_summary(
        self,
        review: PeerReviewResult,
        context: Dict,
    ) -> str:
        """Generate a brief executive summary of the review."""
        prompt = f"""Write a 2-3 sentence executive summary of this peer review.

SCORES:
- Soundness: {review.soundness.score if review.soundness else '?'}/4
- Significance: {review.significance.score if review.significance else '?'}/4
- Novelty: {review.novelty.score if review.novelty else '?'}/4
- Clarity: {review.clarity.score if review.clarity else '?'}/4

KEY STRENGTHS:
{self._format_list(self._extract_contributions(review))}

KEY WEAKNESSES:
{self._format_list(self._extract_major_concerns(review))}

RESEARCH OBJECTIVE: {context['objective']}

Write a concise summary (2-3 sentences). Do not use any formatting or bullet points."""

        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="scoring",
                temperature=0.3,
                max_tokens=200,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Failed to generate executive summary: {e}")
            return f"Report scored {review.overall_score:.1f}/4 overall."
    
    def _extract_contributions(self, review: PeerReviewResult) -> List[str]:
        """Extract key contributions from dimension strengths."""
        contributions = []
        for dim in review.dimensions:
            contributions.extend(dim.strengths)
        # Deduplicate and limit
        seen = set()
        unique = []
        for c in contributions:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)
        return unique[:5]
    
    def _extract_major_concerns(self, review: PeerReviewResult) -> List[str]:
        """Extract major concerns from low-scoring dimensions."""
        concerns = []
        for dim in review.dimensions:
            if dim.score <= 2:
                concerns.extend(dim.weaknesses)
        return concerns[:5]
    
    def _extract_minor_concerns(self, review: PeerReviewResult) -> List[str]:
        """Extract minor concerns from okay-scoring dimensions."""
        concerns = []
        for dim in review.dimensions:
            if dim.score == 3:
                concerns.extend(dim.weaknesses)
        return concerns[:3]
    
    def _estimate_confidence(
        self,
        review: PeerReviewResult,
        findings_count: int,
    ) -> float:
        """Estimate reviewer confidence based on review quality signals."""
        confidence = 0.5  # Base confidence
        
        # More findings = more confident review
        if findings_count >= 20:
            confidence += 0.2
        elif findings_count >= 10:
            confidence += 0.1
        elif findings_count < 5:
            confidence -= 0.1
        
        # Dimension confidence average
        dim_confidences = [d.confidence for d in review.dimensions]
        if dim_confidences:
            confidence += (sum(dim_confidences) / len(dim_confidences) - 0.5) * 0.3
        
        # Penalize if dimensions disagree significantly
        scores = [d.score for d in review.dimensions]
        if scores:
            score_range = max(scores) - min(scores)
            if score_range >= 2:
                confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _count_findings_in_report(self, report_text: str) -> int:
        """Estimate findings count from report text."""
        # Count numbered findings or bullet points in findings section
        import re
        patterns = [
            r'^\d+\.\s+\*\*',  # "1. **Finding..."
            r'^[-•]\s+\*\*',   # "- **Finding..."
            r'Finding\s+\d+:', # "Finding 1:"
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, report_text, re.MULTILINE))
        return max(count, 5)  # Assume at least 5 if we can't detect
    
    def _format_list(self, items: List[str]) -> str:
        """Format list for prompt."""
        if not items:
            return "- None identified"
        return "\n".join(f"- {item}" for item in items[:5])
    
    def _empty_review(self, reason: str) -> PeerReviewResult:
        """Return empty review for error cases."""
        return PeerReviewResult(
            soundness=ReviewDimension(name="Soundness", score=0, confidence=0),
            significance=ReviewDimension(name="Significance", score=0, confidence=0),
            novelty=ReviewDimension(name="Novelty", score=0, confidence=0),
            clarity=ReviewDimension(name="Clarity", score=0, confidence=0),
            overall_score=0,
            overall_recommendation="Error",
            executive_summary=reason,
            reviewer_confidence=0,
        )


def review_report_file(
    report_path: str,
    objective: str,
    llm_client=None,
) -> PeerReviewResult:
    """
    Convenience function to review a report file.
    
    Args:
        report_path: Path to markdown report file
        objective: Research objective
        llm_client: Optional LLMClient (creates one if not provided)
        
    Returns:
        PeerReviewResult
    """
    from pathlib import Path
    
    if llm_client is None:
        from src.utils.llm_client import LLMClient
        llm_client = LLMClient()
    
    report_text = Path(report_path).read_text(encoding="utf-8")
    
    reviewer = PaperReviewer(llm_client)
    return reviewer.review_report(report_text, objective)
