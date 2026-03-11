# -*- coding: utf-8 -*-
"""
ScholarEval — 8-dimension finding quality validator.

Scores every finding on 8 scientific quality dimensions before
it enters the world model. Findings below the minimum threshold
are rejected. Scores are stored alongside findings for use in
report ranking and generation.

Dimensions:
    1. statistical_validity    — p-values, effect sizes, sample sizes
    2. reproducibility         — method clarity
    3. novelty                 — adds beyond existing knowledge
    4. significance            — effect size is meaningful
    5. methodological_soundness — appropriate test for data type
    6. evidence_quality        — direct vs. inferred vs. speculative
    7. claim_calibration       — confidence matches evidence
    8. citation_support        — backed by sources
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum composite score to enter the world model
DEFAULT_MIN_SCORE = 0.40

# Dimension weights by source type (must sum to 1.0 each)
# Data analysis findings: emphasize statistical rigor and reproducibility
# Literature findings: emphasize evidence quality and citation support
DIMENSION_WEIGHTS = {
    "data_analysis": {
        "statistical_validity":     0.25,  # Most important for data
        "reproducibility":          0.15,  # Code is reproducible
        "novelty":                  0.10,  # Less critical
        "significance":             0.15,  # Effect size matters
        "methodological_soundness": 0.15,  # Right test for data type
        "evidence_quality":         0.10,  # Direct evidence from code
        "claim_calibration":        0.05,  # Less critical
        "citation_support":         0.05,  # Notebook path sufficient
    },
    "literature": {
        "statistical_validity":     0.15,  # Paper may not have stats
        "reproducibility":          0.10,  # Less applicable
        "novelty":                  0.15,  # Important for lit review
        "significance":             0.15,  # Practical importance
        "methodological_soundness": 0.10,  # Study design
        "evidence_quality":         0.20,  # Very important - is it real science?
        "claim_calibration":        0.05,  # Less critical
        "citation_support":         0.10,  # Must trace to source
    },
    "unknown": {
        "statistical_validity":     0.20,
        "reproducibility":          0.10,
        "novelty":                  0.15,
        "significance":             0.15,
        "methodological_soundness": 0.10,
        "evidence_quality":         0.15,
        "claim_calibration":        0.10,
        "citation_support":         0.05,
    },
}


@dataclass
class ScholarEvalResult:
    """Result of scoring a finding across all 8 dimensions."""
    
    # Individual dimension scores (0.0 - 1.0)
    statistical_validity:     float = 0.5
    reproducibility:          float = 0.5
    novelty:                  float = 0.5
    significance:             float = 0.5
    methodological_soundness: float = 0.5
    evidence_quality:         float = 0.5
    claim_calibration:        float = 0.5
    citation_support:         float = 0.5
    
    # Computed composite score
    composite_score: float = 0.0
    
    # Whether finding passes the minimum threshold
    passes: bool = True
    
    # Human-readable reasons for any penalties
    penalties: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "schol_eval": {
                "statistical_validity":     round(self.statistical_validity, 3),
                "reproducibility":          round(self.reproducibility, 3),
                "novelty":                  round(self.novelty, 3),
                "significance":             round(self.significance, 3),
                "methodological_soundness": round(self.methodological_soundness, 3),
                "evidence_quality":         round(self.evidence_quality, 3),
                "claim_calibration":        round(self.claim_calibration, 3),
                "citation_support":         round(self.citation_support, 3),
                "composite_score":          round(self.composite_score, 3),
                "passes":                   self.passes,
                "penalties":                self.penalties,
            }
        }


class ScholarEval:
    """
    8-dimension quality scorer for scientific findings.
    
    Usage:
        evaluator = ScholarEval(min_score=0.40)
        result = evaluator.evaluate(finding, source_type="data_analysis")
        if result.passes:
            world_model.add_finding(..., schol_eval=result.to_dict())
    """
    
    def __init__(self, min_score: float = DEFAULT_MIN_SCORE):
        self.min_score = min_score
    
    # Hard-reject patterns — findings matching these are immediately failed
    # regardless of other scores. These catch study metadata, not findings.
    HARD_REJECT_PATTERNS = [
        # Ethics/IRB (study approval, not findings)
        r'irb|institutional review board|ethics committee',
        r'approved by|ethics approval|study approval',
        r'informed consent|consent form',
        r'trial registration|clinicaltrials\.gov',
        # Author/affiliation info (not findings)
        r'corresponding author|email:|@.*\.(edu|org|com)',
        r'department of|university of|institute of',
        # Funding/acknowledgments (not findings)
        r'funded by|grant number|funding source|acknowledge',
        r'conflict of interest|competing interest',
        # Paper structure (not findings)
        r'supplementary (material|data|figure|table)',
        r'see (figure|table|appendix|methods)',
    ]

    def evaluate(
        self,
        finding: dict,
        source_type: str = "unknown",  # "data_analysis" or "literature"
    ) -> ScholarEvalResult:
        """
        Score a finding across all 8 dimensions.
        
        Args:
            finding:     Finding dict with claim, evidence, confidence, tags
            source_type: "data_analysis" or "literature"
        
        Returns:
            ScholarEvalResult with per-dimension scores and composite
        """
        result = ScholarEvalResult()
        
        claim    = finding.get("claim", "").lower()
        evidence = finding.get("evidence", "").lower()
        confidence = finding.get("confidence", 0.5)
        tags     = finding.get("tags", [])
        text     = claim + " " + evidence

        # ── Hard reject: immediately fail non-scientific metadata ─────────
        if source_type == "literature":
            for pattern in self.HARD_REJECT_PATTERNS:
                if re.search(pattern, text):
                    result = ScholarEvalResult()
                    result.composite_score = 0.10
                    result.passes = False
                    result.penalties.append(
                        f"Hard reject: matched pattern '{pattern}'"
                    )
                    logger.debug(
                        f"ScholarEval hard reject: {pattern} | {claim[:60]}"
                    )
                    return result
        
        # ── 1. Statistical Validity ───────────────────────────────────────
        result.statistical_validity = self._score_statistical_validity(
            text, finding
        )
        
        # ── 2. Reproducibility ────────────────────────────────────────────
        result.reproducibility = self._score_reproducibility(
            text, source_type
        )
        
        # ── 3. Novelty ────────────────────────────────────────────────────
        result.novelty = self._score_novelty(text, tags)
        
        # ── 4. Significance ───────────────────────────────────────────────
        result.significance = self._score_significance(text, finding)
        
        # ── 5. Methodological Soundness ───────────────────────────────────
        result.methodological_soundness = self._score_methodology(
            text, source_type
        )
        
        # ── 6. Evidence Quality ───────────────────────────────────────────
        result.evidence_quality = self._score_evidence_quality(
            text, source_type, finding
        )
        
        # ── 7. Claim Calibration ──────────────────────────────────────────
        result.claim_calibration = self._score_claim_calibration(
            text, confidence
        )
        
        # ── 8. Citation Support ───────────────────────────────────────────
        result.citation_support = self._score_citation_support(
            finding, source_type
        )
        
        # ── Composite Score (using source-type-specific weights) ──────────
        weights = DIMENSION_WEIGHTS.get(source_type, DIMENSION_WEIGHTS["unknown"])
        result.composite_score = sum(
            getattr(result, dim) * weight
            for dim, weight in weights.items()
        )
        
        # Sanity check: ensure composite is in [0, 1]
        result.composite_score = max(0.0, min(1.0, result.composite_score))
        
        result.passes = result.composite_score >= self.min_score
        
        if not result.passes:
            result.penalties.append(
                f"Composite score {result.composite_score:.2f} below "
                f"threshold {self.min_score:.2f}"
            )
        
        logger.debug(
            f"ScholarEval: composite={result.composite_score:.2f} "
            f"passes={result.passes} | {claim[:60]}"
        )
        
        return result
    
    # =========================================================================
    # DIMENSION SCORERS
    # =========================================================================
    
    def _score_statistical_validity(self, text: str, finding: dict) -> float:
        """Score based on presence of valid statistical evidence."""
        score = 0.5  # neutral baseline
        
        # Boost: has p-value mentioned
        p_matches = re.findall(
            r'p[\s=<>]*([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)', text
        )
        if p_matches:
            try:
                p_val = float(p_matches[0])
                if p_val < 0.001:
                    score += 0.30  # highly significant
                elif p_val < 0.01:
                    score += 0.20
                elif p_val < 0.05:
                    score += 0.10
                else:
                    score -= 0.10  # non-significant, small penalty
            except ValueError:
                pass
        
        # Boost: mentions sample size
        if re.search(r'\bn\s*=\s*\d+', text) or re.search(r'sample size', text):
            score += 0.10
        
        # Boost: mentions confidence interval
        if re.search(r'confidence interval|95%\s*ci|\bci\b', text):
            score += 0.10
        
        # Penalty: vague language with no numbers
        if not re.search(r'\d+\.?\d*', text):
            score -= 0.20
            
        return max(0.0, min(1.0, score))
    
    def _score_reproducibility(self, text: str, source_type: str) -> float:
        """Score based on method clarity and reproducibility signals."""
        score = 0.5
        
        # Data analysis findings are inherently more reproducible
        # (the notebook exists and is traceable)
        if source_type == "data_analysis":
            score += 0.25
        
        # Boost: specific method names mentioned
        methods = [
            "anova", "t-test", "mann-whitney", "pearson", "spearman",
            "pca", "regression", "correlation", "fold change",
            "benjamini-hochberg", "fdr", "bonferroni"
        ]
        if any(m in text for m in methods):
            score += 0.15
        
        # Penalty: speculative language
        speculative = ["may", "might", "could", "possibly", "perhaps",
                       "suggests that", "it is possible"]
        penalty = sum(0.05 for s in speculative if s in text)
        score -= min(penalty, 0.20)
        
        return max(0.0, min(1.0, score))
    
    def _score_novelty(self, text: str, tags: list) -> float:
        """Score based on novelty signals vs. known/common findings."""
        score = 0.5
        
        # Boost: novelty language
        novel_signals = [
            "novel", "first time", "previously unknown", "new finding",
            "unexpectedly", "surprising", "unexpected", "discovery"
        ]
        if any(s in text for s in novel_signals):
            score += 0.25
        
        # Neutral: confirmatory language (not bad, just not novel)
        confirmatory = [
            "consistent with", "confirms", "in line with", "as expected",
            "known to", "well-established", "previously reported"
        ]
        if any(s in text for s in confirmatory):
            score -= 0.15
        
        # Boost: specific mechanistic insight
        if re.search(r'mechanism|pathway|regulates|inhibits|activates|mediates', text):
            score += 0.10
        
        return max(0.0, min(1.0, score))
    
    def _score_significance(self, text: str, finding: dict) -> float:
        """Score practical/biological significance, not just statistical."""
        score = 0.5
        
        # Boost: large effect sizes mentioned
        if re.search(r'fold.change|effect size|cohen', text):
            fc_matches = re.findall(r'(\d+\.?\d*)[- ]fold', text)
            if fc_matches:
                try:
                    fc = float(fc_matches[0])
                    if fc >= 2.0:
                        score += 0.30
                    elif fc >= 1.5:
                        score += 0.15
                    elif fc >= 1.2:
                        score += 0.05
                except ValueError:
                    pass
        
        # Boost: strong correlation
        corr_matches = re.findall(r'r\s*=\s*([0-9.]+)', text)
        if corr_matches:
            try:
                r = abs(float(corr_matches[0]))
                if r >= 0.8:
                    score += 0.25
                elif r >= 0.6:
                    score += 0.15
                elif r >= 0.4:
                    score += 0.05
            except ValueError:
                pass
        
        # Penalty: trivial claims
        trivial = ["slightly", "marginally", "minimal", "negligible", "weak"]
        if any(t in text for t in trivial):
            score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    def _score_methodology(self, text: str, source_type: str) -> float:
        """Score appropriateness of methods used."""
        score = 0.5
        
        if source_type == "data_analysis":
            # Check for multiple comparison correction
            if re.search(r'fdr|benjamini|bonferroni|adjusted p', text):
                score += 0.20
            
            # Check for normality consideration
            if re.search(r'shapiro|normality|non-parametric|mann-whitney', text):
                score += 0.10
            
            # Check for appropriate group comparison
            if re.search(r'control.*treatment|treatment.*control|group comparison', text):
                score += 0.10
            
            # Penalty: simulated data
            if re.search(r'simulated|placeholder|dummy|synthetic', text):
                score -= 0.30
        
        elif source_type == "literature":
            # Literature findings scored on systematic review signals
            if re.search(r'meta-analysis|systematic review|randomized', text):
                score += 0.25
            elif re.search(r'cohort|clinical trial|prospective', text):
                score += 0.15
            elif re.search(r'case study|case report|anecdotal', text):
                score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    def _score_evidence_quality(
        self, text: str, source_type: str, finding: dict
    ) -> float:
        """Score quality of underlying evidence (direct vs. inferred)."""
        score = 0.5
        
        if source_type == "data_analysis":
            # Direct computational evidence — inherently higher quality
            score += 0.20
            # Has actual numbers
            if re.search(r'\d+\.?\d*', text):
                score += 0.10
        
        elif source_type == "literature":
            # Penalize if finding is IRB/ethics approval (not a discovery)
            if re.search(r'irb|ethics committee|institutional review', text):
                score -= 0.60
            
            # Penalize study metadata masquerading as findings
            if re.search(r'approved by|ethics approval|consent form', text):
                score -= 0.50
        
        # Penalty: strong inferential language
        inferential = ["implies", "inferred", "assumed", "presumably"]
        if any(i in text for i in inferential):
            score -= 0.10
        
        return max(0.0, min(1.0, score))
    
    def _score_claim_calibration(
        self, text: str, confidence: float
    ) -> float:
        """Score whether stated confidence matches evidence strength."""
        score = 0.7  # default: assume reasonable calibration
        
        # Check for overclaiming — strong language with weak evidence
        strong_claims = ["proves", "demonstrates conclusively", "definitive",
                         "certain", "guarantees"]
        weak_evidence = ["limited", "preliminary", "small sample", "pilot"]
        
        has_strong_claim = any(s in text for s in strong_claims)
        has_weak_evidence = any(w in text for w in weak_evidence)
        
        if has_strong_claim and has_weak_evidence:
            score -= 0.30
        elif has_strong_claim:
            score -= 0.10
        
        # Check confidence vs. language alignment
        hedged = ["may", "might", "suggests", "indicates", "appears"]
        is_hedged = any(h in text for h in hedged)
        
        if confidence > 0.90 and is_hedged:
            # High confidence but hedged language — miscalibrated
            score -= 0.15
        elif confidence < 0.50 and not is_hedged:
            # Low confidence but assertive language — miscalibrated
            score -= 0.10
        
        return max(0.0, min(1.0, score))
    
    def _score_citation_support(
        self, finding: dict, source_type: str
    ) -> float:
        """Score based on presence of traceable citations."""
        score = 0.5
        
        if source_type == "literature":
            # Literature findings should always have a paper source
            has_paper_id = bool(finding.get("paper_id"))
            has_doi = bool(finding.get("doi"))
            has_title = bool(finding.get("paper_title"))
            
            if has_paper_id and has_doi:
                score = 1.0
            elif has_paper_id or has_doi:
                score = 0.80
            elif has_title:
                score = 0.60
            else:
                score = 0.20  # literature finding with no citation — suspicious
        
        elif source_type == "data_analysis":
            # Data findings are cited via notebook path
            has_notebook = bool(finding.get("notebook_path") or
                                finding.get("source", {}).get("path"))
            score = 0.80 if has_notebook else 0.40
        
        return max(0.0, min(1.0, score))