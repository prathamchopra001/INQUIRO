# -*- coding: utf-8 -*-
"""Prompt template for gap analysis — identifies what's been answered vs what's still open."""

GAP_ANALYSIS_PROMPT = """You are a research gap analyst. Your job is to compare the research objective against what has been discovered so far, and identify what is STILL MISSING.

## Research Objective
{objective}

## Findings So Far (grouped by cycle)
{findings_by_cycle}

## Completed Tasks (what has already been tried)
{completed_tasks}

## Instructions
Analyze the findings and completed tasks against the research objective. Produce a structured gap analysis:

1. **Answered Questions**: What aspects of the research objective have been addressed? List 2-5 key questions that findings have answered, with the finding that answers them.

2. **Open Questions**: What aspects of the research objective are still UNANSWERED? List 3-5 specific, concrete questions that remain open. These should be things that NO existing finding addresses.

3. **Weak Areas**: Which findings have low confidence or lack supporting evidence? List 1-3 claims that need validation or deeper investigation.

4. **Recommended Focus**: Given the gaps above, what should the NEXT cycle focus on? Be specific — name the exact analysis or literature search that would fill the most important gap.

## Output Format
Respond with ONLY a valid JSON object. No explanation, no markdown.

{{
  "answered": [
    {{"question": "What is the relationship between X and Y?", "answered_by": "Brief reference to the finding"}},
  ],
  "open_questions": [
    {{"question": "Specific unanswered question", "importance": "high|medium|low", "suggested_approach": "data_analysis or literature"}},
  ],
  "weak_areas": [
    {{"claim": "The finding that needs validation", "weakness": "Why it's weak (low confidence, no supporting evidence, etc.)"}},
  ],
  "recommended_focus": "1-2 sentence description of what the next cycle should prioritize"
}}"""
