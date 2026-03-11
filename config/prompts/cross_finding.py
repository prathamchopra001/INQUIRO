"""
Prompts for Cross-Finding Synthesis.

These prompts help identify themes and patterns across multiple
individual findings, creating higher-level insights.
"""

CROSS_FINDING_SYNTHESIS_PROMPT = """You are a senior scientist synthesizing research findings to identify key themes and patterns.

## Research Objective
{objective}

## Individual Findings
Below are {num_findings} findings discovered during this research. Each has a claim, confidence score, and source attribution.

{findings_text}

---

## Your Task

Analyze ALL the findings above and identify **3-5 major themes** that emerge across them. A theme is a higher-level insight that:
- Connects multiple individual findings
- Reveals a pattern or trend in the evidence
- Answers part of the research objective
- Would be valuable in a Discussion section of a paper

For each theme:
1. Write a clear synthesis claim (1-2 sentences)
2. List the finding IDs that support this theme
3. Note any contradictions or caveats
4. Assess overall confidence (0.0-1.0) based on evidence strength

**Important Guidelines:**
- Each theme should be supported by at least 2 findings
- Look for convergent evidence from different sources (data + literature)
- Identify any contradictions between findings
- Don't just repeat individual findings — synthesize them into higher-level insights
- Themes should directly address the research objective

## Output Format

Return a JSON array of themes:
```json
[
  {{
    "theme_id": "theme_1",
    "title": "Short theme title (5-10 words)",
    "synthesis_claim": "The synthesized insight that emerges from combining the findings below...",
    "supporting_finding_ids": ["f_abc123", "f_def456", "f_ghi789"],
    "evidence_summary": "Brief summary of how the findings support this theme",
    "contradictions": "Any conflicting evidence or caveats (or null if none)",
    "confidence": 0.85,
    "relevance_to_objective": "How this theme addresses the research objective"
  }}
]
```

Return ONLY the JSON array, no other text.
"""

THEME_NARRATIVE_PROMPT = """You are a scientific writer crafting the Discussion section of a research paper.

## Research Objective
{objective}

## Theme to Discuss
**Title:** {theme_title}
**Synthesis Claim:** {synthesis_claim}

## Supporting Evidence
{supporting_findings_text}

## Contradictions/Caveats
{contradictions}

---

Write 2-3 paragraphs discussing this theme for a scientific paper's Discussion section.

Guidelines:
- Start by stating the key insight (the synthesis claim)
- Explain how the individual findings support this conclusion
- Acknowledge any limitations or contradictions
- Connect back to the research objective
- Use academic tone but remain accessible
- Do NOT use bullet points — write flowing prose

Write the narrative paragraphs only, no headers or JSON.
"""
