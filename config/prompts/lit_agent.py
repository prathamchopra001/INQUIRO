"""
Updated Literature Agent Prompts

This file contains the UPDATED prompts for the literature search agent.
The key change is domain anchoring in QUERY_GENERATION_PROMPT.

TO APPLY: Replace the corresponding prompts in your existing
config/prompts/lit_agent.py file.
"""

# =============================================================================
# UPDATED QUERY_GENERATION_PROMPT - Optimized for local model compatibility
# =============================================================================
# Key changes:
# - 8 diverse few-shot examples (was 2)
# - Explicit anti-patterns (what NOT to generate)
# - Domain categories to help local models recognize patterns
# - Stricter output format enforcement

QUERY_GENERATION_PROMPT = """Generate {num_queries} academic search queries.

TASK: {task_description}
GOAL: {task_goal}

DOMAIN CONTEXT:
{domain_anchoring_instruction}

## FEW-SHOT EXAMPLES

Example 1 - Healthcare:
Task: "Find papers on diabetes risk factors"
Queries: ["type 2 diabetes risk factors", "diabetes mellitus predictors", "glycemic control determinants", "insulin resistance causes", "diabetes prevention interventions"]

Example 2 - Ecology:
Task: "Investigate biodiversity monitoring methods"
Queries: ["acoustic biodiversity monitoring", "eDNA species detection", "camera trap wildlife survey", "remote sensing habitat mapping", "citizen science biodiversity"]

Example 3 - Computer Science:
Task: "Find papers on distributed systems observability"
Queries: ["distributed tracing microservices", "OpenTelemetry observability", "service mesh monitoring", "log aggregation kubernetes", "metrics collection distributed"]

Example 4 - Economics:
Task: "Analyze cryptocurrency market dynamics"
Queries: ["bitcoin price volatility", "cryptocurrency market efficiency", "blockchain trading patterns", "crypto exchange liquidity", "digital currency speculation"]

Example 5 - Climate Science:
Task: "Review carbon sequestration methods"
Queries: ["soil carbon sequestration", "ocean carbon capture", "forest carbon sink", "biochar carbon storage", "direct air capture CO2"]

Example 6 - Neuroscience:
Task: "Investigate sleep and memory consolidation"
Queries: ["sleep memory consolidation", "REM sleep learning", "hippocampus sleep replay", "slow wave sleep memory", "sleep deprivation cognition"]

Example 7 - Materials Science:
Task: "Find papers on solar cell efficiency"
Queries: ["perovskite solar cell efficiency", "tandem photovoltaic design", "silicon solar optimization", "thin film photovoltaics", "quantum dot solar cells"]

Example 8 - Public Health:
Task: "Review vaccine hesitancy factors"
Queries: ["vaccine hesitancy determinants", "immunization acceptance barriers", "vaccine misinformation social media", "parental vaccine decision", "COVID vaccine uptake factors"]

## ANTI-PATTERNS (DO NOT generate queries like these)

BAD: "conduct a comprehensive literature review on the topic" (too vague)
BAD: "find relevant papers about the research question" (no domain terms)
BAD: "investigate factors affecting outcomes in patients" (generic)
BAD: "analyze data using machine learning methods" (no specific domain)

## YOUR TURN

Generate {num_queries} search queries for the task above.

RULES:
1. Each query must contain domain-specific terms from the task
2. Keep each query 3-7 words (optimal for academic search)
3. No generic words: "study", "analysis", "research", "investigation"
4. Mix: 2 broad queries + 2 specific queries + 1 methodology query

OUTPUT FORMAT - Return ONLY this JSON array, nothing else:
["query 1", "query 2", "query 3", "query 4", "query 5"]
"""


# =============================================================================
# PAPER_RANKING_PROMPT - Unchanged but included for completeness
# =============================================================================

PAPER_RANKING_PROMPT = """You are ranking academic papers by relevance.

RESEARCH OBJECTIVE:
{objective}

CURRENT TASK:
{task_description}

DOMAIN CONTEXT:
Prioritize papers that are directly relevant to the research domain. Papers about software/engineering methodology should be ranked LOWER unless the objective is specifically about software.

PAPERS TO RANK:
{papers_text}

For each paper, consider:
1. Title relevance to the task goal
2. Abstract match to our research needs
3. Citation count (higher = more influential)
4. Recency (prefer recent for fast-moving fields)
5. Whether the paper has a PDF available (prefer accessible papers)

Return a JSON array of paper IDs in order of relevance (most relevant first):
["paper_id_1", "paper_id_2", ...]

Only include papers that are at least marginally relevant. Exclude completely off-topic papers.

IMPORTANT: Return raw JSON only. No markdown, no explanation.
"""


# =============================================================================
# FINDING_EXTRACTION_PROMPT - With improved attribution
# =============================================================================

FINDING_EXTRACTION_PROMPT = """You are extracting research findings from scientific literature.

RESEARCH OBJECTIVE:
{objective}

CURRENT TASK:
Description: {task_description}
Goal: {task_goal}

RELEVANT TEXT FROM PAPERS:
{rag_results}

Extract findings that address the task goal. For each finding:

1. ATTRIBUTION (CRITICAL):
   - Extract the ACTUAL author names from the [Authors: ...] tag in the context
   - Start claims with "Prior work by Smith et al. found that..." or "According to 'Paper Title', researchers showed..."
   - Use the real author names, NOT placeholder text like "[Authors]"
   - Never present literature findings as your own discoveries

2. CLAIM:
   - State the specific finding clearly and concisely
   - Include quantitative details if available (percentages, effect sizes, p-values)
   - Be precise about what was actually shown vs. speculated

3. EVIDENCE:
   - Quote or paraphrase the supporting text
   - Include methodology details if relevant
   - Note sample sizes, conditions, limitations

4. CONFIDENCE:
   - 0.9+ : Direct empirical finding with strong evidence
   - 0.7-0.9: Well-supported finding, some limitations
   - 0.5-0.7: Suggestive finding, needs more evidence
   - <0.5: Preliminary or speculative

Return ONLY a JSON array:
[
  {{
    "claim": "Prior work by Smith et al. found that X leads to Y",
    "confidence": 0.85,
    "evidence": "Supporting quote or description",
    "paper_id": "The Paper ID from the context above",
    "paper_title": "Full title of source paper",
    "authors": "Smith, Jones, et al.",
    "tags": ["topic1", "topic2"]
  }},
  ...
]

IMPORTANT: 
- Use the Paper ID shown in brackets [Paper ID: xxx] from the context above.
- Use the ACTUAL author names from [Authors: xxx], not placeholder text.

RULES:
- Only extract findings directly supported by the provided text
- Do not hallucinate findings not present in the source material
- If no relevant findings, return empty array: []

IMPORTANT: Return raw JSON only. No markdown, no explanation.
"""


# =============================================================================
# CORE_TERM_EXTRACTION_PROMPT - Extract domain terms for query anchoring
# =============================================================================
# Optimized for local models with few-shot examples

CORE_TERM_EXTRACTION_PROMPT = """Extract 4-6 domain-specific terms from this research objective.

## FEW-SHOT EXAMPLES

Objective: "Investigate how OpenTelemetry can be adapted for ecological monitoring systems"
Terms: ["OpenTelemetry", "ecological monitoring", "biodiversity", "distributed tracing", "sensor networks"]

Objective: "Identify risk factors for hospital readmission in diabetic patients"
Terms: ["hospital readmission", "diabetes", "diabetic patients", "HbA1c", "comorbidity"]

Objective: "Analyze the impact of climate change on coral reef ecosystems"
Terms: ["climate change", "coral reef", "ocean acidification", "bleaching", "marine ecosystem"]

Objective: "Evaluate machine learning approaches for fraud detection in banking"
Terms: ["fraud detection", "banking", "anomaly detection", "transaction monitoring", "financial crime"]

## YOUR TURN

RESEARCH OBJECTIVE:
{objective}

RULES:
- Extract DOMAIN terms (biodiversity, diabetes, OpenTelemetry)
- Include PROPER NOUNS (framework names, organizations)
- EXCLUDE: analysis, study, investigation, research, factors, model, approach

Return ONLY this JSON array:
["term1", "term2", "term3", "term4", "term5"]
"""


# =============================================================================
# QUERY_VALIDATION_PROMPT - Validates query relevance before search
# =============================================================================

QUERY_VALIDATION_PROMPT = """Evaluate if this search query is relevant to the research objective.

RESEARCH OBJECTIVE (domain: {primary_domain}):
{objective}

PROPOSED QUERY:
{query}

DOMAIN ANCHOR TERMS (at least one should appear):
{anchor_terms}

Evaluate:
1. Does the query include domain-relevant terms?
2. Would this query find papers in the target domain?
3. Is it too generic (would find papers from many unrelated fields)?

Return ONLY a JSON object:
{{
  "is_relevant": true | false,
  "relevance_score": 0.0-1.0,
  "has_domain_anchor": true | false,
  "reasoning": "Brief explanation",
  "suggested_improvement": "Better version of query if score < 0.7"
}}
"""
