"""
Domain Anchor Extraction Prompts

These prompts extract domain-specific terms from research objectives
to ensure literature search queries stay on-topic.

The problem: Generic queries like "observability engineering" find
software papers when the objective is about ecological monitoring.

The solution: Extract 3-5 domain anchor terms that MUST appear in
every search query to maintain domain relevance.
"""

DOMAIN_ANCHOR_EXTRACTION_PROMPT = """You are extracting domain-specific anchor terms from a research objective.

RESEARCH OBJECTIVE:
{objective}

Your task: Identify 3-5 **domain-specific terms** that MUST appear in literature search queries to keep results on-topic.

RULES:
1. Focus on DOMAIN terms, not methodology terms
   - Good: "biodiversity", "ecological monitoring", "wildlife surveillance"
   - Bad: "framework", "architecture", "analysis", "study"

2. Include FIELD-SPECIFIC vocabulary
   - For ecology: species names, habitat types, ecosystem terms
   - For health: disease names, clinical terms, epidemiological concepts
   - For economics: market types, policy terms, indicator names

3. Exclude GENERIC technical terms that appear across many fields
   - Avoid: "data", "system", "model", "approach", "method"
   - Unless they're combined with domain context: "ecological data" is fine

4. If the objective spans MULTIPLE domains, include terms from each
   - Cross-domain objectives need anchors from both sides
   - Example: "ecological health surveillance" → ["biodiversity", "disease", "One Health"]

5. Prefer SPECIFIC over general
   - "snakebite" is better than "animal incident"
   - "bioacoustic monitoring" is better than "sensor data"

Return ONLY a JSON object with this structure:
{{
  "primary_domain": "the main field (e.g., ecology, economics, health)",
  "anchor_terms": ["term1", "term2", "term3", "term4", "term5"],
  "cross_domain_terms": ["term for secondary domain if applicable"],
  "avoid_terms": ["generic terms that would dilute search results"]
}}

IMPORTANT: Return raw JSON only. No markdown, no explanation, no code blocks.
"""


QUERY_ANCHORING_INSTRUCTION = """
CRITICAL DOMAIN ANCHORING REQUIREMENT:

The research objective is in the domain of: {primary_domain}

Every search query you generate MUST include at least ONE of these domain anchor terms:
{anchor_terms}

AVOID generating queries that ONLY contain these generic terms (they find off-topic papers):
{avoid_terms}

GOOD query pattern: [domain anchor] + [specific concept from task]
BAD query pattern: [generic methodology term] + [generic methodology term]

Examples for ecological monitoring objective:
✓ GOOD: "biodiversity sensor networks real-time monitoring"
✓ GOOD: "wildlife incident detection automated surveillance"  
✗ BAD: "observability engineering distributed systems"
✗ BAD: "data pipeline architecture framework"
"""
