"""
Adaptive Research Decomposition Prompts

These prompts enable INQUIRO to scale research depth appropriately based on
objective complexity. The system:

1. Assesses objective complexity (scope, depth, breadth)
2. Generates research pillars (high-level themes)
3. Expands each pillar into specific sub-questions
4. Allocates appropriate depth to each question

This produces 4-24 questions depending on objective scope, resulting in
8-50 page reports that match the research ambition.
"""

# =============================================================================
# COMPLEXITY ASSESSMENT
# =============================================================================

COMPLEXITY_ASSESSMENT_PROMPT = """You are a research methodology expert assessing the complexity of a research objective.

RESEARCH OBJECTIVE:
{objective}

Analyze this objective on these dimensions:

1. SCOPE BREADTH: How many distinct domains/topics does this cover?
   - Single focused question → 1
   - 2-3 related topics → 2
   - Multiple distinct areas → 3
   - Cross-disciplinary comprehensive → 4-5

2. DEPTH REQUIRED: How deep must the investigation go?
   - Quick factual answer → 1
   - Basic understanding → 2
   - Thorough analysis → 3
   - Comprehensive review → 4
   - Exhaustive treatment → 5

3. OUTPUT EXPECTATION: What deliverable is expected?
   - Brief summary (1-5 pages) → 1
   - Short report (5-15 pages) → 2
   - Full report (15-25 pages) → 3
   - Research paper (25-40 pages) → 4
   - Comprehensive review (40+ pages) → 5

4. METHODOLOGICAL COMPLEXITY: What methods are needed?
   - Simple search → 1
   - Literature review → 2
   - Multi-source synthesis → 3
   - Mixed methods → 4
   - Novel methodology → 5

Return ONLY a JSON object:
{{
  "complexity_score": <1-5 overall>,
  "dimensions": {{
    "scope_breadth": <1-5>,
    "depth_required": <1-5>,
    "output_expectation": <1-5>,
    "methodological_complexity": <1-5>
  }},
  "reasoning": "Brief explanation of complexity assessment",
  "recommended_pillars": <2-6 number based on complexity>,
  "recommended_questions_per_pillar": <2-4>,
  "estimated_report_pages": <8-50>,
  "estimated_cycles": <3-15>
}}

IMPORTANT: Return raw JSON only. No markdown, no code blocks.
"""


# =============================================================================
# PILLAR GENERATION
# =============================================================================

PILLAR_GENERATION_PROMPT = """You are a research architect designing the high-level structure of a research investigation.

RESEARCH OBJECTIVE:
{objective}

COMPLEXITY ASSESSMENT:
- Overall complexity: {complexity_score}/5
- Recommended pillars: {num_pillars}
- Scope breadth: {scope_breadth}

Your task: Identify {num_pillars} PILLARS - high-level research themes that together fully address this objective.

PILLAR DESIGN PRINCIPLES:
1. Each pillar should be a distinct conceptual area
2. Pillars should be MECE (Mutually Exclusive, Collectively Exhaustive)
3. Name pillars with clear 2-5 word titles
4. Order pillars logically (background → methods → results → implications)
5. Pillars will each contain 2-4 specific research questions

EXAMPLES OF GOOD PILLARS:
For "AI in healthcare diagnosis":
- "Theoretical Foundations" (ML algorithms, medical imaging)
- "Clinical Applications" (specific use cases, deployed systems)
- "Performance & Validation" (accuracy, clinical trials)
- "Implementation Barriers" (regulatory, technical, adoption)

For "Exercise and cognitive aging":
- "Neurobiological Mechanisms"
- "Exercise Modalities"
- "Dosage Parameters"
- "Individual Differences"

Return ONLY a JSON object:
{{
  "pillars": [
    {{
      "id": "pillar_1",
      "name": "Pillar Name (2-5 words)",
      "description": "What this pillar covers (1-2 sentences)",
      "scope": "Specific boundaries of this pillar",
      "role_in_report": "introduction" | "background" | "methods" | "findings" | "discussion" | "implications",
      "priority": "high" | "medium" | "low",
      "estimated_questions": <2-4>
    }}
  ],
  "pillar_relationships": [
    {{
      "from": "pillar_1",
      "to": "pillar_2",
      "relationship": "builds_on" | "contrasts_with" | "synthesizes_with"
    }}
  ],
  "cross_cutting_themes": ["Themes that span multiple pillars"]
}}

IMPORTANT: Return raw JSON only.
"""


# =============================================================================
# QUESTION EXPANSION
# =============================================================================

QUESTION_EXPANSION_PROMPT = """You are a research question expert expanding a research pillar into specific sub-questions.

OVERALL OBJECTIVE:
{objective}

PILLAR TO EXPAND:
- Name: {pillar_name}
- Description: {pillar_description}
- Scope: {pillar_scope}
- Role in report: {pillar_role}

TARGET: Generate {num_questions} specific, answerable research questions for this pillar.

QUESTION QUALITY CRITERIA:
1. SPECIFIC: Clear enough that "answered/not answered" is determinable
2. SEARCHABLE: Keywords can be extracted for literature search
3. DISTINCT: Each question covers a different aspect
4. PROGRESSIVE: Questions build toward pillar's goal
5. TYPED: Descriptive, comparative, causal, or methodological

QUESTION TYPES:
- Descriptive: "What is X?" "What are the characteristics of Y?"
- Comparative: "How does X compare to Y?" "What are differences between A and B?"
- Causal: "Does X cause Y?" "What mechanisms explain Z?"
- Methodological: "How is X measured?" "What methods are used for Y?"

Return ONLY a JSON object:
{{
  "pillar_id": "{pillar_id}",
  "pillar_name": "{pillar_name}",
  "questions": [
    {{
      "id": "q_001",
      "question_text": "Specific research question?",
      "question_type": "descriptive" | "comparative" | "causal" | "methodological",
      "keywords": ["keyword1", "keyword2", "keyword3", "keyword4"],
      "expected_answer_type": "Definition/list" | "Comparison table" | "Mechanism explanation" | "Evidence summary",
      "priority": "high" | "medium" | "low",
      "estimated_findings_needed": <2-5>
    }}
  ],
  "pillar_synthesis_question": "What overarching insight emerges from these sub-questions?"
}}

IMPORTANT: Return raw JSON only.
"""


# =============================================================================
# DEPTH ALLOCATION
# =============================================================================

DEPTH_ALLOCATION_PROMPT = """You are allocating research depth across questions based on their importance and difficulty.

RESEARCH OBJECTIVE:
{objective}

ALL QUESTIONS:
{questions_json}

TOTAL BUDGET:
- Max papers to process: {max_papers}
- Max cycles available: {max_cycles}
- Target report pages: {target_pages}

Allocate depth to each question considering:
1. Priority (high gets more depth)
2. Complexity (harder questions need more papers)
3. Foundational questions (background gets less depth than findings)
4. User's implicit interest (what seems most important?)

Return ONLY a JSON object:
{{
  "allocations": [
    {{
      "question_id": "q_001",
      "papers_allocated": <2-10>,
      "cycles_allocated": <1-3>,
      "search_depth": "shallow" | "standard" | "deep",
      "rationale": "Brief reason for this allocation"
    }}
  ],
  "high_priority_questions": ["q_001", "q_003"],
  "can_be_brief": ["q_005", "q_006"],
  "total_papers_planned": <sum>,
  "balance_notes": "Any notes on trade-offs made"
}}

IMPORTANT: Return raw JSON only.
"""
