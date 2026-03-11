"""
Research Plan Generation Prompts

For complex, multi-area research objectives, this prompt decomposes
the objective into a structured research plan BEFORE cycle 1 begins.

Benefits:
1. Ensures all areas get covered (not just the easiest ones)
2. Provides keyword taxonomy for each area
3. Enables coverage tracking across cycles
4. Helps task generation prioritize neglected areas
"""

RESEARCH_PLAN_PROMPT = """You are a research planning assistant decomposing a complex research objective.

RESEARCH OBJECTIVE:
{objective}

Your task: Create a structured research plan that breaks this objective into distinct research areas.

For EACH research area, provide:
1. A clear name (2-5 words)
2. 1-2 specific research questions
3. 4-6 search keywords/phrases that will find relevant papers
4. Success criteria (what finding would "cover" this area?)

RULES:
1. Identify ALL distinct topics/domains mentioned in the objective
2. Don't merge unrelated areas - keep them separate for tracking
3. Keywords should be SPECIFIC enough to find domain-relevant papers
4. Include both technical terms and common synonyms in keywords

Return ONLY a JSON object with this structure:
{{
  "total_areas": <number>,
  "areas": [
    {{
      "id": "area_1",
      "name": "Short descriptive name",
      "questions": [
        "Specific research question 1?",
        "Specific research question 2?"
      ],
      "keywords": [
        "keyword phrase 1",
        "keyword phrase 2",
        "keyword phrase 3",
        "keyword phrase 4"
      ],
      "success_criteria": "What finding would indicate this area is adequately covered",
      "priority": "high" | "medium" | "low"
    }},
    ...
  ],
  "cross_cutting_themes": [
    "Themes that connect multiple areas (if any)"
  ]
}}

IMPORTANT: Return raw JSON only. No markdown, no explanation, no code blocks.
"""


COVERAGE_CHECK_PROMPT = """You are checking research coverage against a research plan.

RESEARCH PLAN AREAS:
{areas_json}

FINDINGS SO FAR:
{findings_summary}

For each research area, determine:
1. How many findings are relevant to this area?
2. Have the research questions been answered?
3. What's still missing?

Return ONLY a JSON object:
{{
  "coverage": [
    {{
      "area_id": "area_1",
      "area_name": "Name",
      "status": "covered" | "partial" | "uncovered",
      "relevant_findings": <count>,
      "questions_answered": ["question 1"] | [],
      "gaps": ["What's still missing"]
    }},
    ...
  ],
  "overall_coverage": <percentage 0-100>,
  "priority_areas": ["area_ids that need attention"],
  "recommendation": "Brief suggestion for next cycle focus"
}}

IMPORTANT: Return raw JSON only. No markdown, no explanation.
"""


MULTI_AREA_DETECTION_PROMPT = """Analyze this research objective to determine if it covers multiple distinct research areas.

OBJECTIVE:
{objective}

Signs of a multi-area objective:
- Numbered sections like (1), (2), (3)
- Semicolons or bullet points separating topics
- Words like "cover", "areas", "domains", "aspects"
- Very long objective (>300 words)
- Multiple distinct fields mentioned (e.g., "ecology AND health AND agriculture")

Return ONLY a JSON object:
{{
  "is_multi_area": true | false,
  "confidence": 0.0-1.0,
  "detected_areas": ["area1", "area2", ...] | [],
  "reasoning": "Brief explanation"
}}
"""


# =============================================================================
# QUESTION DECOMPOSITION - For question-driven research
# =============================================================================

QUESTION_DECOMPOSITION_PROMPT = """You are a research methodology expert creating a structured research plan.

RESEARCH OBJECTIVE:
{objective}

DOMAIN CONTEXT:
{domain_context}

Your task: Create a comprehensive research plan with:
1. 3-5 specific research questions
2. Keyword taxonomy for literature search
3. Methodology candidates (analysis approaches)
4. Success criteria

GOOD research questions:
- Are specific and answerable through data analysis or literature review
- Start with "What", "How", "Why", "Which", "To what extent"
- Can be marked as "answered" when sufficient evidence is found
- Cover different aspects of the objective (breadth)
- Include some deeper questions (depth)

METHODOLOGY CANDIDATES should include:
- Literature methods: systematic review, meta-analysis, narrative synthesis
- Data analysis methods: statistical tests, ML approaches, visualization types
- Be specific to the research domain

Return ONLY a JSON object:
{{
  "research_plan": {{
    "title": "Brief title for this research",
    "domain": "Primary research domain (e.g., medicine, economics, computer science)",
    "scope": "Brief description of research scope and boundaries"
  }},
  "questions": [
    {{
      "question_text": "Specific research question?",
      "keywords": ["keyword1", "keyword2", "keyword3"],
      "area_name": "Research area this belongs to" | null,
      "priority": "high" | "medium" | "low",
      "question_type": "descriptive" | "comparative" | "causal" | "methodological"
    }}
  ],
  "keyword_taxonomy": {{
    "primary_terms": ["core keywords that must appear"],
    "secondary_terms": ["related keywords to expand search"],
    "exclusion_terms": ["terms to exclude from search"]
  }},
  "methodology_candidates": {{
    "literature_methods": [
      {{
        "method": "Method name (e.g., systematic review)",
        "description": "When and why to use this method",
        "applicable_to": ["question_indices that this method addresses"]
      }}
    ],
    "data_analysis_methods": [
      {{
        "method": "Method name (e.g., regression analysis)",
        "description": "When and why to use this method",
        "applicable_to": ["question_indices that this method addresses"],
        "data_requirements": "What data is needed"
      }}
    ]
  }},
  "success_criteria": {{
    "minimum_findings": <number>,
    "questions_to_answer": <number out of total>,
    "quality_threshold": "Description of quality requirements",
    "completion_indicators": ["Specific indicators that research is complete"]
  }},
  "estimated_cycles_needed": <number 2-10>,
  "research_approach": "Brief description of recommended research strategy"
}}

IMPORTANT: Return raw JSON only. No markdown, no code blocks, no explanation.
"""


QUESTION_VALIDATION_PROMPT = """You are evaluating whether research questions have been sufficiently answered.

RESEARCH QUESTIONS:
{questions_json}

FINDINGS COLLECTED:
{findings_summary}

For EACH question, evaluate:
1. Is there at least one finding that directly addresses this question?
2. How confident can we be that the question is answered? (0.0-1.0)
3. What evidence supports the answer?
4. What gaps remain?

STATUS DEFINITIONS:
- "answered": 2+ findings directly address the question with confidence >= 0.7
- "partial": 1 finding addresses it OR confidence 0.4-0.7
- "unanswered": No relevant findings OR confidence < 0.4

Return ONLY a JSON object:
{{
  "evaluations": [
    {{
      "question_id": "q_xxx",
      "question_text": "The question",
      "status": "answered" | "partial" | "unanswered",
      "confidence_score": 0.0-1.0,
      "relevant_finding_ids": ["f_xxx", "f_yyy"] | [],
      "answer_summary": "Brief summary of the answer" | null,
      "gaps": ["What's still missing"] | []
    }},
    ...
  ],
  "overall_progress": {{
    "answered": <count>,
    "partial": <count>,
    "unanswered": <count>,
    "completion_percentage": 0-100
  }},
  "recommendation": "What to focus on next",
  "should_continue": true | false
}}

IMPORTANT: Return raw JSON only. No markdown, no code blocks.
"""


QUESTION_TASK_GENERATION_PROMPT = """You are generating research tasks to answer specific questions.

UNANSWERED QUESTIONS (prioritize these):
{unanswered_questions}

PARTIALLY ANSWERED QUESTIONS (deepen these):
{partial_questions}

EXISTING FINDINGS:
{findings_summary}

Generate {num_tasks} tasks that will help answer the unanswered questions.

RULES:
1. Each task should target a specific question (reference by ID)
2. Prioritize HIGH priority questions first
3. Mix literature searches and data analysis tasks
4. Don't duplicate work already done
5. Be specific about what to search for or analyze

Return ONLY a JSON object:
{{
  "tasks": [
    {{
      "type": "data_analysis" | "literature_search",
      "description": "Specific task description",
      "goal": "What finding would this produce?",
      "target_question_id": "q_xxx",
      "target_question_text": "The question this addresses",
      "priority": "high" | "medium" | "low",
      "search_keywords": ["keyword1", "keyword2"] | null
    }},
    ...
  ]
}}

IMPORTANT: Return raw JSON only.
"""


# =============================================================================
# QUERY GENERATION - For question-driven deep search
# =============================================================================

QUERY_GENERATION_PROMPT = """You are a search query expert. Generate effective literature search queries for a research question.

RESEARCH QUESTION:
{question_text}

QUESTION KEYWORDS:
{question_keywords}

DOMAIN CONTEXT:
{domain_context}

Your task: Generate 3-5 search queries that will find relevant academic papers.

GOOD search queries:
- Are 3-8 words long (not too short, not too long)
- Use domain-specific terminology
- Combine key concepts with operators implicitly
- Include synonyms or alternative phrasings

BAD search queries:
- Too vague ("research on topic")
- Too long (full sentences with articles and prepositions)
- Include stop words (the, a, is, are)
- Use question format ("what is...")

Return ONLY a JSON object:
{{
  "queries": [
    "query 1 keywords here",
    "query 2 keywords here",
    "query 3 alternative phrasing"
  ],
  "primary_terms": ["most", "important", "terms"],
  "synonyms": {{"term1": ["synonym1", "synonym2"]}}
}}

IMPORTANT: Return raw JSON only. No markdown, no explanation.
"""
