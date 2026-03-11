"""
Enhanced Query Generation Prompts for Local Models (A3)

Local models like qwen3:8b benefit from:
1. Shorter, clearer instructions
2. Concrete examples (few-shot learning)
3. Explicit JSON format reminders

These prompts are used when the model router detects a local model.
"""

# =============================================================================
# FEW-SHOT QUERY GENERATION - Better for local models
# =============================================================================

QUERY_GENERATION_PROMPT_FEWSHOT = """Generate academic search queries for this research task.

RESEARCH OBJECTIVE:
{objective}

TASK: {task_description}

DOMAIN TERMS TO INCLUDE: {anchor_terms}

EXAMPLES OF GOOD QUERIES:
Task: "Find papers on diabetes readmission prediction"
Domain: diabetes, readmission, hospital
Good queries:
["diabetes hospital readmission prediction", "machine learning diabetic patient outcomes", "30-day readmission risk factors diabetes"]

Task: "Literature review on renewable energy adoption"  
Domain: renewable energy, adoption, urban
Good queries:
["renewable energy adoption urban areas", "solar panel residential uptake factors", "clean energy transition barriers cities"]

NOW GENERATE {num_queries} QUERIES FOR THE TASK ABOVE.

RULES:
1. Each query MUST include at least one domain term
2. Keep queries 3-6 words (academic search style)
3. Use field-specific vocabulary

OUTPUT FORMAT - Return ONLY a JSON array:
["query 1", "query 2", "query 3"]

JSON ARRAY:
"""


# =============================================================================
# SIMPLIFIED CORE TERM EXTRACTION - For local models
# =============================================================================

CORE_TERM_EXTRACTION_PROMPT_SIMPLE = """Extract 4-5 key domain terms from this research objective.

OBJECTIVE: {objective}

GOOD TERMS: domain-specific nouns, proper names, technical terms
BAD TERMS: generic words like "study", "analysis", "investigate", "factors"

EXAMPLE:
Objective: "Predicting 30-day hospital readmission in diabetic patients using machine learning"
Terms: ["diabetes", "readmission", "hospital", "machine learning", "patients"]

YOUR TERMS (JSON array only):
"""


# =============================================================================
# SIMPLIFIED FINDING EXTRACTION - For local models
# =============================================================================

FINDING_EXTRACTION_PROMPT_SIMPLE = """Extract research findings from the text below.

RESEARCH GOAL: {objective}
TASK: {task_description}

SOURCE TEXT:
{rag_results}

For each finding, provide:
- claim: What was discovered (include author attribution like "Smith et al. found...")
- confidence: 0.5-0.9 based on evidence strength
- evidence: Supporting quote/data
- paper_id: From the [Paper ID: xxx] tag above
- paper_title: From the [Paper: xxx] tag above
- authors: From the [Authors: xxx] tag above
- tags: 2-3 relevant topic tags

EXAMPLE OUTPUT:
[
  {{
    "claim": "Prior work by Johnson et al. demonstrated that X causes Y with p<0.05",
    "confidence": 0.85,
    "evidence": "The study of 500 patients showed...",
    "paper_id": "abc123",
    "paper_title": "Effects of X on Y",
    "authors": "Johnson, Smith, et al.",
    "tags": ["cause-effect", "clinical"]
  }}
]

Return ONLY a JSON array. If no relevant findings, return: []

JSON FINDINGS:
"""


# =============================================================================
# HELPER: Detect if using local model
# =============================================================================

def should_use_fewshot_prompt(llm_client) -> bool:
    """
    Determine if we should use few-shot prompts based on the model being used.
    
    Local models (Ollama, LM Studio) benefit from few-shot examples.
    Cloud models (OpenAI, Anthropic, Gemini) handle complex prompts better.
    """
    try:
        # Check if router is enabled and what provider is used
        from config.settings import settings
        
        if not settings.router.enabled:
            # Single provider mode - check the main provider
            provider = settings.llm.provider.lower()
            return provider in ("ollama", "lm_studio", "local")
        
        # Router enabled - check the fast/local tiers
        # Query generation uses "query_generation" role which maps to fast tier
        fast_provider = settings.router.fast_provider.lower()
        local_provider = settings.router.local_provider.lower()
        
        # If fast tier is local, use few-shot
        if fast_provider in ("ollama", "lm_studio", "local"):
            return True
        
        return False
        
    except Exception:
        # If we can't determine, default to standard prompt
        return False


def get_query_generation_prompt(llm_client, use_fewshot: bool = None) -> str:
    """
    Get the appropriate query generation prompt based on the model.
    
    Args:
        llm_client: The LLM client (used to detect model type)
        use_fewshot: Override automatic detection. None = auto-detect.
        
    Returns:
        The appropriate prompt template string
    """
    if use_fewshot is None:
        use_fewshot = should_use_fewshot_prompt(llm_client)
    
    if use_fewshot:
        return QUERY_GENERATION_PROMPT_FEWSHOT
    else:
        # Import the standard prompt
        from config.prompts.lit_agent import QUERY_GENERATION_PROMPT
        return QUERY_GENERATION_PROMPT
