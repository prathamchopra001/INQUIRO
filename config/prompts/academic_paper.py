"""
Prompts for Academic Research Paper Format.

These prompts transform INQUIRO reports into proper academic paper structure
with Abstract, Introduction, Methods, Results, Discussion, and Conclusion.
"""

ABSTRACT_PROMPT = """You are a scientific writer creating an abstract for a research paper.

## Research Objective
{objective}

## Key Findings Summary
{findings_summary}

## Statistics
- Research cycles completed: {cycles_completed}
- Total findings generated: {total_findings}
- Research questions addressed: {questions_answered}/{questions_total}

## Instructions
Write a structured abstract (200-300 words) following this format:

**Background:** One sentence on the research problem and its significance.
**Objective:** State the specific research question investigated.
**Methods:** Briefly describe the autonomous AI research methodology (INQUIRO system).
**Results:** Summarize the 2-3 most significant findings with quantitative details where available.
**Conclusions:** State the main takeaway and implications.

Write as a single flowing paragraph (no headers in the output). Use precise scientific language.
Do NOT use bullet points. Write in third person ("This study investigated..." not "We investigated...").
"""

INTRODUCTION_PROMPT = """You are a scientific writer creating the Introduction section for a research paper.

## Research Objective
{objective}

## Key Findings Preview
{findings_preview}

## Literature Context Available
{has_literature}

## Instructions
Write an Introduction section (3-4 paragraphs) that:

1. **Opening paragraph:** Establish the research problem and its significance. Why does this matter?
   Use the research objective to frame the problem in a broader scientific context.

2. **Background paragraph:** If literature findings are available, briefly mention what prior work 
   has established. If not, describe the gap in knowledge this research addresses.

3. **Objective paragraph:** Clearly state the specific research questions or hypotheses investigated.
   List the key questions if multiple were addressed.

4. **Contribution paragraph:** Preview what this autonomous analysis contributes — mention the 
   methodology (INQUIRO AI system) and hint at the main findings without giving everything away.

Write in flowing academic prose. No bullet points. Use third person.
End with a brief roadmap: "The remainder of this paper presents..."
"""

METHODS_PROMPT = """You are a scientific writer creating the Methods section for a research paper.

## Research Configuration
- Research cycles completed: {cycles_completed}
- Total findings generated: {total_findings}
- Data analysis findings: {data_findings_count}
- Literature findings: {lit_findings_count}
- Relationships identified: {total_relationships}
- Dataset used: {dataset_info}

## Data Sources Searched
{data_sources}

## Instructions
Write a Methods section (3-4 paragraphs) describing:

1. **Study Design:** Describe the autonomous AI research methodology. INQUIRO is an AI scientist 
   system that iteratively generates research tasks, executes data analysis, searches scientific 
   literature, and synthesizes findings into a knowledge graph.

2. **Data Collection:** Describe the data sources used:
   - If a dataset was provided: describe it
   - Literature sources: Semantic Scholar, ArXiv, PubMed, OpenAlex, CrossRef, CORE, Dimensions
   - PDF processing: PyMuPDF extraction, chunking, embedding with sentence-transformers
   - RAG retrieval: ChromaDB vector database for semantic search

3. **Analysis Pipeline:** Describe the iterative cycle process:
   - Task generation by orchestrator agent
   - Parallel execution by data analysis and literature agents
   - Finding extraction and validation (ScholarEval quality scoring)
   - Knowledge integration into world model (SQLite + NetworkX graph)
   - Cross-finding synthesis to identify themes

4. **Quality Assurance:** Mention finding deduplication, confidence scoring, and the semantic 
   matching used to track research question coverage.

Write in past tense ("The system searched...", "Findings were validated...").
Be specific about the number of cycles, findings, etc. using the statistics provided.
"""

RESULTS_INTRO_PROMPT = """You are a scientific writer creating the opening for the Results section.

## Statistics
- Total findings: {total_findings}
- Data analysis discoveries: {data_count}
- Literature findings: {lit_count}
- Research questions answered: {questions_answered}/{questions_total}

## Instructions
Write a brief opening paragraph (2-3 sentences) for the Results section that:
1. States what the analysis produced overall
2. Previews the structure of the results (discoveries, then literature context, then question coverage)
3. Notes the confidence level of findings where relevant

Keep it concise — the detailed findings will follow immediately after.
"""

LIMITATIONS_PROMPT = """You are a scientific writer discussing limitations for a research paper.

## Research Configuration
- Was synthetic data used: {is_synthetic}
- Dataset available: {has_dataset}
- Data analysis performed: {has_data_analysis}
- Literature only: {literature_only}

## Instructions
Write 1-2 paragraphs discussing limitations of this autonomous analysis:

Consider mentioning:
- If synthetic data: findings reflect model assumptions, not real observations
- If literature-only: no original data analysis, findings are summaries of prior work
- Autonomous nature: LLM-based extraction may miss nuances human researchers would catch
- Search coverage: literature search may not capture all relevant papers
- Temporal scope: knowledge cutoff of underlying models

Be honest but not overly self-deprecating. Frame limitations as areas for future work.
"""
