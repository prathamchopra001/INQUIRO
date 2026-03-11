"""
Literature Search Agent for Inquiro.

Searches for, downloads, reads, and extracts findings from
scientific papers. Uses Semantic Scholar for search, PyMuPDF
for PDF processing, and ChromaDB for RAG-based retrieval.
"""

import json
import logging
from typing import Optional, List

from src.utils.llm_client import LLMClient
from src.validation.schol_eval import ScholarEval
from src.literature.search import SemanticScholarClient
from src.literature.arxiv_search import ArXivClient
from src.literature.pubmed_search import PubMedClient
from src.literature.openalex import OpenAlexClient
from src.literature.crossref_search import CrossRefClient
from src.literature.core_search import COREClient
from src.literature.dimensions_search import DimensionsClient
from src.literature.unpaywall import resolve_pdf_urls_batch, try_arxiv_url
from src.literature.pdf_parser import PDFParser
from src.literature.rag import RAGSystem
from src.literature.models import Paper
from config.settings import settings
from config.prompts.lit_agent import (
    QUERY_GENERATION_PROMPT,
    PAPER_RANKING_PROMPT,
    FINDING_EXTRACTION_PROMPT,
)
from config.prompts.lit_agent import CORE_TERM_EXTRACTION_PROMPT
from src.literature.domain_anchoring import DomainAnchorExtractor, QueryValidator, improve_query_with_anchors


logger = logging.getLogger(__name__)


class LiteratureSearchAgent:
    """
    Agent that searches, reads, and extracts findings from scientific papers.
    
    This is the literature counterpart to DataAnalysisAgent.
    While the Data Agent runs code to analyze datasets, this agent
    reads papers to find what the scientific community already knows.
    
    Usage:
        agent = LiteratureSearchAgent(llm_client=LLMClient())
        result = agent.execute(
            task={"description": "Find papers on...", "goal": "...", "cycle": 1},
            objective="Identify metabolic pathways..."
        )
        print(result["findings"])
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        search_client: SemanticScholarClient = None,
        arxiv_client: ArXivClient = None,
        pubmed_client: PubMedClient = None,
        openalex_client: OpenAlexClient = None,
        crossref_client: CrossRefClient = None,
        core_client: COREClient = None,
        dimensions_client: DimensionsClient = None,
        pdf_parser: PDFParser = None,
        rag_system: RAGSystem = None,
    ):
        self.llm = llm_client
        self.search_client = search_client or SemanticScholarClient()
        self.arxiv_client = arxiv_client or ArXivClient()
        self.pubmed_client = pubmed_client or PubMedClient()
        self.openalex_client = openalex_client or OpenAlexClient()
        self.crossref_client = crossref_client or CrossRefClient()
        self.core_client = core_client or COREClient()
        self.dimensions_client = dimensions_client or DimensionsClient()
        self.pdf_parser = pdf_parser or PDFParser()
        self.rag = rag_system or RAGSystem()
        self.anchor_extractor = DomainAnchorExtractor(llm_client)

    
    # =========================================================================
    # HELPER: Parse JSON from LLM responses (boilerplate — done for you)
    # =========================================================================
    
    def _parse_json_response(self, response_text: str, fallback=None):
        """
        Safely parse JSON from LLM response.
        
        LLMs sometimes wrap JSON in markdown code blocks or add
        conversational text. This handles those cases.
        """
        text = response_text.strip()
        
        # Strip markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Try to find JSON array or object
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON within the text
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                start = text.find(start_char)
                end = text.rfind(end_char)
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        continue

            # Last resort: response was truncated mid-array (hits max_tokens)
            # Salvage any complete {...} objects before the cutoff
            start = text.find("[")
            if start != -1:
                partial = text[start:]
                objects = []
                depth = 0
                obj_start = None
                for i, ch in enumerate(partial):
                    if ch == "{":
                        if depth == 0:
                            obj_start = i
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and obj_start is not None:
                            try:
                                obj = json.loads(partial[obj_start:i + 1])
                                objects.append(obj)
                            except json.JSONDecodeError:
                                pass
                            obj_start = None
                if objects:
                    logger.warning(
                        f"JSON truncated — salvaged {len(objects)} complete objects"
                    )
                    return objects

        logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
        return fallback if fallback is not None else []
    
    def _extract_core_terms(self, objective: str) -> list:
        """
        Extract key domain-specific terms from the objective for query anchoring.
        This prevents literature searches from drifting to unrelated topics.
        
        Args:
            objective: The research objective string
            
        Returns:
            List of core domain terms that must appear in search queries
        """
        from config.prompts.lit_agent import CORE_TERM_EXTRACTION_PROMPT
        
        prompt = CORE_TERM_EXTRACTION_PROMPT.format(objective=objective)
        
        try:
            # Use fast tier for this simple extraction
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="query_generation",
                max_tokens=300
            )
            
            # Parse JSON response
            import json
            text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract JSON array
            if '[' in text and ']' in text:
                start = text.find('[')
                end = text.rfind(']') + 1
                terms = json.loads(text[start:end])
                if isinstance(terms, list) and len(terms) > 0:
                    logger.info(f"Extracted {len(terms)} core terms for anchoring: {terms[:3]}...")
                    return terms
                    
        except Exception as e:
            logger.warning(f"Core term extraction failed: {e}")
        
        # Fallback: extract significant words from objective
        # Remove common stopwords and short words
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'shall', 'can', 'need', 'this', 'that', 'these',
                     'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
                     'which', 'who', 'whom', 'how', 'when', 'where', 'why', 'with',
                     'from', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'under', 'again', 'further', 'then', 'once',
                     'here', 'there', 'all', 'each', 'few', 'more', 'most', 'other',
                     'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'but', 'if', 'our', 'my', 'your',
                     'their', 'its', 'his', 'her', 'as', 'by', 'about', 'against',
                     'behavior', 'analysis', 'study', 'research', 'investigate', 'examine'}
        
        words = objective.lower().replace('-', ' ').split()
        significant = [w for w in words if len(w) > 4 and w not in stopwords]
        
        # Return unique terms, preserving order
        seen = set()
        result = []
        for w in significant:
            if w not in seen:
                seen.add(w)
                result.append(w)
        
        logger.info(f"Fallback core terms: {result[:5]}")
        return result[:6]
    
    
    def _validate_query_relevance(self, query: str, core_terms: list) -> bool:
        """
        Check if a generated query is related to the research domain.
        
        Uses flexible matching: if ANY significant word from core terms
        appears in the query, it's considered relevant. This prevents
        over-filtering of valid queries that use synonyms or variations.
        
        Args:
            query: The search query string
            core_terms: List of domain-specific terms that should appear
            
        Returns:
            True if query is anchored to the research domain
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Build a set of all significant words from core terms
        # Skip very common words that don't add domain specificity
        skip_words = {'the', 'a', 'an', 'of', 'in', 'on', 'for', 'and', 'or', 'to', 'with'}
        
        core_words = set()
        for term in core_terms:
            for word in term.lower().split():
                if len(word) > 3 and word not in skip_words:
                    core_words.add(word)
                    # Also add common variations
                    if word.endswith('ic'):
                        core_words.add(word[:-2])  # diabetic -> diabet
                    if word.endswith('s'):
                        core_words.add(word[:-1])  # patients -> patient
                    if word.endswith('ion'):
                        core_words.add(word[:-3])  # readmission -> readmiss
        
        # Check if ANY core word (or its stem) appears in query
        for core_word in core_words:
            for query_word in query_words:
                # Exact match or stem match (diabetes matches diabetic)
                if core_word in query_word or query_word in core_word:
                    return True
                # Check if they share a common root (5+ chars)
                min_len = min(len(core_word), len(query_word))
                if min_len >= 5 and core_word[:5] == query_word[:5]:
                    return True
        
        return False
    
    
    def _filter_and_validate_queries(self, queries: list, core_terms: list, 
                                      task_description: str) -> list:
        """
        Filter generated queries to ensure they stay on-topic.
        
        Uses QueryValidator for scoring and improve_query_with_anchors for fixing.
        
        Args:
            queries: List of generated search queries
            core_terms: Domain terms for anchoring
            task_description: Fallback if all queries filtered out
            
        Returns:
            List of validated queries (at least 1 guaranteed)
        """
        # Use QueryValidator if we have cached anchors
        if hasattr(self, '_cached_anchors') and self._cached_anchors:
            validator = QueryValidator(self._cached_anchors, strict_mode=False)
            
            # First pass: filter with scoring
            valid_queries = validator.filter_queries(queries, min_score=0.3)
            filtered_count = len(queries) - len(valid_queries)
            
            if filtered_count > 0:
                logger.warning(f"Filtered {filtered_count}/{len(queries)} off-topic queries")
            
            # If too many filtered, try improving them
            if len(valid_queries) < 2 and len(queries) >= 2:
                logger.info("Attempting to improve filtered queries with domain anchors...")
                improved_queries = [
                    improve_query_with_anchors(q, self._cached_anchors)
                    for q in queries
                ]
                valid_queries = validator.filter_queries(improved_queries, min_score=0.25)
                if valid_queries:
                    logger.info(f"Improved {len(valid_queries)} queries with domain anchors")
            
            if valid_queries:
                return valid_queries
        
        # Fallback: use original validation method
        valid_queries = []
        rejected = []
        
        for query in queries:
            if self._validate_query_relevance(query, core_terms):
                valid_queries.append(query)
            else:
                rejected.append(query)
        
        if rejected:
            logger.warning(f"Filtered {len(rejected)} off-topic queries: {rejected[:2]}...")
        
        # If all queries were filtered, create SHORT anchored fallback
        if not valid_queries:
            logger.warning("All generated queries were off-topic. Using keyword-based fallback.")
            # Try to improve queries with anchors first
            if hasattr(self, '_cached_anchors') and self._cached_anchors:
                improved = [improve_query_with_anchors(q, self._cached_anchors) for q in queries[:2]]
                valid_queries = improved
                logger.info(f"Using anchor-improved queries: {improved}")
            else:
                # Extract key nouns from task description (NOT the full description)
                task_words = [w for w in task_description.lower().split() 
                              if len(w) > 4 and w not in {'conduct', 'literature', 'review', 'identify', 
                                                           'analyze', 'investigate', 'examine', 'study', 
                                                           'research', 'focusing', 'factors', 'patients'}]
                # Combine with core terms for a SHORT query
                all_keywords = list(set(task_words[:2] + core_terms[:3]))
                fallback = ' '.join(all_keywords[:5])
                valid_queries = [fallback]
                logger.info(f"Fallback query: '{fallback}'")
        
        return valid_queries

    # =========================================================================
    # === YOUR CODE HERE === (four methods to implement)
    # =========================================================================
    
    def _generate_queries(self, task_description: str, task_goal: str, objective: str, 
                               world_model_summary: str = "", dataset_summary: str = "") -> list:
        """
        Generate search queries with domain anchoring.
        
        This updated version:
        1. Extracts domain anchors from the objective (cached)
        2. Generates queries with domain context
        3. Validates that queries contain domain terms
        4. Filters out off-topic queries
        5. Uses simpler prompts for local models (qwen3:8b, etc.)
        """
        from config.prompts.lit_agent import QUERY_GENERATION_PROMPT
        
        num_queries = 5
        
        # Step 1: Extract domain anchors (cached per objective)
        if not hasattr(self, '_cached_anchors') or getattr(self, '_cached_objective', None) != objective:
            logger.info("Extracting domain anchors from objective...")
            self._cached_anchors = self.anchor_extractor.extract_anchors(objective)
            self._cached_objective = objective
            logger.info(f"Domain: {self._cached_anchors.get('primary_domain', 'unknown')}")
            logger.info(f"Anchor terms: {self._cached_anchors.get('anchor_terms', [])}")
        
        anchors = self._cached_anchors
        core_terms = anchors.get('anchor_terms', [])
        
        # Step 2: Check if using local model (needs simpler prompt)
        use_local_prompt = self.llm.is_local_tier_for_role("query_generation")
        
        if use_local_prompt:
            # Use simplified few-shot prompt for local models
            from config.prompts.lit_agent_local import QUERY_GENERATION_PROMPT_FEWSHOT
            anchor_terms_str = ", ".join(core_terms[:5]) if core_terms else "general terms from task"
            prompt = QUERY_GENERATION_PROMPT_FEWSHOT.format(
                objective=objective,
                task_description=task_description,
                anchor_terms=anchor_terms_str,
                num_queries=num_queries
            )
            logger.debug("Using simplified prompt for local model")
        else:
            # Standard prompt for cloud models
            domain_anchoring_instruction = self.anchor_extractor.format_anchoring_instruction(anchors)
            prompt = QUERY_GENERATION_PROMPT.format(
                objective=objective,
                task_description=task_description,
                task_goal=task_goal,
                world_model_summary=world_model_summary or "No prior knowledge yet.",
                domain_anchoring_instruction=domain_anchoring_instruction,
                num_queries=num_queries
            )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="query_generation",
                max_tokens=500
            )
            
            text = response.content if hasattr(response, 'content') else str(response)
            
            # Handle empty responses from local models
            if not text or not text.strip():
                logger.warning("LLM returned empty response for query generation")
                raise ValueError("Empty LLM response")
            
            # Parse JSON array with repair for local model quirks
            queries = self._parse_query_json(text, use_local_prompt)
            
            if queries and len(queries) > 0:
                # Validate and filter queries
                validated = self._filter_and_validate_queries(
                    queries, core_terms, task_description
                )
                logger.info(f"Generated {len(validated)} anchored queries")
                return validated
                    
        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
        
        # Fallback: create SHORT keyword-based queries from core terms
        # Do NOT use the full task description - it's too long for search APIs
        logger.warning("Query generation failed. Creating keyword-based fallback queries.")
        
        # Extract key nouns from task description (first 3 significant words)
        task_words = [w for w in task_description.lower().split() 
                      if len(w) > 4 and w not in {'conduct', 'literature', 'review', 'identify', 'analyze', 
                                                   'investigate', 'examine', 'study', 'research', 'focusing',
                                                   'papers', 'about', 'using', 'related', 'between'}]
        task_keywords = task_words[:4]
        
        # Create MULTIPLE query variations (not just one)
        fallback_queries = []
        
        # Query 1: Core terms only
        if core_terms:
            fallback_queries.append(' '.join(core_terms[:4]))
        
        # Query 2: Task keywords only
        if task_keywords:
            fallback_queries.append(' '.join(task_keywords[:4]))
        
        # Query 3: Mix of both
        mixed = list(set(task_keywords[:2] + core_terms[:2]))
        if mixed:
            fallback_queries.append(' '.join(mixed[:5]))
        
        # Query 4: Pairs with "AND" style combination
        if len(core_terms) >= 2:
            fallback_queries.append(f"{core_terms[0]} {core_terms[1]}")
        
        # Deduplicate and filter empty
        fallback_queries = list(dict.fromkeys(q.strip() for q in fallback_queries if q.strip()))
        
        if not fallback_queries:
            # Ultimate fallback: just use first few words of task
            fallback_queries = [' '.join(task_description.split()[:6])]
        
        logger.info(f"Using {len(fallback_queries)} fallback queries: {fallback_queries}")
        return fallback_queries
    
    def _parse_query_json(self, text: str, is_local_model: bool = False) -> list:
        """
        Parse JSON array from LLM response with repair for local model quirks.
        
        Local models often produce:
        - Extra text before/after JSON
        - Missing quotes around strings
        - Trailing commas
        - Single quotes instead of double
        - Newlines inside strings
        
        Args:
            text: Raw LLM response text
            is_local_model: If True, apply more aggressive repair
            
        Returns:
            List of query strings, or empty list if parsing fails
        """
        import json
        import re
        
        # Step 1: Find JSON array boundaries
        if '[' not in text or ']' not in text:
            logger.warning("No JSON array found in response")
            return []
        
        start = text.find('[')
        end = text.rfind(']') + 1
        json_str = text[start:end]
        
        # Step 2: Try direct parse first
        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                return [str(q).strip() for q in result if q]
        except json.JSONDecodeError:
            pass
        
        # Step 3: Local model repairs
        if is_local_model:
            repaired = json_str
            
            # Fix single quotes -> double quotes
            repaired = repaired.replace("'", '"')
            
            # Fix trailing commas before ]
            repaired = re.sub(r',\s*]', ']', repaired)
            
            # Fix newlines inside strings
            repaired = re.sub(r'"\s*\n\s*', '" ', repaired)
            
            # Try parsing repaired JSON
            try:
                result = json.loads(repaired)
                if isinstance(result, list):
                    logger.debug("JSON repaired successfully for local model")
                    return [str(q).strip() for q in result if q]
            except json.JSONDecodeError:
                pass
            
            # Step 4: Fallback - extract quoted strings
            matches = re.findall(r'"([^"]+)"', json_str)
            if matches:
                # Filter out very short or very long strings
                queries = [m.strip() for m in matches if 5 < len(m.strip()) < 100]
                if queries:
                    logger.debug(f"Extracted {len(queries)} queries via regex fallback")
                    return queries
        
        logger.warning("Failed to parse query JSON")
        return []
    
    def _search_and_rank(
        self,
        queries: List[str],
        task_description: str,
        objective: str,
        max_papers: int = None,
    ) -> List[Paper]:
        """
        Search Semantic Scholar with multiple queries, deduplicate, and rank.
        
        Args:
            queries: List of search queries from _generate_queries
            task_description: For context when ranking
            objective: Overall research objective
            max_papers: Max papers to return after ranking. 
                        Defaults to settings.literature.max_papers_to_read
        
        Returns:
            List of Paper objects, sorted by relevance (best first).
        
        Algorithm:
            1. For each query, call self.search_client.search_papers(query)
            2. Collect ALL results into one list
            3. Deduplicate by paper_id (keep first occurrence)
               Hint: use a dict keyed by paper_id
            4. Build a text summary of each paper for the LLM:
               f"ID: {paper.paper_id}\nTitle: {paper.title}\n
                Abstract: {paper.abstract[:300]}\nYear: {paper.year}\n
                Citations: {paper.citation_count}\n\n"
            5. Call LLM with PAPER_RANKING_PROMPT to score each paper
            6. Parse the JSON scores
            7. Assign scores to papers (paper.relevance_score = score)
            8. Sort by relevance_score descending
            9. Return top max_papers
            
        Edge cases:
            - If LLM ranking fails, fall back to sorting by citation_count
            - If a paper_id from LLM doesn't match any paper, skip it
        """
        limit = max_papers or settings.literature.max_papers_to_read
        unique_papers = {}

        # 1, 2, 3: Search all sources and deduplicate
        sources = [
            ("Semantic Scholar", self.search_client),
            ("ArXiv",           self.arxiv_client),
            ("PubMed",          self.pubmed_client),
            ("OpenAlex",        self.openalex_client),
            ("CrossRef",        self.crossref_client),
            ("CORE",            self.core_client),
            ("Dimensions",      self.dimensions_client),
        ]

        for query in queries:
            for source_name, client in sources:
                try:
                    results = client.search_papers(query)
                    new_count = 0
                    for paper in results:
                        if paper.paper_id not in unique_papers:
                            unique_papers[paper.paper_id] = paper
                            new_count += 1
                    if new_count:
                        logger.debug(
                            f"{source_name}: +{new_count} papers for '{query[:40]}'"
                        )
                except Exception as e:
                    logger.warning(f"{source_name} search failed for '{query[:40]}': {e}")

        paper_list = list(unique_papers.values())
        if not paper_list:
            return []

        # ── PDF URL resolution ────────────────────────────────────────
        # Many papers have arXiv versions — construct direct PDF URLs
        arxiv_resolved = 0
        for paper in paper_list:
            if not paper.pdf_url:
                if try_arxiv_url(paper):
                    arxiv_resolved += 1
        if arxiv_resolved:
            logger.info(f"  📎 ArXiv URL constructed for {arxiv_resolved} papers")

        # Use Unpaywall to find free PDFs for remaining papers
        unpaywall_resolved = resolve_pdf_urls_batch(paper_list)
        if unpaywall_resolved:
            logger.info(f"  📎 Unpaywall resolved {unpaywall_resolved} PDF URLs")

        # Count accessibility
        with_pdf = sum(1 for p in paper_list if p.pdf_url)
        logger.info(
            f"  PDF availability: {with_pdf}/{len(paper_list)} papers have PDF URLs"
        )

        # 4: Build summary for LLM
        papers_summary = ""
        for paper in paper_list:
            access_tag = "✅ PDF Available" if paper.pdf_url else "❌ No PDF"
            papers_summary += (
                f"ID: {paper.paper_id}\n"
                f"Title: {paper.title}\n"
                f"Abstract: {paper.abstract[:300] if paper.abstract else 'No abstract available'}\n"
                f"Year: {paper.year}\n"
                f"Citations: {paper.citation_count}\n"
                f"Access: {access_tag}\n\n"
            )

        # 5: Rank using LLM
        prompt = PAPER_RANKING_PROMPT.format(
                task_description=task_description,
                objective=objective,
                papers_text=papers_summary 
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="paper_ranking",
                system="You are an expert scientific reviewer."
            )
            raw_scores = self._parse_json_response(response.content)  # also fix .content!

            # Convert list of {"paper_id": ..., "score": ...} to a lookup dict
            scores_map = {}
            if isinstance(raw_scores, list):
                for item in raw_scores:
                    if isinstance(item, dict) and "paper_id" in item and "score" in item:
                        scores_map[item["paper_id"]] = item["score"]

            for paper in paper_list:
                paper.relevance_score = scores_map.get(paper.paper_id, 0.0)
            
            # 8: Sort by LLM score
            paper_list.sort(key=lambda x: x.relevance_score, reverse=True)
            for paper in paper_list:
                if paper.pdf_url:
                    paper.relevance_score = min(1.0, paper.relevance_score + 0.1)
                else:
                    paper.relevance_score = max(0.0, paper.relevance_score - 0.2)

            # Re-sort after adjustment
            paper_list.sort(key=lambda x: x.relevance_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}. Falling back to citation count.")
            # Fallback
            paper_list.sort(key=lambda x: x.citation_count, reverse=True)

        # Final: boost accessible papers, penalize inaccessible ones
        for paper in paper_list:
            if paper.pdf_url:
                paper.relevance_score = min(1.0, (paper.relevance_score or 0) + 0.1)
            else:
                paper.relevance_score = max(0.0, (paper.relevance_score or 0) - 0.2)
        paper_list.sort(key=lambda x: (x.relevance_score or 0), reverse=True)

        return paper_list[:limit]
    
    def _process_papers(self, papers: List[Paper]) -> int:
        """
        Download, extract, chunk, and store papers in the RAG system.
        
        Args:
            papers: List of ranked Paper objects to process.
        
        Returns:
            Total number of chunks added to RAG.
        
        Algorithm:
            1. For each paper that has a pdf_url:
               a. Call self.pdf_parser.process_paper(paper)
                  (this downloads, extracts text, and chunks)
               b. If chunks were returned, add them to RAG:
                  self.rag.add_chunks(chunks)
               c. Keep a running total of chunks added
            2. Log how many papers were processed and total chunks
            3. Return total chunks added
            
        Note:
            Not every paper will have a PDF URL (paywalled papers).
            Not every PDF will download successfully.
            Not every PDF will produce text (scanned images, etc.).
            This is expected — just log and skip failures.
        """
        total_chunks = 0
        processed_count = 0

        for paper in papers:
            if not paper.pdf_url:
                logger.info(f"Skipping paper {paper.paper_id}: No PDF URL.")
                continue

            try:
                chunks = self.pdf_parser.process_paper(paper)
                if chunks:
                    added = self.rag.add_chunks(chunks)
                    total_chunks += added
                    paper.is_processed = True
                    processed_count += 1
                else:
                    logger.warning(f"No text chunks extracted for paper {paper.paper_id}")
            except Exception as e:
                logger.error(f"Failed to process paper {paper.paper_id}: {e}")
                continue

        logger.info(f"Successfully processed {processed_count}/{len(papers)} papers. Total chunks: {total_chunks}")
        return total_chunks
    
    def _extract_findings(
        self,
        task_description: str,
        task_goal: str,
        objective: str,
        papers: List[Paper],
    ) -> List[dict]:
        """
        Query the RAG system and extract structured findings.
        
        This is where the magic happens — we ask questions about what
        we've read, and the RAG system retrieves relevant passages,
        then the LLM extracts structured findings with citations.
        
        Args:
            task_description: What we were looking for
            task_goal: Expected outcome
            objective: Overall research objective
            papers: Papers we processed (for context)
        
        Returns:
            List of finding dicts, each with:
            - claim, confidence, evidence, tags
            - paper_id, paper_title, doi (for citation)
        
        Algorithm:
            1. Query the RAG system with the task_description:
               results = self.rag.query(task_description, top_k=settings.rag.top_k)
            2. Format the RAG results into a text block for the LLM:
               For each (text, metadata, distance):
                 f"[Paper: {metadata['paper_title']}] [DOI: {metadata['doi']}] 
                  [Paper ID: {metadata['paper_id']}]\n{text}\n\n"
            3. Call LLM with FINDING_EXTRACTION_PROMPT
            4. Parse the JSON response
            5. Validate each finding has required fields
            6. Return the list of findings
            
        Edge case:
            If RAG returns no results (empty collection), return empty list.
        """
        # 1: Query RAG
        results = self.rag.query(task_description, top_k=settings.rag.top_k)
        if not results:
            logger.warning("RAG query returned no results.")
            return []

        # 2: Format context for LLM
        context_block = ""
        for text, metadata, distance in results:
            # Format authors string
            authors = metadata.get('authors', '')
            if not authors:
                authors = "Unknown authors"
            
            context_block += (
                f"[Paper: {metadata.get('paper_title', 'Unknown')}] "
                f"[Authors: {authors}] "
                f"[DOI: {metadata.get('doi', 'N/A')}] "
                f"[Paper ID: {metadata.get('paper_id', 'N/A')}]\n"
                f"{text}\n\n"
            )

        # 3: Extract Findings
        prompt = FINDING_EXTRACTION_PROMPT.format(
            task_description=task_description,
            task_goal=task_goal,
            objective=objective,
            rag_results=context_block
        )

        response = self.llm.complete_for_role(
                prompt=prompt,
                role="literature_extraction",  # Use literature-specific skill
                system="You are a precise scientific literature analyst who always attributes findings to their source papers.",
                max_tokens=8192,
            )
        
        # Debug logging for extraction response
        logger.debug(f"Extraction response length: {len(response.content)} chars")
        if len(response.content) < 50:
            logger.warning(f"Short extraction response: {response.content[:200]}")
        
        findings = self._parse_json_response(response.content)
        
        # Log what we got
        if findings is None:
            logger.warning("Extraction returned None")
            findings = []
        elif isinstance(findings, list) and len(findings) == 0:
            logger.warning(f"Extraction returned empty list. Response preview: {response.content[:300]}...")
        elif isinstance(findings, dict):
            # Handle case where LLM returns {"findings": [...]} instead of [...]
            if "findings" in findings:
                findings = findings.get("findings", [])
                logger.info(f"Unwrapped findings from dict: {len(findings)} items")
            else:
                logger.warning(f"Extraction returned dict without 'findings' key: {list(findings.keys())}")
                findings = []

        # 5: Simple Validation
        schol_eval = ScholarEval(min_score=0.35)  # slightly lower bar for literature
        validated_findings = []
        # Required: claim, confidence, evidence, AND (paper_id OR paper_title)
        core_fields = ["claim", "confidence", "evidence"]

        if isinstance(findings, list):
            for f in findings:
                # Step 1: structural validation - need core fields + source attribution
                has_core = all(field in f for field in core_fields)
                has_source = "paper_id" in f or "paper_title" in f
                
                if not has_core or not has_source:
                    logger.warning(f"Skipping malformed finding (missing fields): {f}")
                    continue
                
                # Normalize: ensure both paper_id and paper_title exist
                if "paper_id" not in f:
                    f["paper_id"] = f.get("paper_title", "unknown")[:50]
                if "paper_title" not in f:
                    f["paper_title"] = f.get("paper_id", "Unknown Paper")

                # Step 2: ScholarEval 8-dimension quality score
                eval_result = schol_eval.evaluate(f, source_type="literature")
                if not eval_result.passes:
                    logger.warning(
                        f"📉 Literature finding rejected by ScholarEval "
                        f"(score={eval_result.composite_score:.2f}): "
                        f"{f.get('claim', '')[:80]}"
                    )
                    continue

                # Attach scores to finding
                f.update(eval_result.to_dict())
                validated_findings.append(f)

        return validated_findings
    
    # =========================================================================
    # MAIN ORCHESTRATOR (boilerplate — done for you)
    # =========================================================================
    
    def execute(
    self,
    task: dict,
    objective: str,
    world_model_summary: str = "",
    dataset_summary: str = "",        # ← ADD THIS
) -> dict:
        """
        Main entry point — run the full literature search pipeline.
        
        This orchestrates everything, just like DataAnalysisAgent.execute().
        
        Args:
            task: Dict with "description", "goal", "cycle" keys
            objective: Overall research objective
            world_model_summary: Current state of knowledge
        
        Returns:
            Dict with:
            - "findings": list of finding dicts
            - "papers_found": total papers from search
            - "papers_processed": papers successfully downloaded/chunked
            - "chunks_added": total chunks in RAG
            - "queries_used": the search queries generated
        """
        logger.info(f"Literature Agent starting: {task['description'][:60]}...")

        empty_result = {
            "findings": [],
            "papers_found": 0,
            "papers_processed": 0,
            "chunks_added": 0,
            "queries_used": [],
        }

        try:
            # Step 1: Generate search queries
            queries = self._generate_queries(
                task_description=task["description"],
                task_goal=task.get("goal", ""),
                objective=objective,
                world_model_summary=world_model_summary,
                dataset_summary=dataset_summary,
            )
            logger.info(f"Generated {len(queries)} search queries")
        except Exception as e:
            logger.error(f"Literature Agent: query generation failed: {e}")
            return empty_result

        try:
            # Step 2: Search and rank papers
            papers = self._search_and_rank(
                queries=queries,
                task_description=task["description"],
                objective=objective,
            )
            logger.info(f"Found and ranked {len(papers)} papers")
        except Exception as e:
            logger.error(f"Literature Agent: search/rank failed: {e}")
            return empty_result

        try:
            # Step 3: Process papers (download, chunk, store in RAG)
            chunks_added = self._process_papers(papers)
            papers_processed = sum(1 for p in papers if p.is_processed)
            logger.info(f"Processed {papers_processed} papers, {chunks_added} chunks added")
        except Exception as e:
            logger.error(f"Literature Agent: paper processing failed: {e}")
            return {
                **empty_result,
                "papers_found": len(papers),
                "queries_used": queries,
            }

        try:
            # Step 4: Extract findings from RAG
            findings = []
            if chunks_added > 0:
                findings = self._extract_findings(
                    task_description=task["description"],
                    task_goal=task.get("goal", ""),
                    objective=objective,
                    papers=papers,
                )
                logger.info(f"Extracted {len(findings)} findings")
            else:
                logger.warning("No chunks added to RAG — skipping finding extraction")
        except Exception as e:
            logger.error(f"Literature Agent: finding extraction failed: {e}")
            findings = []

        return {
            "findings": findings,
            "papers_found": len(papers),
            "papers_processed": papers_processed,
            "chunks_added": chunks_added,
            "queries_used": queries,
        }