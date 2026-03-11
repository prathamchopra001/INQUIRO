"""
Question-Driven Deep Research (QDDR) System for INQUIRO.

This module implements focused, deep literature search for individual
research questions. Instead of broad searches across the entire objective,
it:

1. Generates focused queries FROM the question text
2. Uses hybrid search (BM25 keywords + dense embeddings)
3. Ranks papers by question relevance
4. Follows citation chains to find seminal papers
5. Stores findings WITH question_id for multi-run persistence

Benefits:
- Domain filtering: Question text naturally constrains search
- Citation chains: Natural when deep in one topic
- Hybrid search: Question keywords + question embedding
- Persistence: question_id as stable anchor for resume

Usage:
    searcher = QuestionDeepSearcher(llm_client, rag_store)
    findings = searcher.search_for_question(question, max_papers=15)
"""

import logging
import re
import math
from typing import List, Dict, Optional, Tuple
from collections import Counter

from src.literature.models import Paper, TextChunk
from src.literature.openalex import OpenAlexClient
from src.literature.search import SemanticScholarClient
from src.literature.pdf_parser import PDFParser
from src.literature.rag import RAGSystem
from src.literature.unpaywall import resolve_pdf_urls_batch
from src.utils.shared_embeddings import get_shared_embedding_model
from config.settings import settings
from config.prompts.lit_agent import FINDING_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


# =============================================================================
# BM25 SCORING
# =============================================================================

class BM25Scorer:
    """
    BM25 scoring for keyword-based relevance ranking.
    
    BM25 is the standard IR algorithm used by Elasticsearch, Lucene, etc.
    It handles term frequency saturation and document length normalization.
    
    Combined with dense embeddings, this gives us hybrid search:
    - BM25: Exact keyword matching (good for technical terms, names)
    - Dense: Semantic similarity (good for paraphrases, concepts)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Document length normalization (0.75 typical)
        """
        self.k1 = k1
        self.b = b
        self._idf_cache: Dict[str, float] = {}
        self._avg_doc_length = 0.0
        self._doc_count = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        if not text:
            return []
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens
    
    def fit(self, documents: List[str]) -> None:
        """
        Build IDF (Inverse Document Frequency) from corpus.
        
        Args:
            documents: List of document texts to use as corpus
        """
        self._doc_count = len(documents)
        if self._doc_count == 0:
            return
        
        # Calculate document frequencies for each term
        doc_freqs: Counter = Counter()
        total_length = 0
        
        for doc in documents:
            tokens = set(self._tokenize(doc))  # Unique tokens per doc
            for token in tokens:
                doc_freqs[token] += 1
            total_length += len(self._tokenize(doc))
        
        self._avg_doc_length = total_length / self._doc_count
        
        # Calculate IDF for each term
        for term, df in doc_freqs.items():
            # Standard BM25 IDF formula
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1)
            self._idf_cache[term] = idf
    
    def score(self, query: str, document: str) -> float:
        """
        Calculate BM25 score for a document given a query.
        
        Args:
            query: Search query
            document: Document text (title + abstract typically)
            
        Returns:
            BM25 relevance score (higher = more relevant)
        """
        if self._doc_count == 0:
            return 0.0
        
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)
        doc_length = len(doc_tokens)
        
        if doc_length == 0:
            return 0.0
        
        # Count term frequencies in document
        tf_dict: Counter = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in tf_dict:
                continue
            
            tf = tf_dict[term]
            idf = self._idf_cache.get(term, 0.0)
            
            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self._avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score


# =============================================================================
# HYBRID SEARCH
# =============================================================================

class HybridSearcher:
    """
    Combines BM25 keyword search with dense embedding search.
    
    Hybrid search gives us the best of both worlds:
    - BM25: Great for exact terms, names, technical jargon
    - Dense: Great for semantic similarity, paraphrases, concepts
    
    The final score is a weighted combination of both.
    """
    
    def __init__(
        self,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
    ):
        """
        Args:
            bm25_weight: Weight for BM25 score (keyword matching)
            dense_weight: Weight for dense embedding similarity
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        self.bm25 = BM25Scorer()
        self._embedding_model = get_shared_embedding_model()
    
    def rank_papers(
        self,
        question_text: str,
        papers: List[Paper],
        question_keywords: List[str] = None,
    ) -> List[Tuple[Paper, float]]:
        """
        Rank papers by hybrid relevance to a research question.
        
        Args:
            question_text: Full question text for semantic matching
            papers: List of papers to rank
            question_keywords: Optional keywords for BM25 boost
            
        Returns:
            List of (paper, score) tuples, sorted by score descending
        """
        if not papers:
            return []
        
        # Build corpus for BM25
        corpus = []
        for p in papers:
            text = f"{p.title or ''} {p.abstract or ''}"
            corpus.append(text)
        
        self.bm25.fit(corpus)
        
        # Build query text (question + keywords)
        query = question_text
        if question_keywords:
            query = f"{query} {' '.join(question_keywords)}"
        
        # Calculate scores
        results = []
        
        # Get question embedding once
        question_emb = None
        if self._embedding_model.is_available():
            embs = self._embedding_model.encode([query])
            if embs is not None:
                question_emb = embs[0]
        
        for i, paper in enumerate(papers):
            doc_text = corpus[i]
            
            # BM25 score (normalized to 0-1 range approximately)
            bm25_score = self.bm25.score(query, doc_text)
            # Normalize BM25 (typical scores range 0-20)
            bm25_norm = min(1.0, bm25_score / 10.0)
            
            # Dense similarity score
            dense_score = 0.0
            if question_emb is not None and doc_text.strip():
                doc_emb = self._embedding_model.encode([doc_text])
                if doc_emb is not None:
                    # Cosine similarity
                    import numpy as np
                    dot = np.dot(question_emb, doc_emb[0])
                    norm1 = np.linalg.norm(question_emb)
                    norm2 = np.linalg.norm(doc_emb[0])
                    if norm1 > 0 and norm2 > 0:
                        dense_score = float(dot / (norm1 * norm2))
            
            # Combine scores
            hybrid_score = (
                self.bm25_weight * bm25_norm +
                self.dense_weight * dense_score
            )
            
            results.append((paper, hybrid_score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


# =============================================================================
# CITATION CHAIN FOLLOWER
# =============================================================================

class CitationChainFollower:
    """
    Follows citation chains to find seminal and related papers.
    
    When you find a highly relevant paper for a question, the papers
    it cites (references) and papers that cite it (citations) are
    likely also relevant.
    
    This helps discover:
    - Seminal papers that established the field
    - Recent papers building on key findings
    - Related work the search might have missed
    """
    
    def __init__(self):
        self._openalex = None
    
    def _get_client(self) -> OpenAlexClient:
        if self._openalex is None:
            # Note: contact_email is optional for OpenAlex - None is fine
            self._openalex = OpenAlexClient(
                email=getattr(settings.literature, 'contact_email', None)
            )
        return self._openalex
    
    def get_references(
        self,
        paper_doi: str,
        limit: int = 10,
    ) -> List[Paper]:
        """
        Get papers that this paper cites (its references).
        
        Args:
            paper_doi: DOI of the source paper
            limit: Maximum references to return
            
        Returns:
            List of Paper objects
        """
        client = self._get_client()
        
        # OpenAlex API: Get work's references
        # /works/{id}/referenced_works returns DOIs of cited papers
        try:
            # First get the paper to find its OpenAlex ID
            paper_url = f"https://doi.org/{paper_doi}"
            response = client._request("GET", f"/works/{paper_url}")
            
            if not response:
                return []
            
            # Get referenced works (papers this one cites)
            ref_ids = response.get("referenced_works", [])[:limit]
            
            papers = []
            for ref_id in ref_ids:
                # Fetch each reference
                paper = client._parse_work(
                    client._request("GET", f"/works/{ref_id.replace('https://openalex.org/', '')}")
                )
                if paper:
                    papers.append(paper)
            
            logger.info(f"Found {len(papers)} references for DOI:{paper_doi}")
            return papers
            
        except Exception as e:
            logger.warning(f"Failed to get references for {paper_doi}: {e}")
            return []
    
    def get_citations(
        self,
        paper_doi: str,
        limit: int = 10,
    ) -> List[Paper]:
        """
        Get papers that cite this paper.
        
        Args:
            paper_doi: DOI of the source paper
            limit: Maximum citations to return
            
        Returns:
            List of Paper objects
        """
        client = self._get_client()
        
        try:
            # OpenAlex API: Filter works by "cites:{id}"
            paper_url = f"https://doi.org/{paper_doi}"
            
            # Get the OpenAlex ID first
            response = client._request("GET", f"/works/{paper_url}")
            if not response or "id" not in response:
                return []
            
            openalex_id = response["id"]
            
            # Search for papers that cite this one
            params = {
                "filter": f"cites:{openalex_id}",
                "per_page": min(limit, 50),
                "sort": "cited_by_count:desc",  # Most influential first
            }
            
            results = client._request("GET", "/works", params=params)
            
            papers = []
            for item in results.get("results", [])[:limit]:
                paper = client._parse_work(item)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Found {len(papers)} citations for DOI:{paper_doi}")
            return papers
            
        except Exception as e:
            logger.warning(f"Failed to get citations for {paper_doi}: {e}")
            return []
    
    def follow_chain(
        self,
        seed_papers: List[Paper],
        depth: int = 1,
        max_per_paper: int = 5,
        direction: str = "both",
    ) -> List[Paper]:
        """
        Follow citation chains from seed papers.
        
        Args:
            seed_papers: Papers to start from (should be highly relevant)
            depth: How many hops to follow (1 = direct citations only)
            max_per_paper: Max citations/references per seed paper
            direction: "references", "citations", or "both"
            
        Returns:
            List of discovered papers (deduplicated)
        """
        seen_dois = set()
        for p in seed_papers:
            if p.doi:
                seen_dois.add(p.doi.lower())
        
        discovered = []
        current_seeds = seed_papers
        
        for hop in range(depth):
            next_seeds = []
            
            for paper in current_seeds:
                if not paper.doi:
                    continue
                
                # Get references (papers this one cites)
                if direction in ("references", "both"):
                    refs = self.get_references(paper.doi, limit=max_per_paper)
                    for ref in refs:
                        if ref.doi and ref.doi.lower() not in seen_dois:
                            seen_dois.add(ref.doi.lower())
                            discovered.append(ref)
                            next_seeds.append(ref)
                
                # Get citations (papers that cite this one)
                if direction in ("citations", "both"):
                    cites = self.get_citations(paper.doi, limit=max_per_paper)
                    for cite in cites:
                        if cite.doi and cite.doi.lower() not in seen_dois:
                            seen_dois.add(cite.doi.lower())
                            discovered.append(cite)
                            next_seeds.append(cite)
            
            current_seeds = next_seeds
            logger.info(f"Citation chain hop {hop + 1}: discovered {len(next_seeds)} new papers")
        
        return discovered


# =============================================================================
# MAIN: QUESTION DEEP SEARCHER
# =============================================================================

class QuestionDeepSearcher:
    """
    Deep, focused literature search for a single research question.
    
    This is the main class that orchestrates:
    1. Query generation from question
    2. Hybrid search (BM25 + dense)
    3. Paper ranking by question relevance
    4. PDF download and extraction
    5. Citation chain following
    6. Finding extraction with question_id linkage
    
    Usage:
        searcher = QuestionDeepSearcher(llm_client, rag_store)
        findings = searcher.search_for_question(question, max_papers=15)
    """
    
    def __init__(
        self,
        llm_client,
        rag_store=None,
        pdf_cache_dir: str = None,
    ):
        """
        Args:
            llm_client: LLM client for query generation and extraction
            rag_store: RAGSystem for chunk storage and retrieval
            pdf_cache_dir: Directory for caching downloaded PDFs
        """
        self.llm = llm_client
        self.rag = rag_store or RAGSystem()
        self.pdf_cache_dir = pdf_cache_dir or "./data/pdf_cache"
        
        # Initialize components
        self._hybrid_searcher = HybridSearcher()
        self._citation_follower = CitationChainFollower()
        self._openalex = None
        self._semantic_scholar = None
        self._pdf_parser = PDFParser()
    
    def _get_openalex(self) -> OpenAlexClient:
        if self._openalex is None:
            # Note: contact_email is optional for OpenAlex - None is fine
            self._openalex = OpenAlexClient(
                email=getattr(settings.literature, 'contact_email', None)
            )
        return self._openalex
    
    def _get_semantic_scholar(self) -> SemanticScholarClient:
        if self._semantic_scholar is None:
            self._semantic_scholar = SemanticScholarClient(
                api_key=settings.literature.semantic_scholar_api_key
            )
        return self._semantic_scholar
    
    def _generate_queries_from_question(
        self,
        question_text: str,
        question_keywords: List[str] = None,
        num_queries: int = 4,
        use_llm: bool = True,
    ) -> List[str]:
        """
        Generate search queries from a research question.
        
        Uses both heuristics and LLM to create focused queries.
        
        Args:
            question_text: The research question
            question_keywords: Keywords from research plan
            num_queries: Maximum queries to generate
            use_llm: Whether to use LLM for query generation
            
        Returns:
            List of search query strings
        """
        queries = []
        
        # Query 1: Direct from question (truncated)
        direct = question_text.replace("?", "").strip()
        if len(direct) > 60:
            direct = " ".join(direct[:60].split()[:-1])
        queries.append(direct)
        
        # Query 2: Keywords only
        if question_keywords:
            kw_query = " ".join(question_keywords[:5])
            if kw_query and kw_query not in queries:
                queries.append(kw_query)
        
        # Query 3: LLM-generated queries
        if use_llm and self.llm:
            try:
                llm_queries = self._generate_queries_via_llm(
                    question_text, question_keywords
                )
                for q in llm_queries:
                    if q and q not in queries:
                        queries.append(q)
            except Exception as e:
                logger.warning(f"LLM query generation failed: {e}")
        
        # Query 4: Extract key noun phrases (fallback)
        stop_words = {"what", "which", "where", "when", "does", "have", "that", "this", 
                      "with", "from", "about", "between", "among", "through", "could",
                      "would", "should", "their", "there", "these", "those"}
        words = re.findall(r'\b[a-z]{4,}\b', question_text.lower())
        key_terms = [w for w in words if w not in stop_words][:4]
        if key_terms:
            np_query = " ".join(key_terms)
            if np_query not in queries:
                queries.append(np_query)
        
        return queries[:num_queries]
    
    def _generate_queries_via_llm(
        self,
        question_text: str,
        question_keywords: List[str] = None,
    ) -> List[str]:
        """
        Use LLM to generate search queries for a question.
        
        Returns:
            List of query strings
        """
        from config.prompts.research_plan import QUERY_GENERATION_PROMPT
        import json
        
        prompt = QUERY_GENERATION_PROMPT.format(
            question_text=question_text,
            question_keywords=", ".join(question_keywords or []),
            domain_context="Academic literature search",
        )
        
        # Use fast tier for query generation
        if hasattr(self.llm, 'complete_for_role'):
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="query_formulation",
                max_tokens=500,
            )
            text = response.content if hasattr(response, 'content') else str(response)
        else:
            text = self.llm.complete(prompt)
        
        # Parse JSON response
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        
        try:
            result = json.loads(text)
            return result.get("queries", [])
        except json.JSONDecodeError:
            # Try to extract queries array
            match = re.search(r'"queries"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if match:
                items = re.findall(r'"([^"]+)"', match.group(1))
                return items
            return []
    
    def search_for_question(
        self,
        question,  # ResearchQuestion object
        max_papers: int = 15,
        follow_citations: bool = True,
        citation_depth: int = 1,
        objective: str = "",
    ) -> Dict:
        """
        Perform deep search for a single research question.
        
        Args:
            question: ResearchQuestion object with text, keywords, etc.
            max_papers: Maximum papers to process
            follow_citations: Whether to follow citation chains
            citation_depth: How many hops to follow (1 = direct only)
            objective: Overall research objective for context
            
        Returns:
            Dict with:
            - findings: List of extracted findings
            - papers_found: Total papers from search
            - papers_processed: Papers with PDFs downloaded/chunked
            - chunks_added: Total chunks in RAG
            - question_id: The question this search was for
        """
        # Handle both dict and object formats
        if isinstance(question, dict):
            question_text = question.get("question_text", "")
            question_keywords = question.get("keywords", []) or []
            question_id = question.get("id", "unknown")
        else:
            question_text = question.question_text
            question_keywords = getattr(question, 'keywords', []) or []
            question_id = question.id
        
        logger.info(f"🔍 Deep search for question: {question_text[:60]}...")
        
        # Step 1: Generate queries from question
        queries = self._generate_queries_from_question(
            question_text, question_keywords
        )
        logger.info(f"  Generated {len(queries)} search queries")
        
        # Step 2: Search multiple sources
        all_papers = []
        client = self._get_openalex()
        
        for query in queries:
            try:
                papers = client.search_papers(query, limit=20)
                all_papers.extend(papers)
            except Exception as e:
                logger.warning(f"  Search failed for '{query}': {e}")
        
        # Deduplicate by DOI
        seen_dois = set()
        unique_papers = []
        for p in all_papers:
            key = (p.doi.lower() if p.doi else p.paper_id) or p.title.lower()[:50]
            if key not in seen_dois:
                seen_dois.add(key)
                unique_papers.append(p)
        
        logger.info(f"  Found {len(unique_papers)} unique papers from search")
        
        if not unique_papers:
            return {
                "findings": [],
                "papers_found": 0,
                "papers_processed": 0,
                "chunks_added": 0,
                "question_id": question_id,
                "queries_used": queries,
            }
        
        # Step 3: Hybrid ranking by question relevance
        ranked = self._hybrid_searcher.rank_papers(
            question_text=question_text,
            papers=unique_papers,
            question_keywords=question_keywords,
        )
        
        # Take top papers
        top_papers = [p for p, score in ranked[:max_papers]]
        logger.info(f"  Ranked and selected top {len(top_papers)} papers")
        
        # Step 4: Follow citation chains for top 3 papers
        citation_papers_found = 0
        if follow_citations and len(top_papers) >= 3:
            seed_papers = [p for p in top_papers[:3] if p.doi]
            if seed_papers:
                citation_papers = self._citation_follower.follow_chain(
                    seed_papers=seed_papers,
                    depth=citation_depth,
                    max_per_paper=5,
                    direction="both",
                )
                citation_papers_found = len(citation_papers)
                
                # Rank citation papers too
                if citation_papers:
                    citation_ranked = self._hybrid_searcher.rank_papers(
                        question_text=question_text,
                        papers=citation_papers,
                        question_keywords=question_keywords,
                    )
                    # Add top citation papers
                    for p, score in citation_ranked[:5]:
                        key = (p.doi.lower() if p.doi else p.paper_id) or ""
                        if key and key not in seen_dois:
                            top_papers.append(p)
                            seen_dois.add(key)
                
                logger.info(f"  Added papers from citation chains: {citation_papers_found}")
        
        # Step 5: Download PDFs and process into chunks
        chunks_added = self._process_papers_for_question(
            papers=top_papers,
            question_id=question_id,
        )
        papers_processed = sum(1 for p in top_papers if p.is_processed)
        logger.info(f"  Processed {papers_processed} papers, {chunks_added} chunks")
        
        # Step 6: Extract findings from RAG with question context
        findings = []
        if chunks_added > 0:
            findings = self._extract_findings_for_question(
                question_text=question_text,
                question_keywords=question_keywords,
                question_id=question_id,
                objective=objective,
                papers=top_papers,
            )
            logger.info(f"  Extracted {len(findings)} findings for question")
        
        return {
            "findings": findings,
            "papers_found": len(unique_papers),
            "papers_processed": papers_processed,
            "chunks_added": chunks_added,
            "question_id": question_id,
            "queries_used": queries,
            "citation_papers_found": citation_papers_found,
        }
    
    def _process_papers_for_question(
        self,
        papers: List[Paper],
        question_id: str,
    ) -> int:
        """
        Download PDFs and store chunks with question_id metadata.
        
        Args:
            papers: Papers to process
            question_id: Question ID to link chunks to
            
        Returns:
            Number of chunks added to RAG
        """
        # Resolve PDF URLs using Unpaywall
        papers_with_urls = [p for p in papers if not p.pdf_url]
        if papers_with_urls:
            dois = [p.doi for p in papers_with_urls if p.doi]
            if dois:
                try:
                    resolved = resolve_pdf_urls_batch(dois)
                    for paper in papers_with_urls:
                        if paper.doi and paper.doi in resolved:
                            paper.pdf_url = resolved[paper.doi]
                except Exception as e:
                    logger.warning(f"  PDF URL resolution failed: {e}")
        
        total_chunks = 0
        
        for paper in papers:
            if not paper.pdf_url:
                continue
            
            try:
                # Download and parse PDF using PDFParser.process_paper()
                # This returns List[TextChunk] with proper metadata
                chunks = self._pdf_parser.process_paper(paper)
                
                if not chunks:
                    continue
                
                # Add question_id to each chunk's extra_metadata for persistence
                for chunk in chunks:
                    if not hasattr(chunk, 'extra_metadata'):
                        chunk.extra_metadata = {}
                    chunk.extra_metadata["question_id"] = question_id
                
                # Add chunks to RAG (handles metadata internally)
                added = self.rag.add_chunks(chunks)
                total_chunks += added
                
                paper.is_processed = True
                
            except Exception as e:
                logger.debug(f"  Failed to process {paper.title[:30]}...: {e}")
        
        return total_chunks
    
    def _extract_findings_for_question(
        self,
        question_text: str,
        question_keywords: List[str],
        question_id: str,
        objective: str,
        papers: List[Paper],
    ) -> List[Dict]:
        """
        Extract findings from RAG specifically answering this question.
        
        Args:
            question_text: The research question
            question_keywords: Keywords for the question
            question_id: Question ID for metadata
            objective: Overall research objective
            papers: Papers that were processed
            
        Returns:
            List of finding dicts with question_id attached
        """
        import json
        
        # Query RAG with question text
        results = self.rag.query(question_text, top_k=settings.rag.top_k)
        
        if not results:
            logger.warning("  RAG query returned no results for question")
            return []
        
        # Format context for LLM
        context_block = ""
        for text, metadata, distance in results:
            authors = metadata.get('authors', 'Unknown authors')
            context_block += (
                f"[Paper: {metadata.get('paper_title', 'Unknown')}] "
                f"[Authors: {authors}] "
                f"[DOI: {metadata.get('doi', 'N/A')}]\n"
                f"{text}\n\n"
            )
        
        # Build extraction prompt
        prompt = FINDING_EXTRACTION_PROMPT.format(
            task_description=question_text,
            task_goal=f"Answer: {question_text}",
            objective=objective,
            context=context_block,
        )
        
        # Call LLM for extraction
        try:
            if hasattr(self.llm, 'complete_for_role'):
                response = self.llm.complete_for_role(
                    prompt=prompt,
                    role="finding_extraction",
                    max_tokens=2000,
                )
                text = response.content if hasattr(response, 'content') else str(response)
            else:
                text = self.llm.complete(prompt)
            
            # Parse JSON response
            text = text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            
            result = json.loads(text)
            findings = result.get("findings", [])
            
            # Add question_id to each finding
            for f in findings:
                f["question_id"] = question_id
                f["finding_type"] = "literature"
            
            return findings
            
        except json.JSONDecodeError as e:
            logger.warning(f"  Failed to parse findings JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"  Finding extraction failed: {e}")
            return []
    
    def store_findings_with_question_id(
        self,
        findings: List[Dict],
        question_id: str,
        run_id: str,
    ) -> None:
        """
        Store findings in ChromaDB with question_id metadata.
        
        This enables multi-run persistence:
        - question_id links findings to specific research questions
        - run_id tracks which run produced the finding
        - On resume, we can check which questions have findings
        
        Args:
            findings: List of finding dicts with claim, source, etc.
            question_id: ID of the research question
            run_id: ID of the current run
        """
        if not self.rag:
            logger.warning("No RAG store configured - skipping persistence")
            return
        
        for finding in findings:
            # Store the finding claim as a chunk with question_id metadata
            metadata = {
                "question_id": question_id,
                "run_id": run_id,
                "claim": finding.get("claim", "")[:500],
                "confidence": str(finding.get("confidence", 0.0)),
                "finding_type": finding.get("finding_type", "literature"),
                "paper_title": finding.get("paper_title", ""),
                "doi": finding.get("doi", ""),
            }
            
            # Store finding claim as searchable text
            self.rag.add_chunk(
                text=finding.get("claim", ""),
                metadata=metadata,
            )
            logger.debug(f"Stored finding for Q:{question_id}: {finding.get('claim', '')[:50]}...")


# =============================================================================
# PERSISTENCE: Check which questions have findings
# =============================================================================

def check_question_coverage(
    rag_system: RAGSystem,
    questions: List,
    min_findings: int = 2,
) -> Dict[str, str]:
    """
    Check which questions have sufficient findings for resume.
    
    This enables multi-run persistence: on resume, we can skip
    questions that already have enough findings and focus on
    unanswered ones.
    
    Args:
        rag_system: RAGSystem instance with ChromaDB collection
        questions: List of ResearchQuestion objects
        min_findings: Minimum findings to consider "answered"
        
    Returns:
        Dict mapping question_id to status ("answered", "partial", "unanswered")
    """
    coverage = {}
    
    for q in questions:
        # Handle both dict and object formats
        question_id = q.get("id") if isinstance(q, dict) else q.id
        
        try:
            # Query ChromaDB collection directly for findings with this question_id
            collection = rag_system._get_collection()
            
            # Count findings for this question
            results = collection.get(
                where={"question_id": question_id},
                limit=min_findings + 1,  # Just need to know if >= min
            )
            
            finding_count = len(results.get("ids", []))
            
            if finding_count >= min_findings:
                coverage[question_id] = "answered"
            elif finding_count > 0:
                coverage[question_id] = "partial"
            else:
                coverage[question_id] = "unanswered"
                
        except Exception as e:
            logger.debug(f"Could not check coverage for Q:{question_id}: {e}")
            coverage[question_id] = "unanswered"
    
    return coverage


def get_unanswered_questions(
    rag_system: RAGSystem,
    questions: List,
    min_findings: int = 2,
) -> List:
    """
    Get list of questions that still need research.
    
    Args:
        rag_system: RAGSystem instance
        questions: List of ResearchQuestion objects
        min_findings: Minimum findings to consider "answered"
        
    Returns:
        List of ResearchQuestion objects that are unanswered or partial
    """
    coverage = check_question_coverage(rag_system, questions, min_findings)
    
    unanswered = []
    for q in questions:
        # Handle both dict and object formats
        question_id = q.get("id") if isinstance(q, dict) else q.id
        status = coverage.get(question_id, "unanswered")
        if status in ("unanswered", "partial"):
            unanswered.append(q)
    
    return unanswered
