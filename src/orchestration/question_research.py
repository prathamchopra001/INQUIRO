"""
Question-Driven Research Mode for INQUIRO.

This module integrates the QuestionDeepSearcher with the main INQUIRO workflow,
enabling a research mode where each question gets deep, focused attention.

Key differences from standard mode:
- Standard: Generate tasks → Execute tasks → Match findings to questions
- Question-Driven: For each question → Deep search → Extract findings → Store with question_id

Usage:
    from src.orchestration.question_research import QuestionDrivenResearch
    
    qdr = QuestionDrivenResearch(llm_client, rag_system)
    results = qdr.run(questions, objective, max_papers_per_question=10)
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from src.literature.question_deep_search import (
    QuestionDeepSearcher,
    HybridSearcher,
    CitationChainFollower,
    check_question_coverage,
    get_unanswered_questions,
)
from src.literature.pdf_parser import PDFParser
from src.literature.rag import RAGSystem
from src.utils.llm_client import LLMClient
from config.settings import settings

logger = logging.getLogger(__name__)


class QuestionDrivenResearch:
    """
    Orchestrates question-driven deep research.
    
    Instead of the standard INQUIRO workflow (generate tasks → execute → match),
    this directly iterates through research questions and performs deep search
    for each one.
    
    Benefits:
    1. Domain filtering: Question text naturally constrains results
    2. Citation chains: Follows references from top papers
    3. Hybrid search: BM25 + dense embeddings
    4. Persistence: Stores with question_id for resume
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        rag_system: RAGSystem = None,
        pdf_parser: PDFParser = None,
        run_id: str = None,
    ):
        self.llm = llm_client
        self.rag = rag_system or RAGSystem()
        self.pdf_parser = pdf_parser or PDFParser()
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize deep searcher
        self._deep_searcher = QuestionDeepSearcher(
            llm_client=llm_client,
            rag_store=rag_system,
        )
    
    def run(
        self,
        questions: List,
        objective: str,
        max_papers_per_question: int = 10,
        follow_citations: bool = True,
        min_findings_per_question: int = 2,
    ) -> Dict:
        """
        Run question-driven deep research for all questions.
        
        Args:
            questions: List of ResearchQuestion objects
            objective: Overall research objective
            max_papers_per_question: Papers to process per question
            follow_citations: Whether to follow citation chains
            min_findings_per_question: Minimum findings to mark as answered
            
        Returns:
            Dict with results for each question and overall stats
        """
        logger.info(f"🔬 Starting Question-Driven Research for {len(questions)} questions")
        logger.info(f"   Objective: {objective[:60]}...")
        
        results = {
            "questions": {},
            "total_papers_found": 0,
            "total_papers_processed": 0,
            "total_findings": 0,
            "questions_answered": 0,
            "questions_partial": 0,
            "run_id": self.run_id,
        }
        
        # Check which questions already have findings (resume support)
        if self.rag:
            existing_coverage = check_question_coverage(
                self.rag, questions, min_findings_per_question
            )
        else:
            existing_coverage = {}
        
        for i, question in enumerate(questions, 1):
            # Handle both dict and object formats
            if isinstance(question, dict):
                q_id = question.get("id")
                q_text = question.get("question_text", "")
            else:
                q_id = question.id
                q_text = question.question_text
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📝 Question {i}/{len(questions)}: {q_text[:50]}...")
            logger.info(f"   ID: {q_id}")
            
            # Check if already answered (resume support)
            if existing_coverage.get(q_id) == "answered":
                logger.info(f"   ✅ Already answered - skipping")
                results["questions"][q_id] = {
                    "status": "skipped",
                    "reason": "already_answered",
                }
                results["questions_answered"] += 1
                continue
            
            # Deep search for this question using QuestionDeepSearcher
            try:
                search_result = self._deep_searcher.search_for_question(
                    question=question,
                    max_papers=max_papers_per_question,
                    follow_citations=follow_citations,
                    objective=objective,
                )
                
                # Aggregate stats
                results["total_papers_found"] += search_result.get("papers_found", 0)
                results["total_papers_processed"] += search_result.get("papers_processed", 0)
                
                findings = search_result.get("findings", [])
                results["total_findings"] += len(findings)
                
                # Update question status
                if len(findings) >= min_findings_per_question:
                    results["questions_answered"] += 1
                    status = "answered"
                elif len(findings) > 0:
                    results["questions_partial"] += 1
                    status = "partial"
                else:
                    status = "unanswered"
                
                results["questions"][q_id] = {
                    "status": status,
                    "papers_found": search_result.get("papers_found", 0),
                    "papers_processed": search_result.get("papers_processed", 0),
                    "chunks_added": search_result.get("chunks_added", 0),
                    "findings_count": len(findings),
                    "findings": findings,
                    "queries_used": search_result.get("queries_used", []),
                    "citation_papers": search_result.get("citation_papers_found", 0),
                }
                
                # Store findings with question_id for persistence
                if findings and self.rag:
                    self._deep_searcher.store_findings_with_question_id(
                        findings=findings,
                        question_id=q_id,
                        run_id=self.run_id,
                    )
                
                logger.info(f"   📊 Papers: {search_result.get('papers_found', 0)} found, {search_result.get('papers_processed', 0)} processed")
                logger.info(f"   📚 Findings: {len(findings)} extracted → Status: {status}")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing question: {e}")
                import traceback
                traceback.print_exc()
                results["questions"][q_id] = {
                    "status": "error",
                    "error": str(e),
                }
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Question-Driven Research Complete")
        logger.info(f"   Papers found: {results['total_papers_found']}")
        logger.info(f"   Papers processed: {results['total_papers_processed']}")
        logger.info(f"   Findings extracted: {results['total_findings']}")
        logger.info(f"   Questions answered: {results['questions_answered']}/{len(questions)}")
        
        return results
    
    def _check_existing_findings(self, question_id: str) -> int:
        """Check if we already have findings for this question (DEPRECATED - use check_question_coverage)."""
        return 0


class ResearchModeSelector:
    """
    Selects the best research mode based on the research plan.
    
    Modes:
    1. QUESTION_DRIVEN: Deep search per question (best for focused research)
    2. STANDARD: Generate tasks → execute → match (best for exploration)
    3. HYBRID: Start with question-driven, then explore gaps
    """
    
    @staticmethod
    def select_mode(
        research_plan: Dict,
        questions: List,
    ) -> str:
        """
        Select optimal research mode.
        
        Args:
            research_plan: Full research plan from question decomposition
            questions: List of research questions
            
        Returns:
            One of: "question_driven", "standard", "hybrid"
        """
        # Heuristics for mode selection
        num_questions = len(questions)
        
        # If we have a clear set of questions (3-7), question-driven is best
        if 3 <= num_questions <= 7:
            return "question_driven"
        
        # If very few questions, standard exploration might find more
        if num_questions < 3:
            return "standard"
        
        # If many questions, hybrid approach
        if num_questions > 7:
            return "hybrid"
        
        return "question_driven"
