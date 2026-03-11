"""
Research Plan Module for Orchestrator

This module handles:
1. Detecting multi-area objectives
2. Decomposing objectives into structured research plans
3. Tracking coverage across research areas
4. Prioritizing uncovered areas for task generation

INTEGRATION: Import in src/agents/orchestrator.py and call
before the first research cycle.

Usage:
    from src.orchestration.research_plan import ResearchPlanner
    
    planner = ResearchPlanner(llm_client)
    
    # Before cycle 1
    if planner.is_multi_area_objective(objective):
        plan = planner.create_research_plan(objective)
        # plan contains areas, keywords, success criteria
    
    # During task generation
    coverage = planner.check_coverage(plan, findings)
    priority_areas = coverage["priority_areas"]
"""

import json
import logging
import re
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.utils.shared_embeddings import get_shared_embedding_model, is_embedding_available

logger = logging.getLogger(__name__)


# =============================================================================
# SemanticMatcher - Embedding-based similarity for question-finding matching
# =============================================================================

class SemanticMatcher:
    """
    Uses sentence embeddings for semantic similarity matching.
    
    This enables matching questions to findings even when they don't
    share exact keywords but are semantically related.
    
    Example:
        Question: "How can distributed tracing improve monitoring?"
        Finding: "OpenTelemetry enables end-to-end observability"
        Keywords: ["distributed", "tracing"] → NO MATCH
        Semantic: similarity = 0.72 → MATCH!
    
    NOTE: Uses the shared embedding model from src/utils/shared_embeddings.py
    to ensure consistency with RAG and avoid loading multiple model instances.
    """
    
    def __init__(self, similarity_threshold: float = 0.45):
        """
        Args:
            similarity_threshold: Minimum cosine similarity to consider a match.
                                  0.45 is fairly permissive, 0.6 is stricter.
        """
        self.threshold = similarity_threshold
        self._shared_model = get_shared_embedding_model()
        logger.info(f"SemanticMatcher: Initializing with threshold={similarity_threshold}")
        logger.info(f"SemanticMatcher: Using shared embedding model (available={self._shared_model.is_available()})")
    
    def is_available(self) -> bool:
        """Check if semantic matching is available."""
        available = self._shared_model.is_available()
        logger.debug(f"SemanticMatcher.is_available() = {available}")
        return available
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Returns:
            Similarity score between 0 and 1, or -1 if model unavailable.
        """
        return self._shared_model.compute_similarity(text1, text2)
    
    def find_relevant_findings(
        self,
        question_text: str,
        findings: List[dict],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find findings semantically related to a question.
        
        Args:
            question_text: The research question
            findings: List of finding dicts with 'id' and 'claim'
            top_k: Maximum number of matches to return
            
        Returns:
            List of (finding_id, similarity_score) tuples, sorted by score
        """
        if not self._shared_model.is_available():
            logger.debug("SemanticMatcher.find_relevant_findings: Model not available")
            return []
        
        if not findings:
            return []
        
        try:
            # Get all texts to embed
            claims = [f.get("claim", "") for f in findings]
            all_texts = [question_text] + claims
            
            # Embed everything in one batch (efficient)
            embeddings = self._shared_model.encode(all_texts)
            if embeddings is None:
                logger.debug("SemanticMatcher.find_relevant_findings: Encoding failed")
                return []
            
            question_emb = embeddings[0]
            claim_embs = embeddings[1:]
            
            # Compute similarities
            results = []
            for i, (finding, claim_emb) in enumerate(zip(findings, claim_embs)):
                dot_product = np.dot(question_emb, claim_emb)
                norm1 = np.linalg.norm(question_emb)
                norm2 = np.linalg.norm(claim_emb)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = float(dot_product / (norm1 * norm2))
                    if similarity >= self.threshold:
                        results.append((finding.get("id", ""), similarity))
            
            # Sort by similarity (highest first) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(
                f"SemanticMatcher.find_relevant_findings: {len(results)} matches "
                f"(threshold={self.threshold})"
            )
            return results[:top_k]
            
        except Exception as e:
            logger.warning(f"SemanticMatcher: Error in find_relevant_findings: {e}")
            return []


class ResearchPlanner:
    """
    Manages research planning for complex, multi-area objectives.
    
    A multi-area objective is one that covers multiple distinct
    research topics that should each be investigated. Without
    explicit planning, the system tends to explore the easiest
    areas deeply while neglecting harder ones.
    """
    
    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLM client for plan generation
        """
        self.llm_client = llm_client
        self._current_plan: Optional[dict] = None
    
    def is_multi_area_objective(self, objective: str) -> bool:
        """
        Detect if an objective covers multiple distinct research areas.
        
        Heuristics:
        - Numbered sections (1), (2), (3)
        - Semicolons separating topics
        - Keywords like "cover", "areas", "domains"
        - Length > 300 words
        - Multiple distinct fields mentioned
        
        Args:
            objective: The research objective string
            
        Returns:
            True if this appears to be a multi-area objective
        """
        indicators = [
            # Numbered sections
            bool(re.search(r'\(\d+\)', objective)),
            # Multiple semicolons
            objective.count(';') >= 2,
            # Coverage keywords
            any(kw in objective.lower() for kw in ['cover', 'areas:', 'domains', 'aspects']),
            # Very long objective
            len(objective.split()) > 100,
            # Enumeration patterns
            bool(re.search(r'\b(first|second|third|fourth|fifth)\b', objective.lower())),
            # Explicit list markers
            bool(re.search(r'[-•]\s+\w', objective)),
        ]
        
        # Need at least 2 indicators to consider it multi-area
        return sum(indicators) >= 2
    
    def create_research_plan(self, objective: str) -> dict:
        """
        Create a structured research plan from the objective.
        
        Args:
            objective: The research objective string
            
        Returns:
            Research plan dict with areas, keywords, success criteria
        """
        try:
            plan = self._create_plan_via_llm(objective)
            if self._validate_plan(plan):
                self._current_plan = plan
                logger.info(f"Created research plan with {plan.get('total_areas', 0)} areas")
                return plan
        except Exception as e:
            logger.warning(f"LLM plan creation failed: {e}")
        
        # Fallback to heuristic plan
        plan = self._create_plan_heuristic(objective)
        self._current_plan = plan
        return plan
    
    def _create_plan_via_llm(self, objective: str) -> dict:
        """Use LLM to create research plan."""
        from config.prompts.research_plan import RESEARCH_PLAN_PROMPT
        
        prompt = RESEARCH_PLAN_PROMPT.format(objective=objective)
        
        # Use complete_for_role if available (LLMClient), otherwise fall back to complete
        if hasattr(self.llm_client, 'complete_for_role'):
            response = self.llm_client.complete_for_role(
                prompt=prompt,
                role="orchestration",  # Use strong tier for planning
                max_tokens=2000
            )
            text = response.content if hasattr(response, 'content') else str(response)
        else:
            text = self.llm_client.complete(prompt, role="orchestration")
        
        return self._parse_json_response(text)
    
    def _create_plan_heuristic(self, objective: str) -> dict:
        """
        Fallback heuristic plan creation.
        
        Extracts areas from numbered sections or semicolon-separated parts.
        """
        logger.info("Using heuristic research plan creation")
        
        areas = []
        
        # Try extracting numbered sections
        numbered_pattern = r'\((\d+)\)\s*([^(]+?)(?=\(\d+\)|$)'
        numbered_matches = re.findall(numbered_pattern, objective, re.DOTALL)
        
        if numbered_matches:
            for num, content in numbered_matches:
                # Extract area name (first few words or up to first dash/colon)
                name_match = re.match(r'([A-Z][A-Z\s]+)', content.strip())
                if name_match:
                    name = name_match.group(1).strip().title()
                else:
                    name = ' '.join(content.split()[:4]).strip(' -:')
                
                # Extract keywords from content
                words = re.findall(r'\b[a-z]{4,}\b', content.lower())
                keywords = list(dict.fromkeys(words))[:6]  # Dedupe, take first 6
                
                areas.append({
                    "id": f"area_{num}",
                    "name": name[:50],
                    "questions": [f"What are the key findings regarding {name.lower()}?"],
                    "keywords": keywords,
                    "success_criteria": f"At least 2 findings addressing {name.lower()}",
                    "priority": "medium"
                })
        
        # Fallback: split by semicolons
        if not areas:
            parts = [p.strip() for p in objective.split(';') if len(p.strip()) > 20]
            for i, part in enumerate(parts[:6], 1):
                name = ' '.join(part.split()[:5])
                words = re.findall(r'\b[a-z]{4,}\b', part.lower())
                
                areas.append({
                    "id": f"area_{i}",
                    "name": name[:50],
                    "questions": [f"What does the literature say about {name.lower()}?"],
                    "keywords": list(dict.fromkeys(words))[:6],
                    "success_criteria": f"At least 1 finding about {name.lower()}",
                    "priority": "medium"
                })
        
        return {
            "total_areas": len(areas),
            "areas": areas,
            "cross_cutting_themes": []
        }
    
    def check_coverage(self, plan: dict, findings: List[dict]) -> dict:
        """
        Check coverage of research areas against current findings.
        
        Args:
            plan: The research plan from create_research_plan()
            findings: List of finding dicts (must have 'claim' key)
            
        Returns:
            Coverage report dict
        """
        coverage = []
        
        for area in plan.get("areas", []):
            area_id = area["id"]
            area_name = area["name"]
            keywords = [kw.lower() for kw in area.get("keywords", [])]
            
            # Count relevant findings
            relevant = []
            for f in findings:
                claim_lower = f.get("claim", "").lower()
                if any(kw in claim_lower for kw in keywords):
                    relevant.append(f)
            
            # Determine status
            count = len(relevant)
            if count >= 3:
                status = "covered"
            elif count >= 1:
                status = "partial"
            else:
                status = "uncovered"
            
            coverage.append({
                "area_id": area_id,
                "area_name": area_name,
                "status": status,
                "relevant_findings": count,
                "keywords": keywords[:4],
                "questions_answered": [],  # Would need LLM to assess
                "gaps": [] if status == "covered" else [f"Need more findings on {area_name}"]
            })
        
        # Calculate overall coverage
        total = len(coverage)
        covered = sum(1 for c in coverage if c["status"] == "covered")
        partial = sum(1 for c in coverage if c["status"] == "partial")
        
        overall = ((covered * 100) + (partial * 50)) / max(total, 1)
        
        # Identify priority areas
        priority_areas = [c["area_id"] for c in coverage if c["status"] == "uncovered"]
        if not priority_areas:
            priority_areas = [c["area_id"] for c in coverage if c["status"] == "partial"]
        
        return {
            "coverage": coverage,
            "overall_coverage": round(overall, 1),
            "priority_areas": priority_areas,
            "recommendation": self._generate_recommendation(coverage)
        }
    
    def _generate_recommendation(self, coverage: List[dict]) -> str:
        """Generate a recommendation for next cycle focus."""
        uncovered = [c for c in coverage if c["status"] == "uncovered"]
        partial = [c for c in coverage if c["status"] == "partial"]
        
        if uncovered:
            names = [c["area_name"] for c in uncovered[:2]]
            return f"Focus on uncovered areas: {', '.join(names)}"
        elif partial:
            names = [c["area_name"] for c in partial[:2]]
            return f"Deepen coverage of: {', '.join(names)}"
        else:
            return "Good coverage across all areas. Consider synthesis."
    
    def get_priority_keywords(self, plan: dict, coverage: dict) -> List[str]:
        """
        Get keywords for priority areas to guide task generation.
        
        Args:
            plan: The research plan
            coverage: Coverage report from check_coverage()
            
        Returns:
            List of keywords for priority areas
        """
        priority_ids = set(coverage.get("priority_areas", []))
        keywords = []
        
        for area in plan.get("areas", []):
            if area["id"] in priority_ids:
                keywords.extend(area.get("keywords", []))
        
        return list(dict.fromkeys(keywords))  # Dedupe while preserving order
    
    def format_coverage_for_prompt(self, coverage: dict) -> str:
        """
        Format coverage report for inclusion in task generation prompt.
        
        Args:
            coverage: Coverage report from check_coverage()
            
        Returns:
            Formatted string for prompt insertion
        """
        lines = ["RESEARCH AREA COVERAGE STATUS:"]
        
        for c in coverage.get("coverage", []):
            status_emoji = {
                "covered": "✅",
                "partial": "⚠️",
                "uncovered": "❌"
            }.get(c["status"], "?")
            
            lines.append(f"  {status_emoji} {c['area_name']}: {c['status']} ({c['relevant_findings']} findings)")
        
        lines.append(f"\nOverall coverage: {coverage.get('overall_coverage', 0)}%")
        lines.append(f"Recommendation: {coverage.get('recommendation', 'Continue research')}")
        
        if coverage.get("priority_areas"):
            lines.append(f"\nPRIORITY: Focus next tasks on areas marked ❌ or ⚠️")
        
        return "\n".join(lines)
    
    def _validate_plan(self, plan: dict) -> bool:
        """Validate that a plan is usable."""
        if not isinstance(plan, dict):
            return False
        if "areas" not in plan or not isinstance(plan["areas"], list):
            return False
        if len(plan["areas"]) < 1:
            return False
        # Check first area has required fields
        first_area = plan["areas"][0]
        required = ["id", "name", "keywords"]
        if not all(k in first_area for k in required):
            return False
        return True
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        text = response.strip()
        
        # Strip markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try finding JSON object in text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON: {text[:200]}")


# =============================================================================
# Integration helper for orchestrator
# =============================================================================

def format_task_generation_context(
    plan: Optional[dict],
    coverage: Optional[dict],
    gap_analysis: str
) -> str:
    """
    Format all planning context for task generation prompt.
    
    Args:
        plan: Research plan (or None if not multi-area)
        coverage: Coverage report (or None)
        gap_analysis: Existing gap analysis string
        
    Returns:
        Combined context string for prompt
    """
    parts = []
    
    if gap_analysis:
        parts.append(gap_analysis)
    
    if plan and coverage:
        planner = ResearchPlanner(None)  # Just using formatting method
        parts.append(planner.format_coverage_for_prompt(coverage))
        
        # Add priority keywords
        priority_kw = planner.get_priority_keywords(plan, coverage)
        if priority_kw:
            parts.append(f"\nPriority area keywords: {', '.join(priority_kw[:10])}")
    
    return "\n\n".join(parts)


# =============================================================================
# Question Manager - Handles question decomposition and validation
# =============================================================================

class QuestionManager:
    """
    Manages research questions - decomposition, storage, and validation.
    
    This is the core of question-driven research:
    1. Decompose objective into specific questions before cycle 1
    2. Store questions in WorldModel for persistence
    3. Validate which questions are answered after each cycle
    4. Guide task generation to target unanswered questions
    """
    
    def __init__(self, llm_client, world_model=None, use_semantic_matching: bool = True,
                 use_adaptive_decomposition: bool = False):
        """
        Args:
            llm_client: LLM client for question generation/validation
            world_model: WorldModel instance for storing questions
            use_semantic_matching: Whether to use embedding-based matching (default True)
            use_adaptive_decomposition: Whether to use hierarchical pillar-based decomposition
                                         that scales with objective complexity (default False)
        """
        self.llm_client = llm_client
        self.world_model = world_model
        self._questions_generated = False
        self._use_adaptive = use_adaptive_decomposition
        
        # Initialize semantic matcher for better question-finding matching
        self._semantic_matcher = None
        if use_semantic_matching:
            logger.info("QuestionManager: Attempting to initialize SemanticMatcher...")
            self._semantic_matcher = SemanticMatcher(similarity_threshold=0.45)
            available = self._semantic_matcher.is_available()
            logger.info(f"QuestionManager: SemanticMatcher.is_available() = {available}")
            if available:
                logger.info("QuestionManager: Semantic matching ENABLED")
            else:
                logger.warning("QuestionManager: Semantic matching UNAVAILABLE - using keyword fallback")
        else:
            logger.info("QuestionManager: Semantic matching disabled by parameter")
        
        # Store full research plan (populated by decompose_objective)
        self._full_research_plan: Optional[dict] = None
        
        # Adaptive decomposition result (populated if use_adaptive=True)
        self._adaptive_result = None
    
    def decompose_objective(
        self,
        objective: str,
        domain_context: str = "",
        store_in_world_model: bool = True,
        use_adaptive: bool = None,
    ) -> List[dict]:
        """
        Decompose a research objective into a structured research plan.
        
        This creates a comprehensive plan including:
        - 3-5 specific research questions (standard mode)
        - OR 4-24 hierarchical questions organized by pillars (adaptive mode)
        - Keyword taxonomy
        - Methodology candidates
        - Success criteria
        
        Args:
            objective: The research objective
            domain_context: Optional domain information (from DomainAnchorExtractor)
            store_in_world_model: Whether to persist questions
            use_adaptive: Override adaptive mode (default: use instance setting)
            
        Returns:
            List of question dicts with id, text, keywords, priority
        """
        # Determine whether to use adaptive decomposition
        should_use_adaptive = use_adaptive if use_adaptive is not None else self._use_adaptive
        
        if should_use_adaptive:
            return self._decompose_adaptive(objective, store_in_world_model)
        
        try:
            full_plan = self._decompose_via_llm(objective, domain_context)
            
            # Store the full plan for later access
            self._full_research_plan = full_plan
            
            # Extract questions from the plan
            questions = full_plan.get("questions", [])
            
            if store_in_world_model and self.world_model:
                for q in questions:
                    # Store in world model and get the generated ID
                    question_id = self.world_model.add_question(
                        question_text=q["question_text"],
                        area_name=q.get("area_name"),
                        keywords=q.get("keywords", []),
                        priority=q.get("priority", "medium"),
                        cycle=1
                    )
                    # Update the question dict with the generated ID
                    q["id"] = question_id
                logger.info(f"📋 Stored {len(questions)} research questions")
            
            self._questions_generated = True
            
            # Log the full plan
            self._log_research_plan(full_plan)
            
            return questions
            
        except Exception as e:
            logger.warning(f"Question decomposition failed: {e}")
            questions = self._decompose_heuristic(objective)
            
            # Store heuristic questions in world model too
            if store_in_world_model and self.world_model:
                for q in questions:
                    question_id = self.world_model.add_question(
                        question_text=q["question_text"],
                        area_name=q.get("area_name"),
                        keywords=q.get("keywords", []),
                        priority=q.get("priority", "medium"),
                        cycle=1
                    )
                    q["id"] = question_id
                logger.info(f"📋 Stored {len(questions)} heuristic research questions")
            
            self._questions_generated = True
            return questions
    
    def _log_research_plan(self, plan: dict) -> None:
        """Log the research plan for visibility."""
        logger.info("\n" + "=" * 60)
        logger.info("📋 RESEARCH PLAN")
        logger.info("=" * 60)
        
        # Research plan metadata
        rp = plan.get("research_plan", {})
        if rp:
            logger.info(f"Title: {rp.get('title', 'N/A')}")
            logger.info(f"Domain: {rp.get('domain', 'N/A')}")
            logger.info(f"Scope: {rp.get('scope', 'N/A')}")
        
        # Questions
        questions = plan.get("questions", [])
        logger.info(f"\n📊 RESEARCH QUESTIONS ({len(questions)}):")
        for i, q in enumerate(questions, 1):
            priority = q.get('priority', 'medium')
            emoji = '🔴' if priority == 'high' else '🟡' if priority == 'medium' else '🟢'
            logger.info(f"  {emoji} Q{i}: {q.get('question_text', '')[:80]}")
        
        # Keyword taxonomy
        taxonomy = plan.get("keyword_taxonomy", {})
        if taxonomy:
            logger.info(f"\n🔑 KEYWORD TAXONOMY:")
            primary = taxonomy.get("primary_terms", [])
            secondary = taxonomy.get("secondary_terms", [])
            if primary:
                logger.info(f"  Primary: {', '.join(primary[:5])}")
            if secondary:
                logger.info(f"  Secondary: {', '.join(secondary[:5])}")
        
        # Methodology candidates
        methods = plan.get("methodology_candidates", {})
        if methods:
            logger.info(f"\n🔬 METHODOLOGY CANDIDATES:")
            lit_methods = methods.get("literature_methods", [])
            data_methods = methods.get("data_analysis_methods", [])
            if lit_methods:
                logger.info(f"  Literature: {', '.join(m.get('method', '') for m in lit_methods[:3])}")
            if data_methods:
                logger.info(f"  Data Analysis: {', '.join(m.get('method', '') for m in data_methods[:3])}")
        
        # Success criteria
        criteria = plan.get("success_criteria", {})
        if criteria:
            logger.info(f"\n✅ SUCCESS CRITERIA:")
            logger.info(f"  Minimum findings: {criteria.get('minimum_findings', 'N/A')}")
            logger.info(f"  Questions to answer: {criteria.get('questions_to_answer', 'N/A')}")
        
        # Estimated cycles
        cycles = plan.get("estimated_cycles_needed", "N/A")
        logger.info(f"\n⏱️ Estimated cycles needed: {cycles}")
        logger.info("=" * 60 + "\n")
    
    def get_full_research_plan(self) -> Optional[dict]:
        """Get the full research plan including methodology and success criteria."""
        return self._full_research_plan
    
    def get_methodology_candidates(self) -> dict:
        """Get methodology candidates from the research plan."""
        if not self._full_research_plan:
            return {}
        return self._full_research_plan.get("methodology_candidates", {})
    
    def get_success_criteria(self) -> dict:
        """Get success criteria from the research plan."""
        if not self._full_research_plan:
            return {}
        return self._full_research_plan.get("success_criteria", {})
    
    def get_keyword_taxonomy(self) -> dict:
        """Get keyword taxonomy from the research plan."""
        if not self._full_research_plan:
            return {}
        return self._full_research_plan.get("keyword_taxonomy", {})
    
    def _decompose_adaptive(
        self,
        objective: str,
        store_in_world_model: bool = True,
    ) -> List[dict]:
        """
        Use adaptive hierarchical decomposition based on objective complexity.
        
        This method:
        1. Assesses objective complexity (1-5 scale)
        2. Generates research pillars (high-level themes)
        3. Expands each pillar into specific sub-questions
        4. Allocates appropriate depth to each question
        
        Results in 4-24 questions organized by pillars, producing
        appropriately-sized reports (8-50 pages) based on scope.
        
        Args:
            objective: Research objective to decompose
            store_in_world_model: Whether to persist questions
            
        Returns:
            List of question dicts with pillar information
        """
        from src.orchestration.adaptive_decomposition import (
            AdaptiveDecomposer,
            create_decomposition_summary,
        )
        
        logger.info("\n🎯 Using ADAPTIVE decomposition (scales with complexity)")
        
        decomposer = AdaptiveDecomposer(self.llm_client)
        result = decomposer.decompose(objective)
        
        # Store for later access
        self._adaptive_result = result
        
        # Log the decomposition summary
        logger.info(create_decomposition_summary(result))
        
        # Convert to question dicts for compatibility
        questions = []
        for q in result.questions:
            q_dict = {
                "id": q.id,
                "question_text": q.question_text,
                "keywords": q.keywords,
                "area_name": q.pillar_name,  # Map pillar to area for compatibility
                "pillar_id": q.pillar_id,
                "priority": q.priority,
                "question_type": q.question_type,
                "papers_allocated": q.papers_allocated,
                "search_depth": q.search_depth,
            }
            questions.append(q_dict)
        
        # Store in world model if requested
        if store_in_world_model and self.world_model:
            for q in questions:
                # Store in world model and get the generated ID
                question_id = self.world_model.add_question(
                    question_text=q["question_text"],
                    area_name=q.get("area_name"),
                    keywords=q.get("keywords", []),
                    priority=q.get("priority", "medium"),
                    cycle=1
                )
                # Update the question dict with the generated ID
                q["id"] = question_id
            logger.info(f"📋 Stored {len(questions)} research questions ({len(result.pillars)} pillars)")
        
        self._questions_generated = True
        
        # Build a compatible research plan for other components
        self._full_research_plan = {
            "research_plan": {
                "title": f"Adaptive research plan ({result.complexity.complexity_score}/5 complexity)",
                "domain": "Auto-detected",
                "scope": f"{len(result.pillars)} pillars, {len(questions)} questions",
            },
            "questions": questions,
            "pillars": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "role_in_report": p.role_in_report,
                }
                for p in result.pillars
            ],
            "complexity": {
                "score": result.complexity.complexity_score,
                "scope_breadth": result.complexity.scope_breadth,
                "depth_required": result.complexity.depth_required,
                "estimated_pages": result.complexity.estimated_report_pages,
            },
            "estimated_cycles_needed": result.complexity.estimated_cycles,
        }
        
        return questions
    
    def get_adaptive_result(self):
        """Get the full adaptive decomposition result if available."""
        return self._adaptive_result
    
    def get_pillars(self) -> List[dict]:
        """Get research pillars from adaptive decomposition."""
        if not self._adaptive_result:
            return []
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "role_in_report": p.role_in_report,
                "question_count": len(p.questions),
            }
            for p in self._adaptive_result.pillars
        ]

    def _decompose_via_llm(self, objective: str, domain_context: str) -> dict:
        """Use LLM to create a structured research plan."""
        from config.prompts.research_plan import QUESTION_DECOMPOSITION_PROMPT
        
        prompt = QUESTION_DECOMPOSITION_PROMPT.format(
            objective=objective,
            domain_context=domain_context or "No specific domain context provided."
        )
        
        # Use complete_for_role if available
        if hasattr(self.llm_client, 'complete_for_role'):
            response = self.llm_client.complete_for_role(
                prompt=prompt,
                role="orchestration",
                max_tokens=3000  # Increased for full plan
            )
            text = response.content if hasattr(response, 'content') else str(response)
        else:
            text = self.llm_client.complete(prompt)
        
        result = self._parse_json_response(text)
        
        # Ensure result has required structure
        if "questions" not in result:
            result["questions"] = []
        
        return result
    
    def _decompose_heuristic(self, objective: str) -> List[dict]:
        """
        Fallback heuristic question generation.
        
        Creates basic questions from objective structure.
        """
        logger.info("Using heuristic question decomposition")
        questions = []
        
        # Extract key noun phrases as topics
        words = objective.split()
        
        # Generate standard research questions
        base_topic = ' '.join(words[:10]) if len(words) > 10 else objective
        
        templates = [
            ("What are the key findings in the literature regarding {topic}?", "high", "descriptive"),
            ("What methods are commonly used to study {topic}?", "medium", "methodological"),
            ("What are the main factors that influence {topic}?", "high", "causal"),
            ("How do different approaches to {topic} compare?", "medium", "comparative"),
            ("What gaps exist in current research on {topic}?", "low", "descriptive"),
        ]
        
        for i, (template, priority, qtype) in enumerate(templates, 1):
            questions.append({
                "question_text": template.format(topic=base_topic[:50]),
                "keywords": words[:5],
                "area_name": None,
                "priority": priority,
                "question_type": qtype
            })
        
        return questions
    
    def validate_questions(
        self,
        findings: List[dict],
        current_cycle: int
    ) -> dict:
        """
        Validate which questions have been answered by current findings.
        
        UPDATED: Now uses semantic matching as PRIMARY path (fast + free),
        with LLM validation as optional enhancement for reasoning.
        
        Previous bug: LLM validation almost never failed, so semantic
        matching (in _validate_heuristic) was never used.
        
        Args:
            findings: List of finding dicts (must have 'claim' and 'id')
            current_cycle: Current research cycle number
            
        Returns:
            Validation report with status for each question
        """
        if not self.world_model:
            return {"evaluations": [], "overall_progress": {}}
        
        questions = self.world_model.get_all_questions()
        if not questions:
            return {"evaluations": [], "overall_progress": {}}
        
        # FIXED: Always use semantic/heuristic matching as PRIMARY path
        # This is fast (embeddings) and free (no LLM tokens)
        # The semantic matcher finds evidence links that LLM validation misses
        validation = self._validate_heuristic(questions, findings)
        
        # Optional: Could enhance with LLM reasoning in future
        # For now, semantic matching provides accurate evidence linking
        
        # ALWAYS update world model with validation results
        # (This was the bug - heuristic path never updated the world model!)
        for eval_item in validation.get("evaluations", []):
            q_id = eval_item.get("question_id")
            if not q_id:
                continue
            
            try:
                self.world_model.update_question(
                    question_id=q_id,
                    status=eval_item.get("status", "unanswered"),
                    answer_summary=eval_item.get("answer_summary"),
                    confidence_score=eval_item.get("confidence_score", 0.0),
                    evidence_count=eval_item.get("evidence_count", 0),
                    related_finding_ids=eval_item.get("relevant_finding_ids", []),
                    cycle_answered=current_cycle if eval_item.get("status") == "answered" else None
                )
            except Exception as update_error:
                logger.warning(f"Failed to update question {q_id}: {update_error}")
        
        return validation
    
    def _validate_via_llm(self, questions: List, findings: List[dict]) -> dict:
        """Use LLM to validate question answers."""
        from config.prompts.research_plan import QUESTION_VALIDATION_PROMPT
        
        # Format questions for prompt
        questions_json = json.dumps([
            {
                "question_id": q.id,
                "question_text": q.question_text,
                "keywords": q.keywords,
                "current_status": q.status.value if hasattr(q.status, 'value') else q.status
            }
            for q in questions
        ], indent=2)
        
        # Format findings for prompt
        findings_summary = "\n".join([
            f"[{f.get('id', 'unknown')}] {f.get('claim', '')[:200]}"
            for f in findings[:50]  # Limit to 50 findings
        ])
        
        prompt = QUESTION_VALIDATION_PROMPT.format(
            questions_json=questions_json,
            findings_summary=findings_summary or "No findings yet."
        )
        
        if hasattr(self.llm_client, 'complete_for_role'):
            response = self.llm_client.complete_for_role(
                prompt=prompt,
                role="orchestration",
                max_tokens=2000
            )
            text = response.content if hasattr(response, 'content') else str(response)
        else:
            text = self.llm_client.complete(prompt)
        
        return self._parse_json_response(text)
    
    def _validate_heuristic(self, questions: List, findings: List[dict]) -> dict:
        """
        Heuristic validation using semantic similarity + keyword matching fallback.
        
        Tries semantic matching first (embedding-based), falls back to keyword
        matching if embeddings aren't available.
        """
        logger.info(f"_validate_heuristic called with {len(questions)} questions, {len(findings)} findings")
        logger.info(f"  self._semantic_matcher type: {type(self._semantic_matcher)}")
        
        evaluations = []
        answered = partial = unanswered = 0
        
        # Check if semantic matching is available
        sm_present = self._semantic_matcher is not None
        sm_available = self._semantic_matcher.is_available() if sm_present else False
        has_findings = len(findings) > 0
        use_semantic = sm_present and sm_available and has_findings
        
        # Diagnostic logging for semantic matcher status
        logger.info(
            f"Heuristic validation check: "
            f"matcher_present={sm_present}, "
            f"is_available={sm_available}, "
            f"findings_count={len(findings)}, "
            f"USE_SEMANTIC={use_semantic}"
        )
        
        if use_semantic:
            logger.info(f"Using semantic matching for {len(questions)} questions against {len(findings)} findings")
        
        for q in questions:
            relevant = []
            match_method = "keyword"
            
            # Try semantic matching first
            if use_semantic:
                semantic_matches = self._semantic_matcher.find_relevant_findings(
                    question_text=q.question_text,
                    findings=findings,
                    top_k=10
                )
                if semantic_matches:
                    relevant = [fid for fid, score in semantic_matches]
                    match_method = "semantic"
                    logger.debug(
                        f"Q: '{q.question_text[:50]}...' → {len(relevant)} semantic matches "
                        f"(top score: {semantic_matches[0][1]:.2f})"
                    )
            
            # Fall back to keyword matching if no semantic matches
            if not relevant:
                keywords = [kw.lower() for kw in (q.keywords or [])]
                for f in findings:
                    claim = f.get("claim", "").lower()
                    if any(kw in claim for kw in keywords):
                        relevant.append(f.get("id", ""))
                match_method = "keyword"  # Only set to keyword when actually falling back
            
            # Determine status based on match count
            if len(relevant) >= 2:
                status = "answered"
                # Semantic matches get higher base confidence
                base_conf = 0.5 if match_method == "semantic" else 0.4
                confidence = min(0.9, base_conf + 0.1 * len(relevant))
                answered += 1
            elif len(relevant) == 1:
                status = "partial"
                confidence = 0.55 if match_method == "semantic" else 0.5
                partial += 1
            else:
                status = "unanswered"
                confidence = 0.0
                unanswered += 1
            
            evaluations.append({
                "question_id": q.id,
                "question_text": q.question_text,
                "status": status,
                "confidence_score": confidence,
                "relevant_finding_ids": relevant[:5],
                "evidence_count": len(relevant),
                "answer_summary": f"Found {len(relevant)} relevant findings via {match_method} matching" if relevant else None,
                "gaps": [] if status == "answered" else ["Need more findings"]
            })
        
        total = len(questions) or 1
        completion = ((answered * 100) + (partial * 50)) / total
        
        # Log summary
        if use_semantic:
            logger.info(f"Semantic matching: {answered} answered, {partial} partial, {unanswered} unanswered")
        
        return {
            "evaluations": evaluations,
            "overall_progress": {
                "answered": answered,
                "partial": partial,
                "unanswered": unanswered,
                "completion_percentage": round(completion, 1)
            },
            "recommendation": "Focus on unanswered questions" if unanswered else "Deepen partial answers",
            "should_continue": unanswered > 0 or partial > 0
        }
    
    def get_unanswered_questions(self) -> List:
        """Get questions that still need research."""
        if not self.world_model:
            return []
        return self.world_model.get_unanswered_questions()
    
    def format_questions_for_task_generation(self) -> str:
        """
        Format unanswered questions for task generation prompt.
        
        Returns:
            Formatted string for injection into task generation
        """
        if not self.world_model:
            return ""
        
        questions = self.world_model.get_all_questions()
        if not questions:
            return ""
        
        lines = ["RESEARCH QUESTIONS TO ADDRESS:"]
        
        # Group by status (handle both enum and string status)
        def get_status(q):
            return q.status.value if hasattr(q.status, 'value') else q.status
        
        unanswered = [q for q in questions if get_status(q) == "unanswered"]
        partial = [q for q in questions if get_status(q) == "partial"]
        
        if unanswered:
            lines.append("\n❌ UNANSWERED (prioritize these):")
            for q in unanswered:
                priority = q.priority.value if hasattr(q.priority, 'value') else q.priority
                priority_marker = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                lines.append(f"  {priority_marker} [{q.id}] {q.question_text}")
                if q.keywords:
                    lines.append(f"      Search keywords: {', '.join(q.keywords[:4])}")
        
        if partial:
            lines.append("\n⚠️ PARTIALLY ANSWERED (deepen these):")
            for q in partial:
                lines.append(f"  [{q.id}] {q.question_text}")
                lines.append(f"      Current evidence: {q.evidence_count} findings")
        
        lines.append("\nGENERATE TASKS THAT DIRECTLY ADDRESS THESE QUESTIONS.")
        
        return "\n".join(lines)
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        text = response.strip()
        
        # Strip markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try finding JSON object in text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON: {text[:200]}")
    
    def get_completion_status(self) -> dict:
        """
        Get overall research completion status.
        
        Returns:
            Dict with completion metrics and recommendation
        """
        if not self.world_model:
            return {"complete": False, "percentage": 0}
        
        questions = self.world_model.get_all_questions()
        if not questions:
            return {"complete": True, "percentage": 100, "reason": "No questions defined"}
        
        # Handle both enum and string status values
        def get_status(q):
            return q.status.value if hasattr(q.status, 'value') else q.status
        
        answered = sum(1 for q in questions if get_status(q) == "answered")
        partial = sum(1 for q in questions if get_status(q) == "partial")
        total = len(questions)
        
        percentage = ((answered * 100) + (partial * 50)) / total
        
        return {
            "complete": answered == total,
            "percentage": round(percentage, 1),
            "answered": answered,
            "partial": partial,
            "unanswered": total - answered - partial,
            "total": total,
            "should_continue": percentage < 80  # Continue until 80% complete
        }
