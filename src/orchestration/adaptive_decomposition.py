"""
Adaptive Research Decomposition Module

This module enables INQUIRO to scale research depth appropriately based on
the complexity and scope of the research objective.

The system:
1. Assesses objective complexity (1-5 scale)
2. Generates research pillars (high-level themes)
3. Expands each pillar into specific sub-questions
4. Allocates appropriate depth to each question

This produces 4-24 questions depending on objective scope, resulting in
8-50 page reports that match the research ambition.

Usage:
    from src.orchestration.adaptive_decomposition import AdaptiveDecomposer
    
    decomposer = AdaptiveDecomposer(llm_client)
    result = decomposer.decompose(objective)
    
    # result contains:
    # - complexity: ComplexityAssessment
    # - pillars: List[ResearchPillar]
    # - questions: List[ResearchQuestion] with pillar_id
    # - depth_allocations: Dict[question_id, DepthAllocation]
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ComplexityAssessment:
    """Result of complexity analysis."""
    complexity_score: int  # 1-5
    scope_breadth: int
    depth_required: int
    output_expectation: int
    methodological_complexity: int
    reasoning: str
    recommended_pillars: int
    recommended_questions_per_pillar: int
    estimated_report_pages: int
    estimated_cycles: int


@dataclass
class ResearchPillar:
    """A high-level research theme containing multiple questions."""
    id: str
    name: str
    description: str
    scope: str
    role_in_report: str  # introduction, background, methods, findings, discussion, implications
    priority: str  # high, medium, low
    estimated_questions: int
    questions: List['ResearchQuestion'] = field(default_factory=list)


@dataclass 
class ResearchQuestion:
    """A specific, answerable research question."""
    id: str
    question_text: str
    question_type: str  # descriptive, comparative, causal, methodological
    keywords: List[str]
    expected_answer_type: str
    priority: str
    estimated_findings_needed: int
    pillar_id: str
    pillar_name: str
    # Depth allocation (filled in later)
    papers_allocated: int = 5
    search_depth: str = "standard"


@dataclass
class DepthAllocation:
    """How much research depth to allocate to a question."""
    question_id: str
    papers_allocated: int
    cycles_allocated: int
    search_depth: str  # shallow, standard, deep
    rationale: str


@dataclass
class DecompositionResult:
    """Complete result of adaptive decomposition."""
    complexity: ComplexityAssessment
    pillars: List[ResearchPillar]
    questions: List[ResearchQuestion]
    depth_allocations: Dict[str, DepthAllocation]
    total_questions: int
    estimated_pages: int
    created_at: str


# =============================================================================
# ADAPTIVE DECOMPOSER
# =============================================================================

class AdaptiveDecomposer:
    """
    Orchestrates adaptive research decomposition.
    
    This is the main entry point for breaking down a research objective
    into appropriately-sized, hierarchical research questions.
    """
    
    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLM client with complete_for_role() method
        """
        self.llm = llm_client
    
    def decompose(
        self,
        objective: str,
        max_questions: int = 24,
        min_questions: int = 4,
    ) -> DecompositionResult:
        """
        Decompose a research objective into hierarchical questions.
        
        Args:
            objective: The research objective string
            max_questions: Maximum questions to generate
            min_questions: Minimum questions to generate
            
        Returns:
            DecompositionResult with complexity, pillars, questions, allocations
        """
        logger.info("🔬 Starting adaptive research decomposition...")
        
        # Step 1: Assess complexity
        complexity = self._assess_complexity(objective)
        logger.info(f"   Complexity score: {complexity.complexity_score}/5")
        logger.info(f"   Recommended: {complexity.recommended_pillars} pillars × {complexity.recommended_questions_per_pillar} questions")
        
        # Step 2: Generate pillars
        pillars = self._generate_pillars(objective, complexity)
        logger.info(f"   Generated {len(pillars)} research pillars")
        
        # Step 3: Expand pillars into questions
        all_questions = []
        for pillar in pillars:
            questions = self._expand_pillar(objective, pillar, complexity)
            pillar.questions = questions
            all_questions.extend(questions)
            logger.info(f"   Pillar '{pillar.name}': {len(questions)} questions")
        
        # Enforce limits
        if len(all_questions) > max_questions:
            all_questions = self._prioritize_questions(all_questions, max_questions)
            logger.info(f"   Trimmed to {len(all_questions)} questions (max={max_questions})")
        elif len(all_questions) < min_questions:
            logger.warning(f"   Only {len(all_questions)} questions (min={min_questions})")
        
        # Step 4: Allocate depth
        allocations = self._allocate_depth(
            objective, all_questions, 
            complexity.estimated_cycles,
            complexity.estimated_report_pages
        )
        
        # Apply allocations to questions
        for q in all_questions:
            if q.id in allocations:
                q.papers_allocated = allocations[q.id].papers_allocated
                q.search_depth = allocations[q.id].search_depth
        
        logger.info(f"✅ Decomposition complete: {len(all_questions)} questions across {len(pillars)} pillars")
        
        return DecompositionResult(
            complexity=complexity,
            pillars=pillars,
            questions=all_questions,
            depth_allocations=allocations,
            total_questions=len(all_questions),
            estimated_pages=complexity.estimated_report_pages,
            created_at=datetime.now().isoformat(),
        )
    
    # =========================================================================
    # STEP 1: COMPLEXITY ASSESSMENT
    # =========================================================================
    
    def _assess_complexity(self, objective: str) -> ComplexityAssessment:
        """Assess the complexity of the research objective."""
        from config.prompts.adaptive_decomposition import COMPLEXITY_ASSESSMENT_PROMPT
        
        prompt = COMPLEXITY_ASSESSMENT_PROMPT.format(objective=objective)
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="orchestration",
                max_tokens=1000
            )
            text = response.content if hasattr(response, 'content') else str(response)
            data = self._parse_json(text)
            
            return ComplexityAssessment(
                complexity_score=data.get("complexity_score", 3),
                scope_breadth=data.get("dimensions", {}).get("scope_breadth", 3),
                depth_required=data.get("dimensions", {}).get("depth_required", 3),
                output_expectation=data.get("dimensions", {}).get("output_expectation", 3),
                methodological_complexity=data.get("dimensions", {}).get("methodological_complexity", 3),
                reasoning=data.get("reasoning", ""),
                recommended_pillars=data.get("recommended_pillars", 4),
                recommended_questions_per_pillar=data.get("recommended_questions_per_pillar", 3),
                estimated_report_pages=data.get("estimated_report_pages", 25),
                estimated_cycles=data.get("estimated_cycles", 8),
            )
        except Exception as e:
            logger.warning(f"Complexity assessment failed: {e}, using defaults")
            return self._default_complexity()
    
    def _default_complexity(self) -> ComplexityAssessment:
        """Default complexity for fallback."""
        return ComplexityAssessment(
            complexity_score=3,
            scope_breadth=3,
            depth_required=3,
            output_expectation=3,
            methodological_complexity=3,
            reasoning="Default assessment (LLM failed)",
            recommended_pillars=4,
            recommended_questions_per_pillar=3,
            estimated_report_pages=25,
            estimated_cycles=8,
        )

    
    # =========================================================================
    # STEP 2: PILLAR GENERATION
    # =========================================================================
    
    def _generate_pillars(
        self, 
        objective: str, 
        complexity: ComplexityAssessment
    ) -> List[ResearchPillar]:
        """Generate research pillars based on complexity."""
        from config.prompts.adaptive_decomposition import PILLAR_GENERATION_PROMPT
        
        prompt = PILLAR_GENERATION_PROMPT.format(
            objective=objective,
            complexity_score=complexity.complexity_score,
            num_pillars=complexity.recommended_pillars,
            scope_breadth=complexity.scope_breadth,
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="orchestration",
                max_tokens=2000
            )
            text = response.content if hasattr(response, 'content') else str(response)
            data = self._parse_json(text)
            
            pillars = []
            for p in data.get("pillars", []):
                pillars.append(ResearchPillar(
                    id=p.get("id", f"pillar_{len(pillars)+1}"),
                    name=p.get("name", "Unnamed Pillar"),
                    description=p.get("description", ""),
                    scope=p.get("scope", ""),
                    role_in_report=p.get("role_in_report", "findings"),
                    priority=p.get("priority", "medium"),
                    estimated_questions=p.get("estimated_questions", 3),
                ))
            
            return pillars if pillars else self._default_pillars(objective)
            
        except Exception as e:
            logger.warning(f"Pillar generation failed: {e}, using defaults")
            return self._default_pillars(objective)
    
    def _default_pillars(self, objective: str) -> List[ResearchPillar]:
        """Default pillars for fallback."""
        return [
            ResearchPillar(
                id="pillar_1",
                name="Background & Context",
                description="Foundational concepts and existing knowledge",
                scope="Historical context, key definitions, existing frameworks",
                role_in_report="background",
                priority="medium",
                estimated_questions=2,
            ),
            ResearchPillar(
                id="pillar_2",
                name="Core Findings",
                description="Main research findings addressing the objective",
                scope="Primary evidence, key studies, empirical results",
                role_in_report="findings",
                priority="high",
                estimated_questions=3,
            ),
            ResearchPillar(
                id="pillar_3",
                name="Analysis & Synthesis",
                description="Integration of findings and critical analysis",
                scope="Patterns, comparisons, synthesis across sources",
                role_in_report="discussion",
                priority="high",
                estimated_questions=2,
            ),
            ResearchPillar(
                id="pillar_4",
                name="Implications & Future",
                description="Practical implications and future directions",
                scope="Applications, limitations, research gaps",
                role_in_report="implications",
                priority="medium",
                estimated_questions=2,
            ),
        ]
    
    # =========================================================================
    # STEP 3: QUESTION EXPANSION
    # =========================================================================
    
    def _expand_pillar(
        self,
        objective: str,
        pillar: ResearchPillar,
        complexity: ComplexityAssessment,
    ) -> List[ResearchQuestion]:
        """Expand a pillar into specific research questions."""
        from config.prompts.adaptive_decomposition import QUESTION_EXPANSION_PROMPT
        
        num_questions = min(
            pillar.estimated_questions,
            complexity.recommended_questions_per_pillar
        )
        
        prompt = QUESTION_EXPANSION_PROMPT.format(
            objective=objective,
            pillar_id=pillar.id,
            pillar_name=pillar.name,
            pillar_description=pillar.description,
            pillar_scope=pillar.scope,
            pillar_role=pillar.role_in_report,
            num_questions=num_questions,
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="orchestration",
                max_tokens=2000
            )
            text = response.content if hasattr(response, 'content') else str(response)
            data = self._parse_json(text)
            
            questions = []
            for q in data.get("questions", []):
                questions.append(ResearchQuestion(
                    id=q.get("id", f"q_{len(questions)+1:03d}"),
                    question_text=q.get("question_text", ""),
                    question_type=q.get("question_type", "descriptive"),
                    keywords=q.get("keywords", []),
                    expected_answer_type=q.get("expected_answer_type", "Evidence summary"),
                    priority=q.get("priority", "medium"),
                    estimated_findings_needed=q.get("estimated_findings_needed", 3),
                    pillar_id=pillar.id,
                    pillar_name=pillar.name,
                ))
            
            return questions if questions else self._default_questions_for_pillar(pillar)
            
        except Exception as e:
            logger.warning(f"Question expansion failed for {pillar.name}: {e}")
            return self._default_questions_for_pillar(pillar)
    
    def _default_questions_for_pillar(self, pillar: ResearchPillar) -> List[ResearchQuestion]:
        """Default questions for a pillar."""
        return [
            ResearchQuestion(
                id=f"{pillar.id}_q1",
                question_text=f"What are the key findings regarding {pillar.name.lower()}?",
                question_type="descriptive",
                keywords=[pillar.name.lower().replace(" ", "_")],
                expected_answer_type="Evidence summary",
                priority=pillar.priority,
                estimated_findings_needed=3,
                pillar_id=pillar.id,
                pillar_name=pillar.name,
            ),
            ResearchQuestion(
                id=f"{pillar.id}_q2",
                question_text=f"What are the main debates or open questions in {pillar.name.lower()}?",
                question_type="comparative",
                keywords=[pillar.name.lower().replace(" ", "_"), "debate", "controversy"],
                expected_answer_type="Comparison table",
                priority="medium",
                estimated_findings_needed=2,
                pillar_id=pillar.id,
                pillar_name=pillar.name,
            ),
        ]

    
    # =========================================================================
    # STEP 4: DEPTH ALLOCATION
    # =========================================================================
    
    def _allocate_depth(
        self,
        objective: str,
        questions: List[ResearchQuestion],
        max_cycles: int,
        target_pages: int,
    ) -> Dict[str, DepthAllocation]:
        """Allocate research depth across questions."""
        from config.prompts.adaptive_decomposition import DEPTH_ALLOCATION_PROMPT
        
        # Estimate resources
        max_papers = max_cycles * 10  # ~10 papers per cycle
        
        questions_json = json.dumps([
            {
                "id": q.id,
                "question_text": q.question_text,
                "pillar": q.pillar_name,
                "priority": q.priority,
                "estimated_findings_needed": q.estimated_findings_needed,
            }
            for q in questions
        ], indent=2)
        
        prompt = DEPTH_ALLOCATION_PROMPT.format(
            objective=objective,
            questions_json=questions_json,
            max_papers=max_papers,
            max_cycles=max_cycles,
            target_pages=target_pages,
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="orchestration",
                max_tokens=2000
            )
            text = response.content if hasattr(response, 'content') else str(response)
            data = self._parse_json(text)
            
            allocations = {}
            for alloc in data.get("allocations", []):
                q_id = alloc.get("question_id")
                if q_id:
                    allocations[q_id] = DepthAllocation(
                        question_id=q_id,
                        papers_allocated=alloc.get("papers_allocated", 5),
                        cycles_allocated=alloc.get("cycles_allocated", 1),
                        search_depth=alloc.get("search_depth", "standard"),
                        rationale=alloc.get("rationale", ""),
                    )
            
            return allocations
            
        except Exception as e:
            logger.warning(f"Depth allocation failed: {e}, using defaults")
            return self._default_depth_allocation(questions)
    
    def _default_depth_allocation(
        self, 
        questions: List[ResearchQuestion]
    ) -> Dict[str, DepthAllocation]:
        """Default depth allocation based on priority."""
        allocations = {}
        for q in questions:
            papers = {"high": 8, "medium": 5, "low": 3}.get(q.priority, 5)
            depth = {"high": "deep", "medium": "standard", "low": "shallow"}.get(q.priority, "standard")
            
            allocations[q.id] = DepthAllocation(
                question_id=q.id,
                papers_allocated=papers,
                cycles_allocated=1,
                search_depth=depth,
                rationale=f"Default allocation for {q.priority} priority question",
            )
        return allocations
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _prioritize_questions(
        self, 
        questions: List[ResearchQuestion],
        max_count: int,
    ) -> List[ResearchQuestion]:
        """Prioritize and trim questions to max count."""
        # Sort by priority (high > medium > low) and pillar diversity
        priority_order = {"high": 0, "medium": 1, "low": 2}
        
        # First, ensure at least one question per pillar
        pillars_seen = set()
        prioritized = []
        remaining = []
        
        for q in sorted(questions, key=lambda x: priority_order.get(x.priority, 1)):
            if q.pillar_id not in pillars_seen and len(prioritized) < max_count:
                prioritized.append(q)
                pillars_seen.add(q.pillar_id)
            else:
                remaining.append(q)
        
        # Then fill with remaining high-priority questions
        for q in remaining:
            if len(prioritized) >= max_count:
                break
            prioritized.append(q)
        
        return prioritized
    
    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response, handling common issues."""
        # Remove markdown code blocks
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
        
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
        
        raise ValueError("No valid JSON found in response")


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_decomposition_summary(result: DecompositionResult) -> str:
    """Create a human-readable summary of the decomposition."""
    lines = [
        "=" * 60,
        "📊 ADAPTIVE RESEARCH DECOMPOSITION",
        "=" * 60,
        "",
        f"Complexity Score: {result.complexity.complexity_score}/5",
        f"  - Scope breadth: {result.complexity.scope_breadth}",
        f"  - Depth required: {result.complexity.depth_required}",
        f"  - Output expectation: {result.complexity.output_expectation}",
        "",
        f"📚 Research Structure:",
        f"  - {len(result.pillars)} Pillars",
        f"  - {result.total_questions} Questions total",
        f"  - ~{result.estimated_pages} pages expected",
        "",
        "📑 Pillars:",
    ]
    
    for pillar in result.pillars:
        lines.append(f"  {pillar.id}: {pillar.name}")
        lines.append(f"      └─ {len(pillar.questions)} questions, role: {pillar.role_in_report}")
    
    lines.extend([
        "",
        "❓ Questions by Priority:",
    ])
    
    high = [q for q in result.questions if q.priority == "high"]
    medium = [q for q in result.questions if q.priority == "medium"]
    low = [q for q in result.questions if q.priority == "low"]
    
    lines.append(f"  🔴 High: {len(high)}")
    for q in high[:3]:
        lines.append(f"      - {q.question_text[:50]}...")
    
    lines.append(f"  🟡 Medium: {len(medium)}")
    lines.append(f"  🟢 Low: {len(low)}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def questions_to_research_questions(
    decomposition: DecompositionResult
) -> List[Dict[str, Any]]:
    """
    Convert DecompositionResult questions to the format expected by
    QuestionDrivenResearch and ResearchPlanner.
    
    This bridges the adaptive decomposition output to existing INQUIRO components.
    """
    return [
        {
            "id": q.id,
            "question_text": q.question_text,
            "keywords": q.keywords,
            "area_name": q.pillar_name,
            "pillar_id": q.pillar_id,
            "priority": q.priority,
            "question_type": q.question_type,
            "papers_allocated": q.papers_allocated,
            "search_depth": q.search_depth,
        }
        for q in decomposition.questions
    ]
