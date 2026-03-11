"""
WorldModel - Central knowledge store for Inquiro.

This is the main interface for storing and querying all discoveries.
Think of it as a "shared notebook" that all agents read from and write to.

NOT a neural network - it's a structured database + graph!
"""

import logging
import networkx as nx
from typing import List, Optional, Dict, Any
from datetime import datetime

from .database import Database

logger = logging.getLogger(__name__)
from .models import (
    Finding, Relationship, Hypothesis, Task, ResearchQuestion,
    Source, FindingType, RelationshipType, HypothesisStatus,
    TaskType, TaskStatus, QuestionStatus, TaskPriority, generate_id
)


class WorldModel:
    """
    Central knowledge store for all discoveries.
    
    Combines:
    - SQLite database for persistence
    - NetworkX graph for relationship traversal
    
    Example:
        wm = WorldModel("./data/world_model.db")
        
        # Add a finding
        fid = wm.add_finding(
            claim="Gene X is upregulated",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "analysis.ipynb", "cell": 5},
            cycle=1,
            confidence=0.9
        )
        
        # Get summary for LLM
        summary = wm.get_summary()
    """
    
    def __init__(self, db_path: str = "./data/world_model.db"):
        """
        Initialize the world model.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db = Database(db_path)
        self.graph = nx.DiGraph()  # Directed graph for relationships
        
        # Load existing relationships into graph
        self._load_graph_from_db()
    
    def _load_graph_from_db(self) -> None:
        """Load all findings and relationships into the NetworkX graph."""
        # Add all findings as nodes
        for finding in self.db.get_all_findings():
            self.graph.add_node(finding.id, **finding.model_dump())
        
        # Add all relationships as edges
        for rel in self.db.get_all_relationships():
            self.graph.add_edge(
                rel.from_id,
                rel.to_id,
                relationship_type=rel.relationship_type,
                strength=rel.strength,
                reasoning=rel.reasoning
            )
    
    # =========================================================================
    # ADDING DATA
    # =========================================================================
    
    def add_finding(
        self,
        claim: str,
        finding_type: str,
        source: dict,
        cycle: int,
        confidence: float = 0.5,
        tags: List[str] = None,
        evidence: str = None,
        statistical_support: dict = None,
        figures: List[str] = None
    ) -> str:
        """
        Add a new finding to the world model.
        
        Args:
            claim: The scientific claim (e.g., "Gene X is upregulated 2.5-fold")
            finding_type: One of 'data_analysis', 'literature', 'interpretation'
            source: Dict with 'type' and source details
            cycle: Cycle number (1-indexed)
            confidence: Confidence score 0.0-1.0
            tags: Optional categorization tags
            evidence: Optional supporting evidence text
            statistical_support: Optional dict of stats (p-value, etc.)
            figures: Optional list of figure paths associated with this finding
            
        Returns:
            The generated finding ID
        """
        # Create the Finding object
        # Guard: finding_type must be a valid enum value, never None
        valid_types = {t.value for t in FindingType}
        if finding_type not in valid_types:
            logger.warning(
                f"Invalid finding_type '{finding_type}' — defaulting to 'literature'"
            )
            finding_type = "literature"

        # Guard: confidence must be in [0.0, 1.0]
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (ValueError, TypeError):
            confidence = 0.5

        # Guard: cycle must be >= 1
        cycle = max(1, int(cycle)) if cycle else 1

        # Guard: claim must be non-empty
        if not claim or not claim.strip():
            logger.warning("Empty claim — skipping finding")
            return None

        finding = Finding(
            claim=claim,
            finding_type=FindingType(finding_type),
            source=Source(**source),
            cycle=cycle,
            confidence=confidence,
            tags=tags or [],
            evidence=evidence,
            statistical_support=statistical_support,
            figures=figures or []
        )
        
        # Save to database
        self.db.insert_finding(finding)
        
        # Add to graph
        self.graph.add_node(finding.id, **finding.model_dump())
        
        return finding.id
    
    def add_relationship(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str,
        strength: float = 1.0,
        reasoning: str = None
    ) -> None:
        """
        Create a relationship between two findings.
        
        Args:
            from_id: Source finding ID
            to_id: Target finding ID  
            relationship_type: One of 'supports', 'contradicts', 'extends', 'relates_to'
            strength: Relationship strength 0.0-1.0
            reasoning: Why this relationship exists
            
        Raises:
            ValueError: If either finding doesn't exist
        """
        # Verify both findings exist
        if not self.db.get_finding(from_id):
            raise ValueError(f"Finding {from_id} does not exist")
        if not self.db.get_finding(to_id):
            raise ValueError(f"Finding {to_id} does not exist")
        
        # Create relationship
        relationship = Relationship(
            from_id=from_id,
            to_id=to_id,
            relationship_type=RelationshipType(relationship_type),
            strength=strength,
            reasoning=reasoning
        )
        
        # Save to database
        self.db.insert_relationship(relationship)
        
        # Add to graph
        self.graph.add_edge(
            from_id,
            to_id,
            relationship_type=relationship_type,
            strength=strength,
            reasoning=reasoning
        )
    
    def add_hypothesis(
        self,
        statement: str,
        cycle: int,
        supporting_finding_ids: List[str] = None
    ) -> str:
        """
        Add a new hypothesis.
        
        Args:
            statement: The hypothesis statement
            cycle: Cycle when proposed
            supporting_finding_ids: Optional initial supporting findings
            
        Returns:
            The generated hypothesis ID
        """
        hypothesis = Hypothesis(
            statement=statement,
            cycle_proposed=cycle,
            supporting_finding_ids=supporting_finding_ids or []
        )
        
        self.db.insert_hypothesis(hypothesis)
        return hypothesis.id
    
    def update_hypothesis(
        self,
        hypothesis_id: str,
        status: str,
        cycle_resolved: int = None
    ) -> None:
        """Update hypothesis status."""
        self.db.update_hypothesis_status(hypothesis_id, status, cycle_resolved)
    
    def add_task(self, task: Task) -> str:
        """Add a task to the database."""
        return self.db.insert_task(task)
    
    def update_task(
        self,
        task_id: str,
        status: str,
        error_message: str = None,
        execution_time: float = None,
        result_finding_ids: List[str] = None
    ) -> None:
        """Update task status and results."""
        self.db.update_task_status(
            task_id, status, error_message, execution_time, result_finding_ids
        )
    
    # =========================================================================
    # QUERYING
    # =========================================================================
    
    def get_finding(self, finding_id: str) -> Optional[Finding]:
        """Get a finding by ID."""
        return self.db.get_finding(finding_id)
    
    def get_findings_by_cycle(self, cycle: int) -> List[Finding]:
        """Get all findings from a specific cycle."""
        return self.db.get_findings_by_cycle(cycle)
    
    def get_findings_by_type(self, finding_type: str) -> List[Finding]:
        """Get all findings of a specific type."""
        return self.db.get_findings_by_type(finding_type)
    
    def get_recent_findings(self, limit: int = 20) -> List[Finding]:
        """Get the most recent findings."""
        return self.db.get_recent_findings(limit)
    
    def get_all_findings(self) -> List[Finding]:
        """Get all findings."""
        return self.db.get_all_findings()
    
    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get a hypothesis by ID."""
        return self.db.get_hypothesis(hypothesis_id)
    
    def get_all_hypotheses(self) -> List[Hypothesis]:
        """Get all hypotheses."""
        return self.db.get_all_hypotheses()
    
    def get_active_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses that are still being tested."""
        proposed = self.db.get_hypotheses_by_status("proposed")
        testing = self.db.get_hypotheses_by_status("testing")
        return proposed + testing
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.db.get_task(task_id)
    
    def get_tasks_by_cycle(self, cycle: int) -> List[Task]:
        """Get all tasks from a specific cycle."""
        return self.db.get_tasks_by_cycle(cycle)
    
    # =========================================================================
    # GRAPH QUERIES
    # =========================================================================
    
    def get_supporting_findings(self, finding_id: str) -> List[Finding]:
        """Get all findings that support this finding."""
        supporting_ids = []
        for from_id, to_id, data in self.graph.in_edges(finding_id, data=True):
            if data.get("relationship_type") == "supports":
                supporting_ids.append(from_id)
        
        return [self.db.get_finding(fid) for fid in supporting_ids]
    
    def get_contradicting_findings(self, finding_id: str) -> List[Finding]:
        """Get all findings that contradict this finding."""
        contradicting_ids = []
        for from_id, to_id, data in self.graph.in_edges(finding_id, data=True):
            if data.get("relationship_type") == "contradicts":
                contradicting_ids.append(from_id)
        
        return [self.db.get_finding(fid) for fid in contradicting_ids]
    
    def get_unexplored_findings(self) -> List[Finding]:
        """
        Find findings that have no outgoing relationships.
        
        These are potential areas for further investigation.
        """
        unexplored = []
        for finding in self.db.get_all_findings():
            # Check if this finding has any outgoing edges
            if self.graph.out_degree(finding.id) == 0:
                unexplored.append(finding)
        
        return unexplored
    
    def get_evidence_chain(
        self, 
        finding_id: str, 
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Trace all supporting evidence for a finding.
        
        Recursively follows 'supports' relationships to build
        a complete evidence chain back to primary sources.
        
        Args:
            finding_id: Starting finding
            max_depth: Maximum recursion depth
            
        Returns:
            List of dicts with finding info and depth
        """
        chain = []
        visited = set()
        
        def trace(fid: str, depth: int):
            if depth > max_depth or fid in visited:
                return
            
            visited.add(fid)
            finding = self.db.get_finding(fid)
            
            if finding:
                chain.append({
                    "finding_id": fid,
                    "claim": finding.claim,
                    "source": finding.source.model_dump(),
                    "confidence": finding.confidence,
                    "depth": depth
                })
                
                # Follow supporting relationships
                for supporter in self.get_supporting_findings(fid):
                    trace(supporter.id, depth + 1)
        
        trace(finding_id, 0)
        return chain
    
    def get_top_findings(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get findings ranked by support (most supported first).
        
        Considers:
        - Number of supporting relationships
        - Confidence scores
        """
        findings_with_support = []
        
        for finding in self.db.get_all_findings():
            # Count incoming 'supports' relationships
            support_count = sum(
                1 for _, _, d in self.graph.in_edges(finding.id, data=True)
                if d.get("relationship_type") == "supports"
            )
            
            findings_with_support.append({
                "finding": finding,
                "support_count": support_count,
                "score": support_count + finding.confidence
            })
        
        # Sort by score descending
        findings_with_support.sort(key=lambda x: x["score"], reverse=True)
        
        return findings_with_support[:n]
    
    # =========================================================================
    # SUMMARY GENERATION (for LLM context)
    # =========================================================================
    
    def get_summary(self, max_findings: int = 30) -> str:
        """
        Generate a text summary for LLM context.
        
        This is what gets passed to the orchestrator/agents
        so they understand the current state of knowledge.
        
        Args:
            max_findings: Maximum findings to include
            
        Returns:
            Formatted text summary
        """
        stats = self.db.get_statistics()
        
        lines = [
            "=" * 60,
            "CURRENT KNOWLEDGE STATE",
            "=" * 60,
            "",
            f"Cycle: {stats['current_cycle']}",
            f"Total Findings: {stats['total_findings']}",
            f"Total Relationships: {stats['total_relationships']}",
            "",
        ]
        
        # Recent findings
        recent = self.get_recent_findings(max_findings)
        if recent:
            lines.append("RECENT FINDINGS:")
            lines.append("-" * 40)
            for f in recent:
                source_type = f.source.type
                lines.append(
                    f"[{f.id}] (cycle {f.cycle}, {source_type}, conf={f.confidence:.2f})"
                )
                lines.append(f"    {f.claim}")
                lines.append("")
        
        # Active hypotheses
        hypotheses = self.get_active_hypotheses()
        if hypotheses:
            lines.append("ACTIVE HYPOTHESES:")
            lines.append("-" * 40)
            for h in hypotheses:
                lines.append(f"[{h.id}] ({h.status})")
                lines.append(f"    {h.statement}")
                if h.supporting_finding_ids:
                    lines.append(f"    Supported by: {', '.join(h.supporting_finding_ids)}")
                lines.append("")
        
        # Unexplored areas
        unexplored = self.get_unexplored_findings()
        if unexplored:
            lines.append("UNEXPLORED AREAS (findings with no follow-up):")
            lines.append("-" * 40)
            for f in unexplored[:10]:  # Limit to 10
                lines.append(f"[{f.id}] {f.claim[:80]}...")
            lines.append("")
        
        # Top supported findings
        top = self.get_top_findings(5)
        if top:
            lines.append("TOP SUPPORTED FINDINGS:")
            lines.append("-" * 40)
            for item in top:
                f = item["finding"]
                lines.append(
                    f"[{f.id}] (support={item['support_count']}, conf={f.confidence:.2f})"
                )
                lines.append(f"    {f.claim}")
            lines.append("")
        
        # Research questions status (if any exist)
        question_summary = self.get_question_summary()
        if question_summary:
            lines.append(question_summary)
        
        return "\n".join(lines)
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self) -> dict:
        """Get summary statistics."""
        return self.db.get_statistics()

    def get_finding_count(self) -> int:
        """Return total number of findings."""
        stats = self.db.get_statistics()
        return stats.get('total_findings', 0)

    def get_relationship_count(self) -> int:
        """Return total number of relationships."""
        stats = self.db.get_statistics()
        return stats.get('total_relationships', 0)
    
    # =========================================================================
    # RESEARCH QUESTIONS
    # =========================================================================
    
    def add_question(
        self,
        question_text: str,
        area_id: str = None,
        area_name: str = None,
        keywords: List[str] = None,
        priority: str = "medium",
        cycle: int = 1
    ) -> str:
        """
        Add a research question to track.
        
        Args:
            question_text: The research question
            area_id: Optional ID of parent research area
            area_name: Optional name of parent research area
            keywords: Keywords for matching findings
            priority: Priority level (high/medium/low)
            cycle: Cycle when question was created
            
        Returns:
            The question ID
        """
        question = ResearchQuestion(
            question_text=question_text,
            area_id=area_id,
            area_name=area_name,
            keywords=keywords or [],
            priority=TaskPriority(priority),
            cycle_created=cycle
        )
        return self.db.insert_question(question)
    
    def get_question(self, question_id: str) -> Optional[ResearchQuestion]:
        """Get a research question by ID."""
        return self.db.get_question(question_id)
    
    def get_all_questions(self) -> List[ResearchQuestion]:
        """Get all research questions."""
        return self.db.get_all_questions()
    
    def get_unanswered_questions(self) -> List[ResearchQuestion]:
        """Get questions that need research (unanswered or partial)."""
        return self.db.get_unanswered_questions()
    
    def get_questions_by_area(self, area_id: str) -> List[ResearchQuestion]:
        """Get all questions for a research area."""
        return self.db.get_questions_by_area(area_id)
    
    def update_question(
        self,
        question_id: str,
        status: str = None,
        answer_summary: str = None,
        confidence_score: float = None,
        evidence_count: int = None,
        related_finding_ids: List[str] = None,
        cycle_answered: int = None
    ) -> None:
        """
        Update a research question's status and answer information.
        
        Args:
            question_id: The question to update
            status: New status (unanswered/partial/answered/skipped)
            answer_summary: Summary of how the question was answered
            confidence_score: How well answered (0-1)
            evidence_count: Number of supporting findings
            related_finding_ids: IDs of findings that address this question
            cycle_answered: Cycle when question was answered
        """
        self.db.update_question_status(
            question_id=question_id,
            status=status,
            answer_summary=answer_summary,
            confidence_score=confidence_score,
            evidence_count=evidence_count,
            related_finding_ids=related_finding_ids,
            cycle_answered=cycle_answered
        )
    
    def link_finding_to_question(self, question_id: str, finding_id: str) -> None:
        """Link a finding to a research question."""
        self.db.add_finding_to_question(question_id, finding_id)
    
    def get_question_summary(self) -> str:
        """
        Generate a summary of research questions for LLM context.
        
        Returns:
            Formatted text showing question status
        """
        questions = self.get_all_questions()
        if not questions:
            return ""
        
        lines = [
            "",
            "RESEARCH QUESTIONS STATUS:",
            "-" * 40
        ]
        
        # Group by status
        unanswered = [q for q in questions if q.status == QuestionStatus.UNANSWERED]
        partial = [q for q in questions if q.status == QuestionStatus.PARTIAL]
        answered = [q for q in questions if q.status == QuestionStatus.ANSWERED]
        
        if unanswered:
            lines.append(f"\n❌ UNANSWERED ({len(unanswered)}):")
            for q in unanswered:
                lines.append(f"  [{q.id}] {q.question_text}")
                if q.keywords:
                    lines.append(f"      Keywords: {', '.join(q.keywords[:4])}")
        
        if partial:
            lines.append(f"\n⚠️ PARTIALLY ANSWERED ({len(partial)}):")
            for q in partial:
                lines.append(f"  [{q.id}] {q.question_text}")
                lines.append(f"      Evidence: {q.evidence_count} findings, confidence: {q.confidence_score:.2f}")
        
        if answered:
            lines.append(f"\n✅ ANSWERED ({len(answered)}):")
            for q in answered[:5]:  # Only show first 5 answered
                lines.append(f"  [{q.id}] {q.question_text}")
                if q.answer_summary:
                    lines.append(f"      Summary: {q.answer_summary[:100]}...")
        
        lines.append("")
        return "\n".join(lines)
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def close(self) -> None:
        """Close database connection."""
        self.db.close()