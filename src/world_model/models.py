"""
Pydantic models for Inquiro world model.

These define the core data structures:
- Finding: A scientific claim with evidence
- Relationship: Connection between findings
- Hypothesis: A testable scientific statement
- Task: A unit of work for agents
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


# ============================================================================
# ENUMS - The "allowed values" for certain fields
# ============================================================================

class FindingType(str, Enum):
    """Types of findings we can discover."""
    DATA_ANALYSIS = "data_analysis"
    LITERATURE = "literature"
    INTERPRETATION = "interpretation"


class RelationshipType(str, Enum):
    """How two findings can relate to each other."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    RELATES_TO = "relates_to"


class HypothesisStatus(str, Enum):
    """Lifecycle stages of a hypothesis."""
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"


class TaskType(str, Enum):
    """Types of tasks agents can perform."""
    DATA_ANALYSIS = "data_analysis"
    LITERATURE_SEARCH = "literature_search"


class TaskStatus(str, Enum):
    """Lifecycle stages of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionStatus(str, Enum):
    """Status of a research question."""
    UNANSWERED = "unanswered"    # No relevant findings yet
    PARTIAL = "partial"          # Some findings, but incomplete
    ANSWERED = "answered"        # Sufficiently addressed
    SKIPPED = "skipped"          # Intentionally not pursued


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def generate_id(prefix: str) -> str:
    """Generate a unique ID with a prefix (e.g., 'f_abc123')."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ============================================================================
# CORE MODELS
# ============================================================================

class Source(BaseModel):
    """
    Origin of a finding - either from code execution or a paper.
    
    Examples:
        Notebook source: Source(type="notebook", path="analysis.ipynb", cell=15)
        Paper source: Source(type="paper", doi="10.1234/example", title="...")
    """
    type: str = Field(..., description="'notebook' or 'paper'")
    
    # For notebook sources
    path: Optional[str] = Field(None, description="Path to notebook file")
    cell: Optional[int] = Field(None, description="Cell number in notebook")
    
    # For paper sources
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    title: Optional[str] = Field(None, description="Paper title")
    authors: Optional[List[str]] = Field(None, description="List of authors")
    url: Optional[str] = Field(None, description="URL to paper")
    year: Optional[int] = Field(None, description="Publication year")


class Finding(BaseModel):
    """
    A single scientific claim with evidence.
    
    This is the atomic unit of knowledge in our world model.
    Every discovery gets stored as a Finding.
    """
    id: Optional[str] = Field(default_factory=lambda: generate_id("f"))
    claim: str = Field(..., description="The scientific claim text")
    finding_type: FindingType = Field(..., description="Type of finding")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence 0-1")
    cycle: int = Field(..., ge=1, description="Cycle number when discovered")
    source: Source = Field(..., description="Where this finding came from")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    evidence: Optional[str] = Field(None, description="Supporting evidence text")
    statistical_support: Optional[Dict[str, Any]] = Field(
        None, description="Statistical values (p-value, etc.)"
    )
    figures: List[str] = Field(default_factory=list, description="Paths to generated figures")
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {"use_enum_values": True}


class Relationship(BaseModel):
    """
    Connection between two findings.
    
    This is how we build the knowledge graph - by linking
    findings that support, contradict, or extend each other.
    """
    id: Optional[int] = Field(None, description="Auto-increment ID from DB")
    from_id: str = Field(..., description="Source finding ID")
    to_id: str = Field(..., description="Target finding ID")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    strength: float = Field(1.0, ge=0.0, le=1.0, description="Relationship strength")
    reasoning: Optional[str] = Field(None, description="Why this relationship exists")
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {"use_enum_values": True}


class Hypothesis(BaseModel):
    """
    A scientific hypothesis to be tested.
    
    Hypotheses are proposed based on findings and then
    either supported or refuted by further evidence.
    """
    id: Optional[str] = Field(default_factory=lambda: generate_id("h"))
    statement: str = Field(..., description="The hypothesis statement")
    status: HypothesisStatus = Field(HypothesisStatus.PROPOSED)
    cycle_proposed: int = Field(..., ge=1, description="When proposed")
    cycle_resolved: Optional[int] = Field(None, description="When resolved")
    supporting_finding_ids: List[str] = Field(default_factory=list)
    contradicting_finding_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {"use_enum_values": True}


class Task(BaseModel):
    """
    A unit of work for an agent.
    
    The orchestrator creates tasks, and agents execute them.
    """
    id: Optional[str] = Field(default_factory=lambda: generate_id("t"))
    task_type: TaskType = Field(..., description="Type of task")
    description: str = Field(..., description="What to do")
    goal: str = Field(..., description="Expected outcome")
    priority: TaskPriority = Field(TaskPriority.MEDIUM)
    status: TaskStatus = Field(TaskStatus.PENDING)
    cycle: int = Field(..., ge=1, description="Cycle number")
    result_finding_ids: List[str] = Field(default_factory=list)
    error_message: Optional[str] = Field(None)
    execution_time_seconds: Optional[float] = Field(None)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(None)
    
    model_config = {"use_enum_values": True}


class ResearchQuestion(BaseModel):
    """
    A specific research question to be answered.
    
    Research questions are decomposed from the objective and tracked
    throughout the research process. Each question has a status
    indicating whether it has been sufficiently answered.
    
    This is the foundation of question-driven research:
    - Tasks are generated to answer specific questions
    - Findings are linked to the questions they address
    - Research continues until all questions are answered
    """
    id: Optional[str] = Field(default_factory=lambda: generate_id("q"))
    question_text: str = Field(..., description="The research question")
    status: QuestionStatus = Field(QuestionStatus.UNANSWERED)
    priority: TaskPriority = Field(TaskPriority.MEDIUM)
    
    # Linking to research areas and findings
    area_id: Optional[str] = Field(None, description="ID of parent research area")
    area_name: Optional[str] = Field(None, description="Name of parent research area")
    keywords: List[str] = Field(default_factory=list, description="Keywords for matching")
    related_finding_ids: List[str] = Field(default_factory=list, description="Findings addressing this question")
    
    # Answer tracking
    answer_summary: Optional[str] = Field(None, description="Summary of how question was answered")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="How well answered (0-1)")
    evidence_count: int = Field(0, ge=0, description="Number of supporting findings")
    
    # Metadata
    cycle_created: int = Field(1, ge=1, description="When question was created")
    cycle_answered: Optional[int] = Field(None, description="When question was answered")
    created_at: datetime = Field(default_factory=datetime.now)
    answered_at: Optional[datetime] = Field(None)
    
    model_config = {"use_enum_values": True}
    
    def is_answered(self) -> bool:
        """Check if question is sufficiently answered."""
        return self.status == QuestionStatus.ANSWERED
    
    def needs_attention(self) -> bool:
        """Check if question needs more research."""
        return self.status in (QuestionStatus.UNANSWERED, QuestionStatus.PARTIAL)