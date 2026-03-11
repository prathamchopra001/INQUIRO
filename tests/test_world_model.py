"""
Unit tests for the Inquiro World Model.

Run with: pytest tests/ -v
"""

import pytest
import os
import tempfile
from pathlib import Path

from src.world_model.models import (
    Finding, Relationship, Hypothesis, Task,
    Source, FindingType, RelationshipType, HypothesisStatus,
    TaskType, TaskStatus, generate_id
)
from src.world_model.database import Database
from src.world_model.world_model import WorldModel


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup after test
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def database(temp_db_path):
    """Create a fresh database instance."""
    db = Database(temp_db_path)
    yield db
    db.close()


@pytest.fixture
def world_model(temp_db_path):
    """Create a fresh world model instance."""
    wm = WorldModel(temp_db_path)
    yield wm
    wm.close()


@pytest.fixture
def sample_finding():
    """Create a sample finding for testing."""
    return Finding(
        claim="Test metabolite X is elevated 2-fold",
        finding_type=FindingType.DATA_ANALYSIS,
        cycle=1,
        confidence=0.85,
        source=Source(type="notebook", path="test.ipynb", cell=5),
        tags=["test", "metabolomics"]
    )


@pytest.fixture
def sample_source_notebook():
    """Create a sample notebook source."""
    return Source(type="notebook", path="analysis.ipynb", cell=10)


@pytest.fixture
def sample_source_paper():
    """Create a sample paper source."""
    return Source(
        type="paper",
        doi="10.1234/test",
        title="Test Paper",
        authors=["Author A", "Author B"],
        year=2024
    )


# =============================================================================
# MODEL TESTS
# =============================================================================

class TestModels:
    """Tests for Pydantic models."""
    
    def test_generate_id_format(self):
        """Test that generate_id creates proper format."""
        fid = generate_id("f")
        assert fid.startswith("f_")
        assert len(fid) == 10  # f_ + 8 hex chars
    
    def test_generate_id_unique(self):
        """Test that generate_id creates unique IDs."""
        ids = [generate_id("f") for _ in range(100)]
        assert len(set(ids)) == 100  # All unique
    
    def test_finding_creation(self, sample_source_notebook):
        """Test creating a Finding."""
        finding = Finding(
            claim="Test claim",
            finding_type=FindingType.DATA_ANALYSIS,
            cycle=1,
            source=sample_source_notebook
        )
        assert finding.id is not None
        assert finding.claim == "Test claim"
        assert finding.confidence == 0.5  # default
        assert finding.tags == []  # default
    
    def test_finding_confidence_bounds(self, sample_source_notebook):
        """Test that confidence must be between 0 and 1."""
        # Valid
        f = Finding(
            claim="Test",
            finding_type=FindingType.DATA_ANALYSIS,
            cycle=1,
            source=sample_source_notebook,
            confidence=0.0
        )
        assert f.confidence == 0.0
        
        f = Finding(
            claim="Test",
            finding_type=FindingType.DATA_ANALYSIS,
            cycle=1,
            source=sample_source_notebook,
            confidence=1.0
        )
        assert f.confidence == 1.0
        
        # Invalid - should raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            Finding(
                claim="Test",
                finding_type=FindingType.DATA_ANALYSIS,
                cycle=1,
                source=sample_source_notebook,
                confidence=1.5
            )
    
    def test_finding_with_paper_source(self, sample_source_paper):
        """Test creating a Finding with paper source."""
        finding = Finding(
            claim="Literature supports X",
            finding_type=FindingType.LITERATURE,
            cycle=2,
            source=sample_source_paper,
            confidence=0.95
        )
        assert finding.source.type == "paper"
        assert finding.source.doi == "10.1234/test"
        assert finding.source.year == 2024
    
    def test_relationship_creation(self):
        """Test creating a Relationship."""
        rel = Relationship(
            from_id="f_001",
            to_id="f_002",
            relationship_type=RelationshipType.SUPPORTS,
            strength=0.9,
            reasoning="Finding 1 supports Finding 2"
        )
        assert rel.from_id == "f_001"
        assert rel.to_id == "f_002"
        assert rel.relationship_type == RelationshipType.SUPPORTS
    
    def test_hypothesis_creation(self):
        """Test creating a Hypothesis."""
        hyp = Hypothesis(
            statement="Treatment X activates pathway Y",
            cycle_proposed=3,
            supporting_finding_ids=["f_001", "f_002"]
        )
        assert hyp.id is not None
        assert hyp.status == HypothesisStatus.PROPOSED
        assert len(hyp.supporting_finding_ids) == 2
    
    def test_task_creation(self):
        """Test creating a Task."""
        task = Task(
            task_type=TaskType.DATA_ANALYSIS,
            description="Analyze differential expression",
            goal="Find significantly changed genes",
            cycle=1
        )
        assert task.id is not None
        assert task.status == TaskStatus.PENDING
        assert task.priority.value == "medium"


# =============================================================================
# DATABASE TESTS
# =============================================================================

class TestDatabase:
    """Tests for database operations."""
    
    def test_database_creation(self, temp_db_path):
        """Test that database creates tables."""
        db = Database(temp_db_path)
        
        # Check tables exist
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        
        assert "findings" in tables
        assert "relationships" in tables
        assert "hypotheses" in tables
        assert "tasks" in tables
        
        db.close()
    
    def test_insert_and_get_finding(self, database, sample_finding):
        """Test inserting and retrieving a finding."""
        # Insert
        finding_id = database.insert_finding(sample_finding)
        assert finding_id == sample_finding.id
        
        # Retrieve
        retrieved = database.get_finding(finding_id)
        assert retrieved is not None
        assert retrieved.claim == sample_finding.claim
        assert retrieved.confidence == sample_finding.confidence
        assert retrieved.cycle == sample_finding.cycle
    
    def test_get_findings_by_cycle(self, database):
        """Test filtering findings by cycle."""
        # Add findings to different cycles
        for cycle in [1, 1, 2, 2, 2, 3]:
            f = Finding(
                claim=f"Finding from cycle {cycle}",
                finding_type=FindingType.DATA_ANALYSIS,
                cycle=cycle,
                source=Source(type="notebook", path="test.ipynb", cell=1)
            )
            database.insert_finding(f)
        
        # Query
        cycle_1 = database.get_findings_by_cycle(1)
        cycle_2 = database.get_findings_by_cycle(2)
        cycle_3 = database.get_findings_by_cycle(3)
        
        assert len(cycle_1) == 2
        assert len(cycle_2) == 3
        assert len(cycle_3) == 1
    
    def test_insert_and_get_relationship(self, database):
        """Test inserting and retrieving relationships."""
        # Create two findings first
        f1 = Finding(
            claim="Finding 1",
            finding_type=FindingType.DATA_ANALYSIS,
            cycle=1,
            source=Source(type="notebook", path="test.ipynb", cell=1)
        )
        f2 = Finding(
            claim="Finding 2",
            finding_type=FindingType.DATA_ANALYSIS,
            cycle=1,
            source=Source(type="notebook", path="test.ipynb", cell=2)
        )
        database.insert_finding(f1)
        database.insert_finding(f2)
        
        # Create relationship
        rel = Relationship(
            from_id=f1.id,
            to_id=f2.id,
            relationship_type=RelationshipType.SUPPORTS,
            strength=0.8
        )
        rel_id = database.insert_relationship(rel)
        assert rel_id is not None
        
        # Query relationships
        from_rels = database.get_relationships_from(f1.id)
        to_rels = database.get_relationships_to(f2.id)
        
        assert len(from_rels) == 1
        assert len(to_rels) == 1
        assert from_rels[0].relationship_type == RelationshipType.SUPPORTS
    
    def test_hypothesis_crud(self, database):
        """Test hypothesis create, read, update."""
        # Create
        hyp = Hypothesis(
            statement="Test hypothesis",
            cycle_proposed=1
        )
        hyp_id = database.insert_hypothesis(hyp)
        
        # Read
        retrieved = database.get_hypothesis(hyp_id)
        assert retrieved.statement == "Test hypothesis"
        assert retrieved.status == HypothesisStatus.PROPOSED
        
        # Update status
        database.update_hypothesis_status(hyp_id, "supported", cycle_resolved=5)
        
        updated = database.get_hypothesis(hyp_id)
        assert updated.status == HypothesisStatus.SUPPORTED
        assert updated.cycle_resolved == 5
    
    def test_task_crud(self, database):
        """Test task create, read, update."""
        # Create
        task = Task(
            task_type=TaskType.DATA_ANALYSIS,
            description="Test task",
            goal="Test goal",
            cycle=1
        )
        task_id = database.insert_task(task)
        
        # Read
        retrieved = database.get_task(task_id)
        assert retrieved.description == "Test task"
        assert retrieved.status == TaskStatus.PENDING
        
        # Update
        database.update_task_status(
            task_id,
            status="completed",
            execution_time=10.5,
            result_finding_ids=["f_001", "f_002"]
        )
        
        updated = database.get_task(task_id)
        assert updated.status == TaskStatus.COMPLETED
        assert updated.execution_time_seconds == 10.5
        assert updated.result_finding_ids == ["f_001", "f_002"]
    
    def test_statistics(self, database):
        """Test statistics generation."""
        # Add some data
        for i in range(5):
            f = Finding(
                claim=f"Finding {i}",
                finding_type=FindingType.DATA_ANALYSIS if i < 3 else FindingType.LITERATURE,
                cycle=1,
                source=Source(type="notebook", path="test.ipynb", cell=i)
            )
            database.insert_finding(f)
        
        stats = database.get_statistics()
        
        assert stats["total_findings"] == 5
        assert stats["findings_by_type"]["data_analysis"] == 3
        assert stats["findings_by_type"]["literature"] == 2
        assert stats["current_cycle"] == 1


# =============================================================================
# WORLD MODEL TESTS
# =============================================================================

class TestWorldModel:
    """Tests for the WorldModel class."""
    
    def test_add_finding(self, world_model):
        """Test adding a finding via WorldModel."""
        fid = world_model.add_finding(
            claim="Test claim",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1,
            confidence=0.75
        )
        
        assert fid is not None
        
        # Verify in database
        finding = world_model.get_finding(fid)
        assert finding.claim == "Test claim"
        assert finding.confidence == 0.75
        
        # Verify in graph
        assert fid in world_model.graph.nodes
    
    def test_add_relationship(self, world_model):
        """Test adding a relationship."""
        # Add two findings
        f1 = world_model.add_finding(
            claim="Finding 1",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1
        )
        f2 = world_model.add_finding(
            claim="Finding 2",
            finding_type="interpretation",
            source={"type": "notebook", "path": "test.ipynb", "cell": 2},
            cycle=1
        )
        
        # Add relationship
        world_model.add_relationship(
            from_id=f1,
            to_id=f2,
            relationship_type="supports",
            strength=0.9
        )
        
        # Verify in graph
        assert world_model.graph.has_edge(f1, f2)
        edge_data = world_model.graph.edges[f1, f2]
        assert edge_data["relationship_type"] == "supports"
        assert edge_data["strength"] == 0.9
    
    def test_relationship_validation(self, world_model):
        """Test that relationships require valid finding IDs."""
        f1 = world_model.add_finding(
            claim="Real finding",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1
        )
        
        # Should raise error for non-existent finding
        with pytest.raises(ValueError):
            world_model.add_relationship(
                from_id=f1,
                to_id="fake_id",
                relationship_type="supports"
            )
    
    def test_get_supporting_findings(self, world_model):
        """Test querying supporting findings."""
        # Create findings
        f1 = world_model.add_finding(
            claim="Evidence 1",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1
        )
        f2 = world_model.add_finding(
            claim="Evidence 2",
            finding_type="literature",
            source={"type": "paper", "doi": "10.1234/test"},
            cycle=1
        )
        f3 = world_model.add_finding(
            claim="Conclusion",
            finding_type="interpretation",
            source={"type": "notebook", "path": "test.ipynb", "cell": 10},
            cycle=1
        )
        
        # Create support relationships
        world_model.add_relationship(f1, f3, "supports")
        world_model.add_relationship(f2, f3, "supports")
        
        # Query
        supporters = world_model.get_supporting_findings(f3)
        supporter_ids = [s.id for s in supporters]
        
        assert len(supporters) == 2
        assert f1 in supporter_ids
        assert f2 in supporter_ids
    
    def test_get_unexplored_findings(self, world_model):
        """Test finding unexplored areas."""
        # Create findings
        f1 = world_model.add_finding(
            claim="Explored finding",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1
        )
        f2 = world_model.add_finding(
            claim="Unexplored finding",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 2},
            cycle=1
        )
        f3 = world_model.add_finding(
            claim="Another unexplored",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 3},
            cycle=1
        )
        
        # f1 has outgoing relationship, f2 and f3 don't
        world_model.add_relationship(f1, f2, "supports")
        
        unexplored = world_model.get_unexplored_findings()
        unexplored_ids = [f.id for f in unexplored]
        
        # f2 and f3 should be unexplored (no outgoing edges)
        assert f2 in unexplored_ids
        assert f3 in unexplored_ids
        # f1 has outgoing edge, so it's "explored"
        assert f1 not in unexplored_ids
    
    def test_get_evidence_chain(self, world_model):
        """Test evidence chain traversal."""
        # Create a chain: f1 -> f2 -> f3
        f1 = world_model.add_finding(
            claim="Primary evidence",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1
        )
        f2 = world_model.add_finding(
            claim="Secondary evidence",
            finding_type="literature",
            source={"type": "paper", "doi": "10.1234/test"},
            cycle=2
        )
        f3 = world_model.add_finding(
            claim="Final conclusion",
            finding_type="interpretation",
            source={"type": "notebook", "path": "test.ipynb", "cell": 10},
            cycle=3
        )
        
        world_model.add_relationship(f1, f2, "supports")
        world_model.add_relationship(f2, f3, "supports")
        
        # Get chain from f3
        chain = world_model.get_evidence_chain(f3)
        
        # Should include f3 (depth 0), f2 (depth 1), f1 (depth 2)
        assert len(chain) == 3
        depths = {item["finding_id"]: item["depth"] for item in chain}
        assert depths[f3] == 0
        assert depths[f2] == 1
        assert depths[f1] == 2
    
    def test_get_summary(self, world_model):
        """Test summary generation."""
        # Add some data
        f1 = world_model.add_finding(
            claim="Test finding 1",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1,
            confidence=0.8
        )
        
        world_model.add_hypothesis(
            statement="Test hypothesis",
            cycle=1,
            supporting_finding_ids=[f1]
        )
        
        summary = world_model.get_summary()
        
        # Check summary contains key sections
        assert "CURRENT KNOWLEDGE STATE" in summary
        assert "RECENT FINDINGS" in summary
        assert "Test finding 1" in summary
        assert "ACTIVE HYPOTHESES" in summary
        assert "Test hypothesis" in summary
    
    def test_hypothesis_workflow(self, world_model):
        """Test complete hypothesis workflow."""
        # Add finding
        f1 = world_model.add_finding(
            claim="Supporting evidence",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1
        )
        
        # Add hypothesis
        h1 = world_model.add_hypothesis(
            statement="Test hypothesis",
            cycle=1,
            supporting_finding_ids=[f1]
        )
        
        # Check initial state
        hyp = world_model.get_hypothesis(h1)
        assert hyp.status == HypothesisStatus.PROPOSED
        
        # Update to testing
        world_model.update_hypothesis(h1, "testing")
        hyp = world_model.get_hypothesis(h1)
        assert hyp.status == HypothesisStatus.TESTING
        
        # Update to supported
        world_model.update_hypothesis(h1, "supported", cycle_resolved=5)
        hyp = world_model.get_hypothesis(h1)
        assert hyp.status == HypothesisStatus.SUPPORTED
        assert hyp.cycle_resolved == 5
    
    def test_get_top_findings(self, world_model):
        """Test ranking findings by support."""
        # Create findings with different support levels
        f1 = world_model.add_finding(
            claim="Highly supported",
            finding_type="interpretation",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1,
            confidence=0.7
        )
        f2 = world_model.add_finding(
            claim="Less supported",
            finding_type="interpretation",
            source={"type": "notebook", "path": "test.ipynb", "cell": 2},
            cycle=1,
            confidence=0.9
        )
        
        # Create supporting findings
        for i in range(3):
            supporter = world_model.add_finding(
                claim=f"Support for f1 - {i}",
                finding_type="data_analysis",
                source={"type": "notebook", "path": "test.ipynb", "cell": 10 + i},
                cycle=1
            )
            world_model.add_relationship(supporter, f1, "supports")
        
        supporter = world_model.add_finding(
            claim="Support for f2",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 20},
            cycle=1
        )
        world_model.add_relationship(supporter, f2, "supports")
        
        # Get top findings
        top = world_model.get_top_findings(2)
        
        # f1 should be first (3 supports) even though f2 has higher confidence
        assert top[0]["finding"].id == f1
        assert top[0]["support_count"] == 3
    
    def test_statistics(self, world_model):
        """Test statistics retrieval."""
        # Add some data
        world_model.add_finding(
            claim="Test",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1
        )
        
        stats = world_model.get_statistics()
        
        assert "total_findings" in stats
        assert stats["total_findings"] == 1
        assert "current_cycle" in stats


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_research_cycle_simulation(self, world_model):
        """Simulate a mini research cycle."""
        # Cycle 1: Initial data analysis
        f1 = world_model.add_finding(
            claim="Gene X expression increased 3-fold in treatment",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "cycle1_analysis.ipynb", "cell": 15},
            cycle=1,
            confidence=0.88,
            tags=["gene_expression", "treatment"],
            statistical_support={"fold_change": 3.0, "p_value": 0.001}
        )
    
        f2 = world_model.add_finding(
            claim="Gene Y shows correlation with Gene X (r=0.85)",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "cycle1_analysis.ipynb", "cell": 22},
            cycle=1,
            confidence=0.82,
            tags=["correlation", "gene_expression"]
        )
    
        # Cycle 1: Literature search
        f3 = world_model.add_finding(
            claim="Gene X is known regulator of pathway Z",
            finding_type="literature",
            source={
                "type": "paper",
                "doi": "10.1234/genex",
                "title": "Gene X regulatory function",
                "authors": ["Smith J", "Jones A"],
                "year": 2023
            },
            cycle=1,
            confidence=0.95
        )
    
        # Cycle 1: Interpretation based on data + literature
        f4 = world_model.add_finding(
            claim="Treatment likely activates pathway Z via Gene X upregulation",
            finding_type="interpretation",
            source={"type": "notebook", "path": "cycle1_interpretation.ipynb", "cell": 5},
            cycle=1,
            confidence=0.75
        )
    
        # Create hypothesis (tracks supporting findings internally)
        h1 = world_model.add_hypothesis(
            statement="Treatment activates pathway Z through Gene X upregulation",
            cycle=1,
            supporting_finding_ids=[f1, f3, f4]
        )
    
        # Create relationships BETWEEN FINDINGS (not to hypotheses)
        world_model.add_relationship(f1, f4, "supports", 0.9,
                                      reasoning="Data shows Gene X upregulation")
        world_model.add_relationship(f3, f4, "supports", 0.85,
                                      reasoning="Literature confirms Gene X regulates pathway Z")
        world_model.add_relationship(f2, f1, "relates_to", 0.7,
                                      reasoning="Gene Y correlates with Gene X")
    
        # Cycle 2: Follow-up analysis
        f5 = world_model.add_finding(
            claim="Pathway Z metabolites elevated in treatment group",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "cycle2_metabolomics.ipynb", "cell": 8},
            cycle=2,
            confidence=0.91,
            tags=["metabolomics", "pathway_z"]
        )
    
        # This new finding supports our interpretation
        world_model.add_relationship(f5, f4, "supports", 0.92,
                                      reasoning="Metabolite data confirms pathway activation")
    
        # Update hypothesis status based on accumulating evidence
        world_model.update_hypothesis(h1, "supported", cycle_resolved=2)
    
        # Verify final state
        stats = world_model.get_statistics()
        assert stats["total_findings"] == 5
        assert stats["total_relationships"] == 4
        assert stats["current_cycle"] == 2
    
        # Verify hypothesis
        hyp = world_model.get_hypothesis(h1)
        assert hyp.status == HypothesisStatus.SUPPORTED
    
        # Verify evidence chain for the interpretation
        chain = world_model.get_evidence_chain(f4)
        assert len(chain) >= 3  # f4 + its supporters
    
        # Verify summary generation
        summary = world_model.get_summary()
        assert "Gene X" in summary