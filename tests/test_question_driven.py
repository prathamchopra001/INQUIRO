"""
Tests for Question-Driven Research System (Piece 1 & 2).

Tests:
1. ResearchQuestion model
2. Database CRUD for questions
3. WorldModel question methods
4. QuestionManager decomposition and validation
5. Orchestrator question integration
"""

import pytest
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.world_model.models import (
    ResearchQuestion, QuestionStatus, TaskPriority, generate_id
)
from src.world_model.database import Database
from src.world_model.world_model import WorldModel


class TestResearchQuestionModel:
    """Test the ResearchQuestion Pydantic model."""
    
    def test_create_question(self):
        """Basic question creation."""
        q = ResearchQuestion(
            question_text="What factors affect solar cell efficiency?",
            priority=TaskPriority.HIGH,
            keywords=["solar", "efficiency", "photovoltaic"]
        )
        
        assert q.id.startswith("q_")
        assert q.question_text == "What factors affect solar cell efficiency?"
        assert q.status == QuestionStatus.UNANSWERED
        assert q.priority == TaskPriority.HIGH
        assert len(q.keywords) == 3
        assert q.confidence_score == 0.0
        assert q.evidence_count == 0
    
    def test_question_status_methods(self):
        """Test is_answered() and needs_attention() helpers."""
        q = ResearchQuestion(
            question_text="Test question",
            status=QuestionStatus.UNANSWERED
        )
        
        assert not q.is_answered()
        assert q.needs_attention()
        
        q.status = QuestionStatus.PARTIAL
        assert not q.is_answered()
        assert q.needs_attention()
        
        q.status = QuestionStatus.ANSWERED
        assert q.is_answered()
        assert not q.needs_attention()
        
        q.status = QuestionStatus.SKIPPED
        assert not q.is_answered()
        assert not q.needs_attention()


class TestDatabaseQuestions:
    """Test database CRUD for research questions."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = Database(db_path)
            yield db
            db.close()
    
    def test_insert_and_get_question(self, db):
        """Insert and retrieve a question."""
        q = ResearchQuestion(
            question_text="What is the optimal learning rate?",
            area_name="Machine Learning",
            keywords=["learning rate", "optimization"],
            priority=TaskPriority.HIGH
        )
        
        q_id = db.insert_question(q)
        assert q_id == q.id
        
        retrieved = db.get_question(q_id)
        assert retrieved is not None
        assert retrieved.question_text == q.question_text
        assert retrieved.area_name == "Machine Learning"
        assert retrieved.keywords == ["learning rate", "optimization"]
        assert retrieved.status == QuestionStatus.UNANSWERED
    
    def test_get_all_questions(self, db):
        """Get all questions."""
        for i in range(3):
            q = ResearchQuestion(
                question_text=f"Question {i}",
                priority=TaskPriority.MEDIUM
            )
            db.insert_question(q)
        
        all_q = db.get_all_questions()
        assert len(all_q) == 3
    
    def test_get_unanswered_questions(self, db):
        """Get only unanswered/partial questions."""
        # Create mix of statuses
        statuses = [
            QuestionStatus.UNANSWERED,
            QuestionStatus.PARTIAL,
            QuestionStatus.ANSWERED,
            QuestionStatus.UNANSWERED,
        ]
        
        for i, status in enumerate(statuses):
            q = ResearchQuestion(
                question_text=f"Question {i}",
                status=status
            )
            db.insert_question(q)
        
        unanswered = db.get_unanswered_questions()
        assert len(unanswered) == 3  # 2 unanswered + 1 partial
    
    def test_update_question_status(self, db):
        """Update question status and answer info."""
        q = ResearchQuestion(
            question_text="Test question",
            status=QuestionStatus.UNANSWERED
        )
        q_id = db.insert_question(q)
        
        # Update to answered
        db.update_question_status(
            question_id=q_id,
            status="answered",
            answer_summary="The answer is 42",
            confidence_score=0.85,
            evidence_count=3,
            cycle_answered=5
        )
        
        updated = db.get_question(q_id)
        assert updated.status == QuestionStatus.ANSWERED
        assert updated.answer_summary == "The answer is 42"
        assert updated.confidence_score == 0.85
        assert updated.evidence_count == 3
        assert updated.cycle_answered == 5
        assert updated.answered_at is not None
    
    def test_add_finding_to_question(self, db):
        """Link findings to a question."""
        q = ResearchQuestion(question_text="Test question")
        q_id = db.insert_question(q)
        
        # Add findings
        db.add_finding_to_question(q_id, "f_abc123")
        db.add_finding_to_question(q_id, "f_def456")
        db.add_finding_to_question(q_id, "f_abc123")  # Duplicate - should not add
        
        updated = db.get_question(q_id)
        assert len(updated.related_finding_ids) == 2
        assert updated.evidence_count == 2


class TestWorldModelQuestions:
    """Test WorldModel question methods."""
    
    @pytest.fixture
    def wm(self):
        """Create a temporary WorldModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            wm = WorldModel(db_path)
            yield wm
            wm.close()
    
    def test_add_and_get_question(self, wm):
        """Add question through WorldModel API."""
        q_id = wm.add_question(
            question_text="How do Q-learning agents converge?",
            area_name="Reinforcement Learning",
            keywords=["Q-learning", "convergence", "agent"],
            priority="high",
            cycle=1
        )
        
        assert q_id.startswith("q_")
        
        retrieved = wm.get_question(q_id)
        assert retrieved is not None
        assert retrieved.question_text == "How do Q-learning agents converge?"
        assert retrieved.priority == TaskPriority.HIGH
    
    def test_get_question_summary(self, wm):
        """Test question summary generation."""
        # Add questions with different statuses
        wm.add_question("Unanswered question 1", priority="high")
        wm.add_question("Unanswered question 2", priority="medium")
        
        # Get summary
        summary = wm.get_question_summary()
        
        assert "RESEARCH QUESTIONS STATUS:" in summary
        assert "UNANSWERED (2):" in summary
        assert "Unanswered question 1" in summary
    
    def test_update_question(self, wm):
        """Test updating question through WorldModel."""
        q_id = wm.add_question("Test question")
        
        wm.update_question(
            question_id=q_id,
            status="partial",
            confidence_score=0.5,
            evidence_count=1
        )
        
        updated = wm.get_question(q_id)
        assert updated.status == QuestionStatus.PARTIAL
        assert updated.confidence_score == 0.5
    
    def test_world_model_summary_includes_questions(self, wm):
        """Verify get_summary() includes question status."""
        wm.add_question("Test question 1", priority="high")
        wm.add_question("Test question 2", priority="medium")
        
        summary = wm.get_summary()
        
        assert "RESEARCH QUESTIONS STATUS:" in summary
        assert "UNANSWERED" in summary


class TestQuestionManager:
    """Test QuestionManager decomposition and validation."""
    
    @pytest.fixture
    def wm(self):
        """Create a temporary WorldModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            wm = WorldModel(db_path)
            yield wm
            wm.close()
    
    def test_heuristic_decomposition(self, wm):
        """Test fallback heuristic decomposition without LLM."""
        from src.orchestration.research_plan import QuestionManager
        
        # Create manager with None LLM (will use heuristic)
        manager = QuestionManager(llm_client=None, world_model=wm)
        
        # Decompose without storing (to avoid LLM call)
        questions = manager._decompose_heuristic(
            "Investigate the impact of temperature on battery performance"
        )
        
        assert len(questions) >= 3
        assert all("question_text" in q for q in questions)
        assert all("keywords" in q for q in questions)
        assert all("priority" in q for q in questions)
    
    def test_heuristic_validation(self, wm):
        """Test fallback heuristic validation."""
        from src.orchestration.research_plan import QuestionManager
        
        # Add questions
        wm.add_question(
            "What affects battery performance?",
            keywords=["battery", "performance"]
        )
        wm.add_question(
            "How does temperature impact efficiency?",
            keywords=["temperature", "efficiency"]
        )
        
        manager = QuestionManager(llm_client=None, world_model=wm)
        
        # Create mock findings
        findings = [
            {"id": "f_1", "claim": "Battery performance degrades at high temperatures"},
            {"id": "f_2", "claim": "Lower temperatures improve battery longevity"},
            {"id": "f_3", "claim": "Temperature efficiency follows exponential curve"},
        ]
        
        questions = wm.get_all_questions()
        validation = manager._validate_heuristic(questions, findings)
        
        assert "evaluations" in validation
        assert "overall_progress" in validation
        assert len(validation["evaluations"]) == 2
    
    def test_format_questions_for_task_generation(self, wm):
        """Test formatting questions for task generation prompt."""
        from src.orchestration.research_plan import QuestionManager
        
        # Add questions
        wm.add_question(
            "What are the key factors?",
            keywords=["factors", "key"],
            priority="high"
        )
        wm.add_question(
            "How do they interact?",
            keywords=["interaction"],
            priority="medium"
        )
        
        manager = QuestionManager(llm_client=None, world_model=wm)
        
        formatted = manager.format_questions_for_task_generation()
        
        assert "RESEARCH QUESTIONS TO ADDRESS:" in formatted
        assert "UNANSWERED" in formatted
        assert "What are the key factors?" in formatted
        assert "🔴" in formatted  # High priority marker
    
    def test_completion_status(self, wm):
        """Test get_completion_status()."""
        from src.orchestration.research_plan import QuestionManager
        
        manager = QuestionManager(llm_client=None, world_model=wm)
        
        # No questions = complete
        status = manager.get_completion_status()
        assert status["complete"] == True
        assert status["percentage"] == 100
        
        # Add questions
        q1 = wm.add_question("Q1")
        q2 = wm.add_question("Q2")
        wm.update_question(q1, status="answered")
        
        status = manager.get_completion_status()
        assert status["complete"] == False
        assert status["percentage"] == 50  # 1 answered, 1 unanswered
        assert status["answered"] == 1
        assert status["unanswered"] == 1


class TestDatabaseStatistics:
    """Test that database statistics include question counts."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = Database(db_path)
            yield db
            db.close()
    
    def test_statistics_include_questions(self, db):
        """Verify get_statistics() includes question metrics."""
        # Add some questions
        for status in [QuestionStatus.UNANSWERED, QuestionStatus.PARTIAL, QuestionStatus.ANSWERED]:
            q = ResearchQuestion(question_text=f"Q with {status}", status=status)
            db.insert_question(q)
        
        stats = db.get_statistics()
        
        assert "questions_by_status" in stats
        assert "total_questions" in stats
        assert stats["total_questions"] == 3
        assert stats["questions_by_status"]["unanswered"] == 1
        assert stats["questions_by_status"]["partial"] == 1
        assert stats["questions_by_status"]["answered"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
