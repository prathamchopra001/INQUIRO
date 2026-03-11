"""
Tests for Piece 3: Report Validation Loop

Tests:
1. Orchestrator.get_research_gaps()
2. ReportGenerator._generate_question_section()
3. Integration of question section into reports
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.world_model.world_model import WorldModel
from src.world_model.models import ResearchQuestion, QuestionStatus, TaskPriority
from src.agents.orchestrator import OrchestratorAgent
from src.reports.generator import ReportGenerator


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except:
        pass


@pytest.fixture
def world_model(temp_db):
    """Create a WorldModel instance."""
    return WorldModel(db_path=temp_db)


@pytest.fixture
def orchestrator(world_model):
    """Create an OrchestratorAgent with mock LLM."""
    class MockLLM:
        def complete_for_role(self, prompt, role, system=None):
            class Response:
                content = '[]'
            return Response()
    
    orch = OrchestratorAgent(
        llm_client=MockLLM(),
        world_model=world_model
    )
    # Mark questions as initialized
    orch._questions_initialized = True
    return orch


@pytest.fixture
def report_generator(world_model):
    """Create a ReportGenerator with mock LLM."""
    class MockLLM:
        def complete_for_role(self, prompt, role, system=None):
            class Response:
                content = "This is a mock narrative for testing purposes."
            return Response()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ReportGenerator(
            llm_client=MockLLM(),
            world_model=world_model,
            output_dir=tmpdir
        )


# =============================================================================
# TEST: get_research_gaps()
# =============================================================================

class TestGetResearchGaps:
    """Tests for Orchestrator.get_research_gaps()"""
    
    def test_no_questions_initialized(self, world_model):
        """Returns no gaps when questions not initialized."""
        class MockLLM:
            def complete_for_role(self, *args, **kwargs):
                class Response:
                    content = '[]'
                return Response()
        
        orch = OrchestratorAgent(MockLLM(), world_model)
        # Don't initialize questions
        
        gaps = orch.get_research_gaps()
        
        assert gaps["has_gaps"] is False
        assert gaps["gap_count"] == 0
        assert gaps["gap_summary"] == "No questions initialized"
    
    def test_no_questions_exist(self, orchestrator, world_model):
        """Returns no gaps when no questions in world model."""
        gaps = orchestrator.get_research_gaps()
        
        assert gaps["has_gaps"] is False
        assert gaps["gap_count"] == 0
    
    def test_all_questions_answered(self, orchestrator, world_model):
        """Returns no gaps when all questions are answered."""
        # Add answered questions
        for i in range(3):
            q = ResearchQuestion(
                question_text=f"Question {i}",
                status=QuestionStatus.ANSWERED,
                priority=TaskPriority.MEDIUM
            )
            world_model.db.insert_question(q)
        
        gaps = orchestrator.get_research_gaps()
        
        assert gaps["has_gaps"] is False
        assert gaps["gap_count"] == 0
        assert "addressed" in gaps["gap_summary"].lower()
    
    def test_some_unanswered_questions(self, orchestrator, world_model):
        """Returns gaps when unanswered questions exist."""
        # Add mix of answered and unanswered
        world_model.db.insert_question(ResearchQuestion(
            question_text="Answered question",
            status=QuestionStatus.ANSWERED
        ))
        world_model.db.insert_question(ResearchQuestion(
            question_text="Unanswered question 1",
            status=QuestionStatus.UNANSWERED,
            priority=TaskPriority.HIGH
        ))
        world_model.db.insert_question(ResearchQuestion(
            question_text="Unanswered question 2",
            status=QuestionStatus.UNANSWERED,
            priority=TaskPriority.MEDIUM
        ))
        
        gaps = orchestrator.get_research_gaps(min_unanswered=2)
        
        assert gaps["has_gaps"] is True
        assert gaps["gap_count"] == 2
        assert len(gaps["high_priority_gaps"]) == 1
        assert len(gaps["all_gaps"]) == 2
    
    def test_partial_questions_count_as_gaps(self, orchestrator, world_model):
        """Partial status should count as gaps."""
        world_model.db.insert_question(ResearchQuestion(
            question_text="Partial question 1",
            status=QuestionStatus.PARTIAL
        ))
        world_model.db.insert_question(ResearchQuestion(
            question_text="Partial question 2",
            status=QuestionStatus.PARTIAL
        ))
        
        gaps = orchestrator.get_research_gaps(min_unanswered=2)
        
        assert gaps["has_gaps"] is True
        assert gaps["gap_count"] == 2
    
    def test_high_priority_single_gap_triggers(self, orchestrator, world_model):
        """Single high-priority gap should trigger has_gaps."""
        world_model.db.insert_question(ResearchQuestion(
            question_text="Critical unanswered question",
            status=QuestionStatus.UNANSWERED,
            priority=TaskPriority.HIGH
        ))
        
        # min_unanswered=2 but 1 high-priority should still trigger
        gaps = orchestrator.get_research_gaps(min_unanswered=2)
        
        assert gaps["has_gaps"] is True
        assert gaps["gap_count"] == 1
        assert len(gaps["high_priority_gaps"]) == 1
    
    def test_gap_summary_content(self, orchestrator, world_model):
        """Gap summary should describe the gaps."""
        world_model.db.insert_question(ResearchQuestion(
            question_text="What is the optimal learning rate for Q-learning?",
            status=QuestionStatus.UNANSWERED,
            priority=TaskPriority.HIGH
        ))
        world_model.db.insert_question(ResearchQuestion(
            question_text="How does exploration decay affect convergence?",
            status=QuestionStatus.UNANSWERED,
            priority=TaskPriority.MEDIUM
        ))
        
        gaps = orchestrator.get_research_gaps()
        
        assert "2 questions remain" in gaps["gap_summary"]
        assert "1 high-priority" in gaps["gap_summary"]


# =============================================================================
# TEST: _generate_question_section()
# =============================================================================

class TestGenerateQuestionSection:
    """Tests for ReportGenerator._generate_question_section()"""
    
    def test_empty_questions_list(self, report_generator):
        """Returns empty list for no questions."""
        lines = report_generator._generate_question_section([])
        assert lines == []
    
    def test_section_header(self, report_generator, world_model):
        """Section should have proper header."""
        q = ResearchQuestion(
            question_text="Test question",
            status=QuestionStatus.ANSWERED
        )
        world_model.db.insert_question(q)
        questions = world_model.get_all_questions()
        
        lines = report_generator._generate_question_section(questions)
        
        assert any("Research Questions Addressed" in line for line in lines)
    
    def test_coverage_summary(self, report_generator, world_model):
        """Should show coverage statistics."""
        # Add mix of statuses
        world_model.db.insert_question(ResearchQuestion(
            question_text="Answered Q", status=QuestionStatus.ANSWERED
        ))
        world_model.db.insert_question(ResearchQuestion(
            question_text="Partial Q", status=QuestionStatus.PARTIAL
        ))
        world_model.db.insert_question(ResearchQuestion(
            question_text="Unanswered Q", status=QuestionStatus.UNANSWERED
        ))
        questions = world_model.get_all_questions()
        
        lines = report_generator._generate_question_section(questions)
        section_text = "\n".join(lines)
        
        assert "1/3 fully answered" in section_text
        assert "1 partially answered" in section_text
        assert "1 open" in section_text
    
    def test_answered_section(self, report_generator, world_model):
        """Answered questions should appear in ✅ section."""
        q = ResearchQuestion(
            question_text="What causes X?",
            status=QuestionStatus.ANSWERED,
            priority=TaskPriority.HIGH,
            answer_summary="X is caused by Y.",
            confidence_score=0.85,
            evidence_count=3
        )
        world_model.db.insert_question(q)
        questions = world_model.get_all_questions()
        
        lines = report_generator._generate_question_section(questions)
        section_text = "\n".join(lines)
        
        assert "✅ Answered" in section_text
        assert "What causes X?" in section_text
        assert "X is caused by Y" in section_text
        assert "85%" in section_text  # confidence
        assert "3 findings" in section_text  # evidence count
    
    def test_partial_section(self, report_generator, world_model):
        """Partial questions should appear in 🟡 section."""
        q = ResearchQuestion(
            question_text="How does Z work?",
            status=QuestionStatus.PARTIAL,
            priority=TaskPriority.MEDIUM
        )
        world_model.db.insert_question(q)
        questions = world_model.get_all_questions()
        
        lines = report_generator._generate_question_section(questions)
        section_text = "\n".join(lines)
        
        assert "🟡 Partially Answered" in section_text
        assert "How does Z work?" in section_text
    
    def test_unanswered_section(self, report_generator, world_model):
        """Unanswered questions should appear in ❌ section."""
        q = ResearchQuestion(
            question_text="Why is A better than B?",
            status=QuestionStatus.UNANSWERED,
            priority=TaskPriority.LOW
        )
        world_model.db.insert_question(q)
        questions = world_model.get_all_questions()
        
        lines = report_generator._generate_question_section(questions)
        section_text = "\n".join(lines)
        
        assert "❌ Open Questions" in section_text
        assert "Why is A better than B?" in section_text
    
    def test_priority_badges(self, report_generator, world_model):
        """Questions should have priority badges."""
        world_model.db.insert_question(ResearchQuestion(
            question_text="High Q", status=QuestionStatus.ANSWERED, priority=TaskPriority.HIGH
        ))
        world_model.db.insert_question(ResearchQuestion(
            question_text="Medium Q", status=QuestionStatus.ANSWERED, priority=TaskPriority.MEDIUM
        ))
        world_model.db.insert_question(ResearchQuestion(
            question_text="Low Q", status=QuestionStatus.ANSWERED, priority=TaskPriority.LOW
        ))
        questions = world_model.get_all_questions()
        
        lines = report_generator._generate_question_section(questions)
        section_text = "\n".join(lines)
        
        assert "🔴" in section_text  # High
        assert "🟡" in section_text  # Medium
        assert "🟢" in section_text  # Low


# =============================================================================
# TEST: Report includes question section
# =============================================================================

class TestReportIncludesQuestions:
    """Tests for question section integration in reports."""
    
    def test_report_assembles_with_questions(self, report_generator, world_model):
        """_assemble_report should include question section."""
        # Add a question
        world_model.db.insert_question(ResearchQuestion(
            question_text="Test research question",
            status=QuestionStatus.ANSWERED
        ))
        questions = world_model.get_all_questions()
        
        # Add a finding for the report
        world_model.add_finding(
            claim="Test finding",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1,
            confidence=0.8
        )
        
        stats = world_model.get_statistics()
        sections = []  # Empty sections - will use questions section
        
        report = report_generator._assemble_report(
            objective="Test objective",
            sections=sections,
            stats=stats,
            cycles_completed=1,
            questions=questions
        )
        
        assert "Research Questions Addressed" in report
        assert "Test research question" in report
    
    def test_report_without_questions(self, report_generator, world_model):
        """Report should work without questions."""
        # Add a finding
        world_model.add_finding(
            claim="Test finding",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb", "cell": 1},
            cycle=1,
            confidence=0.8
        )
        
        stats = world_model.get_statistics()
        
        report = report_generator._assemble_report(
            objective="Test objective",
            sections=[],
            stats=stats,
            cycles_completed=1,
            questions=None
        )
        
        # Should not crash and should not have question section
        assert "Research Questions Addressed" not in report
        assert "INQUIRO" in report  # Basic report structure present


# =============================================================================
# INTEGRATION TEST: Full flow
# =============================================================================

class TestReportValidationIntegration:
    """Integration tests for the validation loop components."""
    
    def test_gap_detection_to_report_flow(self, world_model):
        """Test flow from gap detection to report generation."""
        class MockLLM:
            def complete_for_role(self, *args, **kwargs):
                class Response:
                    content = "Mock response"
                return Response()
        
        # Set up orchestrator
        orch = OrchestratorAgent(MockLLM(), world_model)
        orch._questions_initialized = True
        
        # Add unanswered questions (gaps)
        world_model.db.insert_question(ResearchQuestion(
            question_text="Critical question 1",
            status=QuestionStatus.UNANSWERED,
            priority=TaskPriority.HIGH
        ))
        world_model.db.insert_question(ResearchQuestion(
            question_text="Critical question 2",
            status=QuestionStatus.UNANSWERED,
            priority=TaskPriority.HIGH
        ))
        
        # Check gaps
        gaps = orch.get_research_gaps()
        assert gaps["has_gaps"] is True
        assert gaps["gap_count"] == 2
        
        # Generate report with questions
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(MockLLM(), world_model, output_dir=tmpdir)
            
            # Add finding so report generates
            world_model.add_finding(
                claim="Finding",
                finding_type="literature",
                source={"type": "paper", "title": "Test", "doi": ""},
                cycle=1,
                confidence=0.7
            )
            
            report_path = gen.generate_report(
                objective="Test objective",
                cycles_completed=1
            )
            
            # Read report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Report should include questions section with gaps
            assert "Research Questions Addressed" in report_content
            assert "Critical question 1" in report_content
            assert "❌" in report_content  # Open questions marker


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
