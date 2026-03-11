"""
Test Task Skill Generator integration with Data Analysis Agent.

This tests the UpSkill integration:
1. TaskSkillGenerator correctly identifies techniques from task descriptions
2. Generates appropriate skills with correct API patterns
3. Detects infeasible tasks (ESM, ProtBERT, etc.)
"""

import pytest
from src.skills.task_skill_generator import TaskSkillGenerator, KNOWN_PATTERNS


class TestKnownPatterns:
    """Test the KNOWN_PATTERNS dictionary."""
    
    def test_biopython_alignment_pattern_exists(self):
        """Verify BioPython alignment pattern has correct modern API."""
        pattern = KNOWN_PATTERNS.get("biopython_alignment")
        assert pattern is not None
        assert "from Bio import Align" in pattern["correct_imports"][0]
        assert "pairwise2" in str(pattern["wrong_imports"])  # Should warn against deprecated
    
    def test_statsmodels_multitest_pattern(self):
        """Verify statsmodels pattern catches the multitest vs multicomp confusion."""
        pattern = KNOWN_PATTERNS.get("statsmodels_multitest")
        assert pattern is not None
        assert "multitest" in pattern["correct_imports"][0]
        assert "multicomp" in str(pattern["wrong_imports"])  # Wrong module
    
    def test_infeasible_protein_llm_pattern(self):
        """Verify infeasible task detection for ESM/ProtBERT."""
        pattern = KNOWN_PATTERNS.get("infeasible_protein_llm")
        assert pattern is not None
        assert pattern.get("is_infeasible") == True
        assert "esm" in pattern["keywords"]
        assert "protbert" in pattern["keywords"]
        assert "SANDBOX" in pattern["example_code"]  # Should warn about sandbox limitations


class TestTaskSkillGeneratorPatternMatching:
    """Test pattern matching without LLM."""
    
    def test_match_blosum_task(self):
        """Test matching a BLOSUM alignment task to correct pattern."""
        gen = TaskSkillGenerator(llm_client=None, enabled=False)
        
        # Direct pattern matching test
        matches = gen._find_matching_patterns("Calculate BLOSUM62 sequence similarity")
        
        assert len(matches) > 0
        pattern_names = [m["name"] for m in matches]
        assert "biopython_alignment" in pattern_names
    
    def test_match_spearman_task(self):
        """Test matching a correlation task."""
        gen = TaskSkillGenerator(llm_client=None, enabled=False)
        
        matches = gen._find_matching_patterns("Compute Spearman rank correlation between features")
        
        assert len(matches) > 0
        pattern_names = [m["name"] for m in matches]
        assert "correlation_analysis" in pattern_names
    
    def test_match_esm_task_as_infeasible(self):
        """Test that ESM tasks are matched to infeasible pattern."""
        gen = TaskSkillGenerator(llm_client=None, enabled=False)
        
        matches = gen._find_matching_patterns("Fine-tune ESM-1v on antibody developability data")
        
        assert len(matches) > 0
        # Check that infeasible pattern is matched
        infeasible_match = [m for m in matches if m.get("is_infeasible")]
        assert len(infeasible_match) > 0, "ESM task should match infeasible pattern"
    
    def test_match_protbert_task_as_infeasible(self):
        """Test that ProtBERT tasks are matched to infeasible pattern."""
        gen = TaskSkillGenerator(llm_client=None, enabled=False)
        
        matches = gen._find_matching_patterns("Extract ProtBERT embeddings from protein sequences")
        
        assert len(matches) > 0
        infeasible_match = [m for m in matches if m.get("is_infeasible")]
        assert len(infeasible_match) > 0, "ProtBERT task should match infeasible pattern"


class TestPatternFormatting:
    """Test that patterns are formatted correctly for injection into prompts."""
    
    def test_format_known_patterns(self):
        """Test formatting of known patterns for prompt injection."""
        gen = TaskSkillGenerator(llm_client=None, enabled=False)
        
        matches = gen._find_matching_patterns("Calculate Spearman correlation")
        formatted = gen._format_known_patterns(matches[:1])
        
        assert "Correct imports" in formatted
        assert "scipy.stats" in formatted
        assert "spearmanr" in formatted


class TestSkillInjection:
    """Test that skills are actually injected into code generation."""
    
    def test_get_skill_for_code_generation_returns_content(self):
        """Test that get_skill_for_code_generation returns non-empty for known patterns."""
        gen = TaskSkillGenerator(llm_client=None, enabled=True)
        
        # Test ESM task (infeasible)
        skill = gen.get_skill_for_code_generation("Fine-tune ESM-1v on antibody data")
        assert skill, "Should return skill content for ESM task"
        assert "<task_specific_skill>" in skill
        assert "SANDBOX" in skill or "infeasible" in skill.lower() or "GPU" in skill
    
    def test_get_skill_for_spearman_returns_scipy(self):
        """Test that Spearman task gets scipy pattern."""
        gen = TaskSkillGenerator(llm_client=None, enabled=True)
        
        skill = gen.get_skill_for_code_generation("Calculate Spearman correlation")
        assert skill, "Should return skill content for correlation task"
        assert "scipy" in skill.lower() or "spearmanr" in skill
    
    def test_disabled_generator_returns_empty(self):
        """Test that disabled generator returns empty string."""
        gen = TaskSkillGenerator(llm_client=None, enabled=False)
        
        skill = gen.get_skill_for_code_generation("Any task here")
        assert skill == "", "Disabled generator should return empty string"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
