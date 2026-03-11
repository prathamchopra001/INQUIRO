"""
Tests for the INQUIRO Skill System.

Tests:
1. SkillLoader - loading and caching skills
2. SkillGenerator - generating skills on-the-fly
3. SkillManager - combined interface
4. Integration - skill injection into LLM calls
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.skills.skill_loader import (
    Skill,
    SkillLoader,
    get_skill_name_for_role,
    ROLE_TO_SKILL_MAP,
)
from src.skills.skill_generator import (
    SkillGenerator,
    SkillManager,
    get_role_context,
    ROLE_CONTEXTS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_skills_dir():
    """Create a temporary skills directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_skill_content():
    """Sample SKILL.md content."""
    return """# Test Skill

## Task
Extract key information from text.

## Output Format
```json
{"result": "extracted text"}
```

## Decision Framework
- If text mentions X, extract it
- Otherwise return empty

## Examples
Input: "Hello world"
Output: {"result": "Hello world"}
"""


@pytest.fixture
def loader(temp_skills_dir):
    """Create a SkillLoader with temp directory."""
    return SkillLoader(skills_dir=temp_skills_dir, enabled=True)


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    mock = Mock()
    mock.complete_for_role = Mock(return_value=Mock(content="""# Generated Skill

## Task
Perform the task as requested.

## Output Format
Return valid JSON.

## Decision Framework
1. Analyze input
2. Generate output

## Examples
Input: test
Output: {"result": "test"}
"""))
    return mock


# =============================================================================
# SKILL LOADER TESTS
# =============================================================================

class TestSkillLoader:
    """Tests for SkillLoader."""
    
    def test_init_creates_directory(self, temp_skills_dir):
        """Should create skills directory if it doesn't exist."""
        new_dir = Path(temp_skills_dir) / "new_skills"
        loader = SkillLoader(skills_dir=str(new_dir), enabled=True)
        assert new_dir.exists()
    
    def test_disabled_returns_none(self, temp_skills_dir, sample_skill_content):
        """When disabled, always returns None."""
        loader = SkillLoader(skills_dir=temp_skills_dir, enabled=False)
        
        # Create a skill
        skill_dir = Path(temp_skills_dir) / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        # Should return None even though skill exists
        assert loader.get_skill("test_skill") is None
    
    def test_load_existing_skill(self, loader, temp_skills_dir, sample_skill_content):
        """Should load a skill from disk."""
        skill_dir = Path(temp_skills_dir) / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        skill = loader.get_skill("test_skill")
        
        assert skill is not None
        assert skill.name == "test_skill"
        assert "# Test Skill" in skill.content
        assert skill.token_estimate > 0
    
    def test_load_with_metadata(self, loader, temp_skills_dir, sample_skill_content):
        """Should load skill metadata if present."""
        skill_dir = Path(temp_skills_dir) / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        (skill_dir / "skill_meta.json").write_text(json.dumps({
            "version": "2.0.0",
            "description": "Test skill",
            "auto_generated": False,
        }))
        
        skill = loader.get_skill("test_skill")
        
        assert skill.version == "2.0.0"
        assert skill.description == "Test skill"
        assert skill.was_auto_generated is False
    
    def test_missing_skill_returns_none(self, loader):
        """Should return None for non-existent skill."""
        assert loader.get_skill("nonexistent") is None
    
    def test_caching(self, loader, temp_skills_dir, sample_skill_content):
        """Should cache loaded skills."""
        skill_dir = Path(temp_skills_dir) / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        # Load twice
        skill1 = loader.get_skill("test_skill")
        skill2 = loader.get_skill("test_skill")
        
        # Should be same object (cached)
        assert skill1 is skill2
    
    def test_cache_disabled(self, temp_skills_dir, sample_skill_content):
        """Cache can be disabled."""
        loader = SkillLoader(skills_dir=temp_skills_dir, cache_enabled=False)
        
        skill_dir = Path(temp_skills_dir) / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        # Load twice
        skill1 = loader.get_skill("test_skill")
        skill2 = loader.get_skill("test_skill")
        
        # Should be different objects (not cached)
        assert skill1 is not skill2
        assert skill1.content == skill2.content
    
    def test_normalize_role(self, loader, temp_skills_dir, sample_skill_content):
        """Should normalize role names."""
        skill_dir = Path(temp_skills_dir) / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        # Various formats should work
        assert loader.get_skill("test_skill") is not None
        assert loader.get_skill("TEST_SKILL") is not None
        assert loader.get_skill("test-skill") is not None
    
    def test_save_skill(self, loader, temp_skills_dir):
        """Should save a skill to disk."""
        content = "# New Skill\n\nThis is new."
        metadata = {"version": "1.0.0", "auto_generated": True}
        
        path = loader.save_skill("new_skill", content, metadata)
        
        assert path.exists()
        assert path.read_text() == content
        
        meta_path = path.parent / "skill_meta.json"
        assert meta_path.exists()
        assert json.loads(meta_path.read_text()) == metadata
    
    def test_save_invalidates_cache(self, loader, temp_skills_dir, sample_skill_content):
        """Saving a skill should invalidate its cache."""
        skill_dir = Path(temp_skills_dir) / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        # Load to cache
        skill1 = loader.get_skill("test_skill")
        assert "# Test Skill" in skill1.content
        
        # Save new version
        loader.save_skill("test_skill", "# Updated\n\nNew content.")
        
        # Should load new version
        skill2 = loader.get_skill("test_skill")
        assert "# Updated" in skill2.content
    
    def test_list_available_skills(self, loader, temp_skills_dir, sample_skill_content):
        """Should list all available skills."""
        # Create multiple skills
        for name in ["skill_a", "skill_b", "skill_c"]:
            skill_dir = Path(temp_skills_dir) / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        skills = loader.list_available_skills()
        
        assert sorted(skills) == ["skill_a", "skill_b", "skill_c"]
    
    def test_format_skill_for_injection(self, loader, temp_skills_dir, sample_skill_content):
        """Should format skill with XML-like tags."""
        skill_dir = Path(temp_skills_dir) / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        skill = loader.get_skill("test_skill")
        formatted = loader.format_skill_for_injection(skill)
        
        assert '<skill name="test_skill"' in formatted
        assert '</skill>' in formatted
        assert sample_skill_content in formatted


# =============================================================================
# SKILL GENERATOR TESTS
# =============================================================================

class TestSkillGenerator:
    """Tests for SkillGenerator."""
    
    def test_disabled_returns_none(self, mock_llm, loader):
        """When disabled, should return None."""
        generator = SkillGenerator(mock_llm, loader, enabled=False)
        
        result = generator.generate_skill("test_role")
        
        assert result is None
        mock_llm.complete_for_role.assert_not_called()
    
    def test_skip_if_exists(self, mock_llm, loader, temp_skills_dir, sample_skill_content):
        """Should not regenerate existing skills."""
        # Create existing skill
        skill_dir = Path(temp_skills_dir) / "test_role"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        generator = SkillGenerator(mock_llm, loader)
        result = generator.generate_skill("test_role")
        
        # Should load existing, not generate
        mock_llm.complete_for_role.assert_not_called()
        assert result is not None
        assert "# Test Skill" in result.content
    
    def test_generate_new_skill(self, mock_llm, loader, temp_skills_dir):
        """Should generate and save new skill."""
        generator = SkillGenerator(mock_llm, loader)
        
        result = generator.generate_skill("new_role", example_prompt="Test prompt")
        
        # Should have called LLM
        mock_llm.complete_for_role.assert_called_once()
        call_kwargs = mock_llm.complete_for_role.call_args
        assert call_kwargs[1]["skip_skill_injection"] is True
        
        # Should have saved skill
        skill_file = Path(temp_skills_dir) / "new_role" / "SKILL.md"
        assert skill_file.exists()
        
        # Should return skill
        assert result is not None
        assert result.was_auto_generated is True
    
    def test_force_regenerate(self, mock_llm, loader, temp_skills_dir, sample_skill_content):
        """force=True should regenerate even if exists."""
        # Create existing skill
        skill_dir = Path(temp_skills_dir) / "test_role"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        generator = SkillGenerator(mock_llm, loader)
        result = generator.generate_skill("test_role", force=True)
        
        # Should have called LLM
        mock_llm.complete_for_role.assert_called_once()
        
        # Should have new content
        assert "# Generated Skill" in result.content
    
    def test_validate_content_too_short(self, mock_llm, loader):
        """Should reject too-short content."""
        mock_llm.complete_for_role.return_value = Mock(content="Too short")
        
        generator = SkillGenerator(mock_llm, loader)
        result = generator.generate_skill("new_role")
        
        assert result is None
    
    def test_validate_content_no_headers(self, mock_llm, loader):
        """Should reject content without markdown headers."""
        mock_llm.complete_for_role.return_value = Mock(
            content="This is just plain text without any headers. " * 20
        )
        
        generator = SkillGenerator(mock_llm, loader)
        result = generator.generate_skill("new_role")
        
        assert result is None


# =============================================================================
# SKILL MANAGER TESTS
# =============================================================================

class TestSkillManager:
    """Tests for SkillManager."""
    
    def test_get_existing_skill(self, mock_llm, temp_skills_dir, sample_skill_content):
        """Should return existing skill without generating."""
        # Create existing skill
        skill_dir = Path(temp_skills_dir) / "test_role"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        manager = SkillManager(mock_llm, temp_skills_dir, auto_generate=True)
        skill = manager.get_skill("test_role")
        
        assert skill is not None
        mock_llm.complete_for_role.assert_not_called()
    
    def test_auto_generate_missing(self, mock_llm, temp_skills_dir):
        """Should auto-generate missing skills when enabled."""
        manager = SkillManager(mock_llm, temp_skills_dir, auto_generate=True)
        skill = manager.get_skill("new_role")
        
        # Should have generated
        mock_llm.complete_for_role.assert_called_once()
        assert skill is not None
    
    def test_no_auto_generate(self, mock_llm, temp_skills_dir):
        """Should not auto-generate when disabled."""
        manager = SkillManager(mock_llm, temp_skills_dir, auto_generate=False)
        skill = manager.get_skill("new_role")
        
        assert skill is None
        mock_llm.complete_for_role.assert_not_called()
    
    def test_inject_skill(self, mock_llm, temp_skills_dir, sample_skill_content):
        """Should inject skill into system prompt."""
        # Create skill
        skill_dir = Path(temp_skills_dir) / "test_role"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(sample_skill_content)
        
        manager = SkillManager(mock_llm, temp_skills_dir)
        enhanced = manager.inject_skill(
            "test_role",
            system_prompt="You are helpful.",
        )
        
        assert '<skill name="test_role"' in enhanced
        assert "You are helpful." in enhanced
    
    def test_inject_skill_no_skill(self, mock_llm, temp_skills_dir):
        """Should return original prompt if no skill."""
        manager = SkillManager(mock_llm, temp_skills_dir, auto_generate=False)
        
        enhanced = manager.inject_skill(
            "nonexistent",
            system_prompt="You are helpful.",
        )
        
        assert enhanced == "You are helpful."


# =============================================================================
# ROLE MAPPING TESTS
# =============================================================================

class TestRoleMappings:
    """Tests for role-to-skill mappings."""
    
    def test_get_skill_name_for_role(self):
        """Should map roles to skill names."""
        assert get_skill_name_for_role("extraction") == "finding_extraction"
        assert get_skill_name_for_role("finding_extraction") == "finding_extraction"
        assert get_skill_name_for_role("query_generation") == "query_formulation"
    
    def test_unknown_role_returns_normalized(self):
        """Unknown roles should return normalized name."""
        assert get_skill_name_for_role("unknown_role") == "unknown_role"
        assert get_skill_name_for_role("Unknown-Role") == "unknown_role"
    
    def test_role_contexts_exist(self):
        """All mapped roles should have context."""
        for skill_name in set(ROLE_TO_SKILL_MAP.values()):
            if skill_name in ROLE_CONTEXTS:
                context = get_role_context(skill_name)
                assert len(context) > 10


# =============================================================================
# SEED SKILL TESTS
# =============================================================================

class TestSeedSkills:
    """Tests for pre-built seed skills."""
    
    def test_finding_extraction_skill_exists(self):
        """The finding_extraction seed skill should exist."""
        skill_path = Path("./skills/finding_extraction/SKILL.md")
        assert skill_path.exists(), f"Seed skill not found at {skill_path}"
    
    def test_finding_extraction_skill_valid(self):
        """The finding_extraction skill should have required sections."""
        skill_path = Path("./skills/finding_extraction/SKILL.md")
        content = skill_path.read_text(encoding='utf-8')
        
        # Check for expected sections
        assert "## Task" in content
        assert "## Output Format" in content
        assert "## Decision Framework" in content
        assert "## Examples" in content
        assert "json" in content.lower()  # Should mention JSON


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
