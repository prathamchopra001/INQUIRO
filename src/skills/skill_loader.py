"""
Skill Loader for INQUIRO.

Loads and caches SKILL.md files for prompt injection.
Works with SkillGenerator to create missing skills on-the-fly.

Inspired by HuggingFace UpSkill - but integrated into INQUIRO's
existing model routing infrastructure.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """
    Represents a loaded skill with its content and metadata.
    
    Attributes:
        name: Skill identifier (matches directory name)
        content: The SKILL.md content to inject into prompts
        metadata: Optional metadata from skill_meta.json
        loaded_at: When this skill was loaded
    """
    name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=datetime.now)
    
    @property
    def token_estimate(self) -> int:
        """Rough token count estimate (words * 1.3)."""
        return int(len(self.content.split()) * 1.3)
    
    @property
    def version(self) -> str:
        """Skill version from metadata, or '1.0.0'."""
        return self.metadata.get("version", "1.0.0")
    
    @property
    def description(self) -> str:
        """Skill description from metadata."""
        return self.metadata.get("description", "")
    
    @property
    def was_auto_generated(self) -> bool:
        """Whether this skill was auto-generated."""
        return self.metadata.get("auto_generated", False)


class SkillLoader:
    """
    Loads and caches skill files for prompt injection.
    
    Skills are stored as SKILL.md files in a skills directory,
    organized by role name:
    
        skills/
        ├── finding_extraction/
        │   ├── SKILL.md
        │   └── skill_meta.json
        ├── query_formulation/
        │   ├── SKILL.md
        │   └── skill_meta.json
        └── ...
    
    Usage:
        loader = SkillLoader(skills_dir="./skills")
        
        # Get skill for a role (returns None if not found)
        skill = loader.get_skill("finding_extraction")
        
        if skill:
            enhanced_prompt = f"{skill.content}\\n\\n{original_prompt}"
    
    The loader caches skills in memory to avoid repeated disk reads.
    """
    
    def __init__(
        self,
        skills_dir: str = "./skills",
        enabled: bool = True,
        cache_enabled: bool = True,
    ):
        """
        Initialize the SkillLoader.
        
        Args:
            skills_dir: Path to the skills directory
            enabled: Whether skill loading is enabled
            cache_enabled: Whether to cache loaded skills in memory
        """
        self.skills_dir = Path(skills_dir)
        self.enabled = enabled
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Optional[Skill]] = {}
        
        # Create skills directory if it doesn't exist
        if self.enabled and not self.skills_dir.exists():
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created skills directory: {self.skills_dir}")
    
    def get_skill(self, role: str) -> Optional[Skill]:
        """
        Get the skill for a given role.
        
        Args:
            role: The role name (e.g., "finding_extraction")
            
        Returns:
            Skill object if found, None otherwise
        """
        if not self.enabled:
            return None
        
        # Normalize role name
        normalized_role = self._normalize_role(role)
        
        # Check cache first
        if self.cache_enabled and normalized_role in self._cache:
            return self._cache[normalized_role]
        
        # Load from disk
        skill = self._load_skill(normalized_role)
        
        # Cache the result
        if self.cache_enabled:
            self._cache[normalized_role] = skill
        
        return skill
    
    def skill_exists(self, role: str) -> bool:
        """Check if a skill file exists for the given role."""
        normalized_role = self._normalize_role(role)
        skill_file = self.skills_dir / normalized_role / "SKILL.md"
        return skill_file.exists()
    
    def _normalize_role(self, role: str) -> str:
        """Normalize role name for consistent lookup."""
        return role.lower().replace("-", "_").replace(" ", "_")
    
    def _load_skill(self, role: str) -> Optional[Skill]:
        """Load a skill from disk."""
        skill_dir = self.skills_dir / role
        skill_file = skill_dir / "SKILL.md"
        meta_file = skill_dir / "skill_meta.json"
        
        if not skill_file.exists():
            return None
        
        try:
            # Load skill content
            content = skill_file.read_text(encoding="utf-8").strip()
            
            if not content:
                logger.warning(f"Skill file is empty: {skill_file}")
                return None
            
            # Load metadata if available
            metadata = {}
            if meta_file.exists():
                try:
                    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid skill metadata: {e}")
            
            skill = Skill(
                name=role,
                content=content,
                metadata=metadata,
            )
            
            logger.info(
                f"📚 Loaded skill '{role}' "
                f"(~{skill.token_estimate} tokens, v{skill.version})"
            )
            
            return skill
            
        except Exception as e:
            logger.error(f"Failed to load skill '{role}': {e}")
            return None
    
    def save_skill(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a skill to disk.
        
        Args:
            role: Role name for the skill
            content: SKILL.md content
            metadata: Optional metadata dict
            
        Returns:
            Path to the saved SKILL.md file
        """
        normalized_role = self._normalize_role(role)
        skill_dir = self.skills_dir / normalized_role
        skill_dir.mkdir(parents=True, exist_ok=True)
        
        # Save skill content
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")
        
        # Save metadata
        if metadata:
            meta_file = skill_dir / "skill_meta.json"
            meta_file.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8"
            )
        
        # Invalidate cache for this role
        if normalized_role in self._cache:
            del self._cache[normalized_role]
        
        logger.info(f"💾 Saved skill '{normalized_role}' to {skill_file}")
        return skill_file
    
    def format_skill_for_injection(self, skill: Skill) -> str:
        """
        Format a skill for injection into a prompt.
        
        Wraps the skill content in clear markers.
        """
        return (
            f"<skill name=\"{skill.name}\" version=\"{skill.version}\">\n"
            f"{skill.content}\n"
            f"</skill>"
        )
    
    def clear_cache(self) -> int:
        """Clear the skill cache. Returns count of cleared items."""
        count = len(self._cache)
        self._cache.clear()
        return count
    
    def list_available_skills(self) -> list[str]:
        """List all available skills in the skills directory."""
        if not self.skills_dir.exists():
            return []
        
        skills = []
        for item in self.skills_dir.iterdir():
            if item.is_dir() and (item / "SKILL.md").exists():
                skills.append(item.name)
        
        return sorted(skills)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded skills."""
        available = self.list_available_skills()
        cached_skills = [s for s in self._cache.values() if s is not None]
        
        return {
            "enabled": self.enabled,
            "skills_dir": str(self.skills_dir),
            "available_skills": available,
            "available_count": len(available),
            "cached_count": len(cached_skills),
            "total_cached_tokens": sum(s.token_estimate for s in cached_skills),
        }


# =============================================================================
# ROLE MAPPINGS
# =============================================================================

# Maps LLM roles to skill directory names for flexibility
ROLE_TO_SKILL_MAP = {
    # Finding extraction (data analysis - discovers new things)
    "extraction": "finding_extraction",
    "finding_extraction": "finding_extraction",
    
    # Literature extraction (summarizes papers - requires attribution)
    "literature_extraction": "literature_extraction",
    "lit_extraction": "literature_extraction",
    
    # Query generation
    "query_generation": "query_formulation",
    "query_formulation": "query_formulation",
    
    # Data analysis
    "code_generation": "code_generation",
    "plan_generation": "code_generation",
    
    # Orchestrator
    "task_generation": "task_generation",
    "gap_analysis": "gap_analysis",
    "completion_check": "completion_check",
    
    # Reports
    "report_narrative": "report_narrative",
    "executive_summary": "executive_summary",
    
    # Questions
    "question_decomposition": "question_decomposition",
    "question_validation": "question_validation",
}


def get_skill_name_for_role(role: str) -> str:
    """
    Get the skill directory name for a given LLM role.
    
    Args:
        role: The role passed to complete_for_role()
        
    Returns:
        Skill directory name (may be different from role)
    """
    normalized = role.lower().replace("-", "_")
    return ROLE_TO_SKILL_MAP.get(normalized, normalized)
