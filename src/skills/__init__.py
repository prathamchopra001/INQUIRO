"""
INQUIRO Skills System.

Provides skill-based prompt enhancement for LLM calls.
Skills are portable markdown files that inject domain knowledge
into prompts, enabling smaller models to perform better.

Key Components:
- SkillLoader: Loads and caches SKILL.md files
- SkillGenerator: Auto-generates missing skills using teacher model
- SkillManager: High-level interface combining both

Usage:
    from src.skills import SkillManager
    
    manager = SkillManager(llm_client, skills_dir="./skills")
    skill = manager.get_skill("finding_extraction")
"""

from src.skills.skill_loader import (
    Skill,
    SkillLoader,
    ROLE_TO_SKILL_MAP,
    get_skill_name_for_role,
)

from src.skills.skill_generator import (
    SkillGenerator,
    SkillManager,
    ROLE_CONTEXTS,
    get_role_context,
)

__all__ = [
    "Skill",
    "SkillLoader",
    "SkillGenerator", 
    "SkillManager",
    "ROLE_TO_SKILL_MAP",
    "ROLE_CONTEXTS",
    "get_skill_name_for_role",
    "get_role_context",
]
