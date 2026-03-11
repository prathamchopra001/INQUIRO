"""
Skill Generator for INQUIRO.

Automatically generates SKILL.md files using a teacher model
when a skill doesn't exist. This enables the system to build
its own skill library over time.

Flow:
1. Agent needs skill for role "X"
2. SkillLoader checks: skill exists? NO
3. SkillGenerator creates skill using teacher model
4. Skill saved to disk for future use
5. Subsequent calls load from cache (fast + cheap)
"""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.llm_client import LLMClient

from src.skills.skill_loader import SkillLoader, get_skill_name_for_role, Skill

logger = logging.getLogger(__name__)


# =============================================================================
# SKILL GENERATION PROMPT
# =============================================================================

SKILL_GENERATION_PROMPT = '''You are an expert at creating skill files that help language models perform specific tasks better.

## Task
Create a SKILL.md file for the role: "{role_name}"

## Context About This Role
{role_context}

## Example Prompt That Uses This Role
```
{example_prompt}
```

## Requirements for the Skill File

Create a comprehensive skill file that includes:

1. **Task Description**: What this role does in 1-2 sentences

2. **Output Format**: Exact format expected (JSON schema if applicable)

3. **Decision Framework**: Step-by-step rules for making decisions
   - When to do X vs Y
   - Confidence scoring guidelines
   - Edge case handling

4. **Common Patterns**: Examples of good outputs
   - 2-3 concrete examples with input → output

5. **Anti-Patterns**: What NOT to do
   - Common mistakes to avoid
   - Invalid outputs to never produce

6. **Quality Checklist**: Final validation steps

## Format Guidelines
- Use clear markdown headers
- Keep total length under 600 words (fits in ~500 tokens)
- Be specific and actionable, not vague
- Include concrete examples, not abstract descriptions

## Output
Return ONLY the skill file content in markdown format.
Do not include any preamble or explanation outside the skill content.
Start directly with the first header.
'''


# =============================================================================
# ROLE CONTEXT DEFINITIONS
# =============================================================================

# Provides context about each role to help generate better skills
ROLE_CONTEXTS = {
    "finding_extraction": """
This role extracts scientific findings from DATA ANALYSIS code outputs.
Input: Code execution output showing statistical results, correlations, patterns
Output: JSON array of findings, each with claim, confidence, evidence, tags
Used by: Data Analysis Agent (NOT Literature Agent)
Critical: Must output valid JSON. Findings are INQUIRO's own discoveries, not from papers.
""",

    "literature_extraction": """
This role extracts findings from ACADEMIC PAPERS with proper attribution.
Input: Text chunks from papers with [Paper ID: xxx] metadata
Output: JSON array of findings with claim (MUST start with attribution), confidence, evidence, paper_id, paper_title, tags
Used by: Literature Search Agent
Critical: EVERY claim must start with "Prior work by [Authors] found..." or "According to [Paper Title]...". 
Never present literature findings as original discoveries. Must extract paper_id from context.
""",

    "query_formulation": """
This role converts task descriptions into effective search queries.
Input: A task description like "Find papers about Q-learning convergence"
Output: JSON array of 3-5 search query strings
Used by: Literature Search Agent  
Critical: Queries should be specific enough to find relevant papers, broad enough to not miss important ones.
""",

    "code_generation": """
This role generates Python code for data analysis tasks.
Input: Task description, dataset info, objective
Output: Python code using pandas, matplotlib, scipy
Used by: Data Analysis Agent
Critical: Code must be executable, handle errors, produce clear outputs.
""",

    "task_generation": """
This role generates research tasks for the next cycle.
Input: Objective, world model summary, current findings
Output: JSON array of tasks with type, description, goal, priority
Used by: Orchestrator Agent
Critical: Tasks should address gaps, avoid redundancy, balance exploration/exploitation.
""",

    "gap_analysis": """
This role analyzes what research questions have been answered vs remain open.
Input: Objective, findings by cycle, completed tasks
Output: JSON with answered questions, open questions, weak areas
Used by: Orchestrator Agent
Critical: Must identify genuine gaps, not just rephrase completed work.
""",

    "question_decomposition": """
This role breaks down a research objective into specific questions.
Input: Research objective, optional domain context
Output: JSON array of questions with text, priority, success criteria
Used by: Question Manager
Critical: Questions should be answerable, measurable, cover the objective comprehensively.
""",

    "question_validation": """
This role evaluates whether research questions have been answered by findings.
Input: Questions, findings collected so far
Output: JSON with status (answered/partial/unanswered) per question
Used by: Question Manager
Critical: Must match findings to questions accurately, assess confidence.
""",
}


def get_role_context(role: str) -> str:
    """Get the context description for a role."""
    normalized = role.lower().replace("-", "_")
    skill_name = get_skill_name_for_role(normalized)
    return ROLE_CONTEXTS.get(skill_name, f"This role handles: {role}")


# =============================================================================
# SKILL GENERATOR
# =============================================================================

class SkillGenerator:
    """
    Generates SKILL.md files using a teacher model.
    
    When a skill doesn't exist, the generator:
    1. Uses context about the role to understand what's needed
    2. Calls the teacher model (expensive, but one-time cost)
    3. Saves the generated skill for future use
    
    Usage:
        generator = SkillGenerator(llm_client, skill_loader)
        
        # Generate a missing skill
        skill = generator.generate_skill(
            role="finding_extraction",
            example_prompt="Extract findings from: {text}"
        )
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        skill_loader: SkillLoader,
        teacher_model: str = "strong",  # Uses model router tier
        enabled: bool = True,
    ):
        """
        Initialize the SkillGenerator.
        
        Args:
            llm_client: LLM client for calling the teacher model
            skill_loader: SkillLoader instance to save generated skills
            teacher_model: Model tier to use for generation ("strong" recommended)
            enabled: Whether auto-generation is enabled
        """
        self.llm = llm_client
        self.skill_loader = skill_loader
        self.teacher_model = teacher_model
        self.enabled = enabled
        self._generation_in_progress: set = set()  # Prevent concurrent generation
    
    def generate_skill(
        self,
        role: str,
        example_prompt: str = "",
        force: bool = False,
    ) -> Optional[Skill]:
        """
        Generate a skill file for the given role.
        
        Args:
            role: The role to generate a skill for
            example_prompt: An example prompt that uses this role
            force: If True, regenerate even if skill exists
            
        Returns:
            The generated Skill object, or None if generation failed
        """
        if not self.enabled:
            logger.debug("Skill generation is disabled")
            return None
        
        skill_name = get_skill_name_for_role(role)
        
        # Check if already exists (unless forcing)
        if not force and self.skill_loader.skill_exists(skill_name):
            logger.debug(f"Skill '{skill_name}' already exists, skipping generation")
            return self.skill_loader.get_skill(skill_name)
        
        # Prevent concurrent generation of the same skill
        if skill_name in self._generation_in_progress:
            logger.debug(f"Skill '{skill_name}' generation already in progress")
            return None
        
        self._generation_in_progress.add(skill_name)
        
        try:
            logger.info(f"🔧 Generating skill for role '{skill_name}'...")
            
            # Get context about this role
            role_context = get_role_context(skill_name)
            
            # Build the generation prompt
            prompt = SKILL_GENERATION_PROMPT.format(
                role_name=skill_name,
                role_context=role_context,
                example_prompt=example_prompt or "(No example provided)",
            )
            
            # Call the teacher model
            # Note: We use skip_skill_injection=True to avoid recursion!
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="orchestrator",  # Maps to "strong" tier
                system="You are an expert at creating concise, effective skill files for LLMs.",
                skip_skill_injection=True,  # Important: avoid recursion
            )
            
            skill_content = response.content.strip()
            
            # Validate the generated content
            if not self._validate_skill_content(skill_content):
                logger.warning(f"Generated skill content validation failed for '{skill_name}'")
                return None
            
            # Create metadata
            metadata = {
                "version": "1.0.0",
                "auto_generated": True,
                "generated_at": datetime.now().isoformat(),
                "teacher_model": self.teacher_model,
                "description": f"Auto-generated skill for {skill_name}",
            }
            
            # Save the skill
            self.skill_loader.save_skill(
                role=skill_name,
                content=skill_content,
                metadata=metadata,
            )
            
            logger.info(f"✅ Generated and saved skill '{skill_name}'")
            
            # Return the loaded skill
            return self.skill_loader.get_skill(skill_name)
            
        except Exception as e:
            logger.error(f"Failed to generate skill '{skill_name}': {e}")
            return None
            
        finally:
            self._generation_in_progress.discard(skill_name)
    
    def _validate_skill_content(self, content: str) -> bool:
        """
        Validate that generated skill content is reasonable.
        
        Returns:
            True if content looks valid, False otherwise
        """
        if not content:
            return False
        
        # Check minimum length (skills should be substantial)
        if len(content) < 200:
            logger.warning("Generated skill is too short")
            return False
        
        # Check maximum length (skills should be concise)
        if len(content) > 5000:
            logger.warning("Generated skill is too long")
            return False
        
        # Check for markdown structure
        if "#" not in content:
            logger.warning("Generated skill has no headers")
            return False
        
        return True
    
    def get_or_generate_skill(
        self,
        role: str,
        example_prompt: str = "",
    ) -> Optional[Skill]:
        """
        Get an existing skill or generate one if missing.
        
        This is the main entry point for the skill system.
        
        Args:
            role: Role name
            example_prompt: Example prompt (used if generating)
            
        Returns:
            Skill object (loaded or generated), or None if both fail
        """
        skill_name = get_skill_name_for_role(role)
        
        # Try to load existing skill first
        skill = self.skill_loader.get_skill(skill_name)
        
        if skill is not None:
            return skill
        
        # Generate if missing
        return self.generate_skill(role, example_prompt)


# =============================================================================
# SKILL MANAGER (combines loader + generator)
# =============================================================================

class SkillManager:
    """
    High-level manager that combines SkillLoader and SkillGenerator.
    
    This is the main interface for the skill system in INQUIRO.
    
    Usage:
        manager = SkillManager(llm_client, skills_dir="./skills")
        
        # Get skill (loads if exists, generates if not)
        skill = manager.get_skill("finding_extraction", example_prompt="...")
        
        if skill:
            enhanced_system = f"{skill.content}\\n\\n{system_prompt}"
    """
    
    def __init__(
        self,
        llm_client: "LLMClient" = None,
        skills_dir: str = "./skills",
        auto_generate: bool = True,
        enabled: bool = True,
    ):
        """
        Initialize the SkillManager.
        
        Args:
            llm_client: LLM client (required if auto_generate=True)
            skills_dir: Path to skills directory
            auto_generate: Whether to auto-generate missing skills
            enabled: Whether the skill system is enabled
        """
        self.enabled = enabled
        self.auto_generate = auto_generate
        
        self.loader = SkillLoader(
            skills_dir=skills_dir,
            enabled=enabled,
        )
        
        self.generator = SkillGenerator(
            llm_client=llm_client,
            skill_loader=self.loader,
            enabled=auto_generate and llm_client is not None,
        ) if auto_generate and llm_client else None
    
    def get_skill(
        self,
        role: str,
        example_prompt: str = "",
    ) -> Optional[Skill]:
        """
        Get a skill for the given role.
        
        If auto_generate is enabled and skill doesn't exist,
        it will be generated using the teacher model.
        
        Args:
            role: Role name
            example_prompt: Example prompt (used for generation)
            
        Returns:
            Skill object or None
        """
        if not self.enabled:
            return None
        
        # Try loader first
        skill = self.loader.get_skill(role)
        
        if skill is not None:
            return skill
        
        # Try generator if enabled
        if self.generator and self.auto_generate:
            return self.generator.generate_skill(role, example_prompt)
        
        return None
    
    def inject_skill(
        self,
        role: str,
        system_prompt: str,
        example_prompt: str = "",
    ) -> str:
        """
        Inject a skill into a system prompt if available.
        
        Args:
            role: Role name
            system_prompt: Original system prompt
            example_prompt: Example for generation
            
        Returns:
            Enhanced system prompt (or original if no skill)
        """
        skill = self.get_skill(role, example_prompt)
        
        if skill is None:
            return system_prompt
        
        skill_block = self.loader.format_skill_for_injection(skill)
        
        if system_prompt:
            return f"{skill_block}\n\n{system_prompt}"
        return skill_block
    
    def list_skills(self) -> list[str]:
        """List all available skills."""
        return self.loader.list_available_skills()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get skill system statistics."""
        stats = self.loader.get_stats()
        stats["auto_generate_enabled"] = self.auto_generate
        return stats
