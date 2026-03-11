# -*- coding: utf-8 -*-
"""
Domain Skill Injector for INQUIRO.

Provides automatic domain skill injection into LLM prompts based on
the research objective. Integrates with the existing skill system.

Usage:
    injector = DomainSkillInjector()
    
    # Detect domain and cache skill
    injector.set_objective("Impact of interest rates on housing prices")
    
    # Inject skill into any prompt
    enhanced_prompt = injector.inject(original_prompt)
    
    # Or get skill info directly
    print(f"Domain: {injector.current_domain}")
    print(f"Skill: {injector.current_skill[:100]}...")
"""

import logging
from typing import Optional, Tuple

from config.prompts.domain_skills import (
    detect_domain,
    get_domain_skill,
    get_domain_skill_for_objective,
    DOMAIN_KEYWORDS,
)

logger = logging.getLogger(__name__)


class DomainSkillInjector:
    """
    Manages domain skill injection for INQUIRO prompts.
    
    Features:
    - Automatic domain detection from research objective
    - Skill caching to avoid repeated detection
    - Configurable injection position (start, end, section)
    - Graceful fallback to general skill
    """
    
    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: Whether skill injection is active
        """
        self.enabled = enabled
        self._objective: Optional[str] = None
        self._domain: Optional[str] = None
        self._skill: Optional[str] = None
    
    def set_objective(self, objective: str) -> str:
        """
        Set the research objective and detect domain.
        
        Args:
            objective: Research objective text
            
        Returns:
            Detected domain name
        """
        self._objective = objective
        self._domain = detect_domain(objective)
        self._skill = get_domain_skill(self._domain)
        
        logger.info(f"Domain detected: {self._domain} ({len(self._skill.split())} word skill)")
        
        return self._domain
    
    @property
    def current_domain(self) -> Optional[str]:
        """Get the currently detected domain."""
        return self._domain
    
    @property
    def current_skill(self) -> Optional[str]:
        """Get the current domain skill prompt."""
        return self._skill
    
    def inject(
        self,
        prompt: str,
        position: str = "start",
        section_header: str = "## Domain Expertise",
    ) -> str:
        """
        Inject domain skill into a prompt.
        
        Args:
            prompt: Original prompt text
            position: Where to inject - "start", "end", or "section"
            section_header: Header for section mode
            
        Returns:
            Enhanced prompt with skill injected
        """
        if not self.enabled or not self._skill:
            return prompt
        
        if position == "start":
            return f"{self._skill}\n\n{prompt}"
        elif position == "end":
            return f"{prompt}\n\n{self._skill}"
        elif position == "section":
            return f"{prompt}\n\n{section_header}\n{self._skill}"
        else:
            return f"{self._skill}\n\n{prompt}"
    
    def inject_for_role(
        self,
        prompt: str,
        role: str,
    ) -> str:
        """
        Inject domain skill with role-specific formatting.
        
        Args:
            prompt: Original prompt
            role: LLM role (e.g., "finding_extraction", "query_formulation")
            
        Returns:
            Enhanced prompt
        """
        if not self.enabled or not self._skill:
            return prompt
        
        # Role-specific injection strategies
        role_strategies = {
            "finding_extraction": "start",
            "query_formulation": "start",
            "task_generation": "start",
            "paper_ranking": "end",
            "report_writing": "section",
            "scoring": "end",
        }
        
        position = role_strategies.get(role, "start")
        return self.inject(prompt, position=position)
    
    def get_domain_context(self) -> str:
        """
        Get a brief domain context string for prompts.
        
        Returns:
            Short domain context (e.g., "Research domain: economics")
        """
        if self._domain and self._domain != "general":
            return f"Research domain: {self._domain.replace('_', ' ').title()}"
        return ""


# Global instance for easy access
_global_injector: Optional[DomainSkillInjector] = None


def get_domain_skill_injector() -> DomainSkillInjector:
    """Get or create the global domain skill injector."""
    global _global_injector
    if _global_injector is None:
        _global_injector = DomainSkillInjector()
    return _global_injector


def inject_domain_skill(prompt: str, objective: str = None) -> str:
    """
    Convenience function to inject domain skill into a prompt.
    
    Args:
        prompt: Original prompt
        objective: Research objective (optional, uses cached if not provided)
        
    Returns:
        Enhanced prompt with domain skill
    """
    injector = get_domain_skill_injector()
    
    if objective:
        injector.set_objective(objective)
    
    return injector.inject(prompt)
