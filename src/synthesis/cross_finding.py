"""
Cross-Finding Synthesis Engine.

Analyzes multiple findings to identify overarching themes and patterns,
creating higher-level insights for the Discussion section of reports.
"""

import json
import logging
from typing import List, Dict, Optional, Any

from src.utils.llm_client import LLMClient
from config.prompts.cross_finding import (
    CROSS_FINDING_SYNTHESIS_PROMPT,
    THEME_NARRATIVE_PROMPT,
)

logger = logging.getLogger(__name__)


class CrossFindingSynthesizer:
    """
    Synthesizes individual findings into higher-level themes.
    
    This addresses the "disconnected findings" problem by asking:
    "Given all these discoveries, what are the key patterns and themes?"
    
    Usage:
        synthesizer = CrossFindingSynthesizer(llm_client)
        themes = synthesizer.synthesize(findings, objective)
        # themes = [{"theme_id": "theme_1", "synthesis_claim": "...", ...}]
    """
    
    def __init__(self, llm_client: LLMClient, min_findings: int = 3):
        """
        Initialize the synthesizer.
        
        Args:
            llm_client: LLM client for generation
            min_findings: Minimum findings required for synthesis (default 3)
        """
        self.llm = llm_client
        self.min_findings = min_findings

    def _format_findings_for_prompt(self, findings: List[Dict]) -> str:
        """
        Format findings into readable text for the LLM prompt.
        
        Args:
            findings: List of finding dictionaries
            
        Returns:
            Formatted string with all findings
        """
        lines = []
        for i, f in enumerate(findings, 1):
            finding_id = f.get("id", f.get("finding_id", f"f_{i}"))
            claim = f.get("claim", "No claim")
            confidence = f.get("confidence", 0.0)
            finding_type = f.get("finding_type", f.get("type", "unknown"))
            
            # Source attribution
            source = f.get("source", {})
            if isinstance(source, dict):
                source_str = source.get("paper_title") or source.get("path") or "Unknown source"
            else:
                source_str = str(source) if source else "Unknown source"
            
            # Evidence/context
            evidence = f.get("evidence", f.get("context", ""))
            if len(evidence) > 300:
                evidence = evidence[:300] + "..."
            
            lines.append(f"""### Finding {i}
- **ID:** {finding_id}
- **Type:** {finding_type}
- **Claim:** {claim}
- **Confidence:** {confidence:.2f}
- **Source:** {source_str}
- **Evidence:** {evidence}
""")
        
        return "\n".join(lines)
    
    def _parse_themes_response(self, response_text: str) -> List[Dict]:
        """
        Parse the LLM response into structured themes.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            List of theme dictionaries
        """
        text = response_text.strip()
        
        # Strip markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Try to parse JSON
        try:
            themes = json.loads(text)
            if isinstance(themes, list):
                return themes
            elif isinstance(themes, dict) and "themes" in themes:
                return themes["themes"]
            else:
                logger.warning(f"Unexpected theme response format: {type(themes)}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse themes JSON: {e}")
            logger.debug(f"Response was: {text[:500]}")
            
            # Try to find JSON array in response
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            
            return []

    def synthesize(
        self,
        findings: List[Dict],
        objective: str,
        max_themes: int = 5,
    ) -> List[Dict]:
        """
        Synthesize findings into higher-level themes.
        
        Args:
            findings: List of finding dictionaries from the world model
            objective: The research objective
            max_themes: Maximum number of themes to generate (default 5)
            
        Returns:
            List of theme dictionaries, each containing:
            - theme_id: Unique identifier
            - title: Short theme title
            - synthesis_claim: The synthesized insight
            - supporting_finding_ids: IDs of findings that support this theme
            - evidence_summary: How findings support the theme
            - contradictions: Any conflicting evidence
            - confidence: Overall confidence (0.0-1.0)
            - relevance_to_objective: How theme addresses objective
        """
        if len(findings) < self.min_findings:
            logger.warning(
                f"Only {len(findings)} findings available, need at least "
                f"{self.min_findings} for synthesis. Skipping."
            )
            return []
        
        logger.info(f"🔬 Synthesizing {len(findings)} findings into themes...")
        
        # Format findings for the prompt
        findings_text = self._format_findings_for_prompt(findings)
        
        # Build the synthesis prompt
        prompt = CROSS_FINDING_SYNTHESIS_PROMPT.format(
            objective=objective,
            num_findings=len(findings),
            findings_text=findings_text,
        )
        
        try:
            # Use strong model for synthesis (this is high-level reasoning)
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="report_writing",  # Uses strong model
                system="You are a senior research scientist synthesizing findings.",
                max_tokens=4096,
            )
            
            text = response.content if hasattr(response, "content") else str(response)
            themes = self._parse_themes_response(text)
            
            # Validate and clean themes
            validated_themes = []
            for theme in themes[:max_themes]:
                if self._validate_theme(theme, findings):
                    validated_themes.append(theme)
                else:
                    logger.warning(f"Theme validation failed: {theme.get('title', 'unknown')}")
            
            logger.info(f"✅ Generated {len(validated_themes)} synthesis themes")
            return validated_themes
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return []
    
    def _validate_theme(self, theme: Dict, findings: List[Dict]) -> bool:
        """
        Validate that a theme has required fields and valid references.
        
        Args:
            theme: Theme dictionary to validate
            findings: Original findings (to validate IDs)
            
        Returns:
            True if theme is valid
        """
        required_fields = ["theme_id", "synthesis_claim", "supporting_finding_ids"]
        
        for field in required_fields:
            if field not in theme:
                logger.warning(f"Theme missing required field: {field}")
                return False
        
        # Verify supporting finding IDs exist
        finding_ids = {
            f.get("id", f.get("finding_id", f"f_{i}"))
            for i, f in enumerate(findings, 1)
        }
        
        supporting = theme.get("supporting_finding_ids", [])
        if not supporting or len(supporting) < 1:
            logger.warning("Theme has no supporting findings")
            return False
        
        # At least one supporting ID should be valid
        # (LLM might generate slightly different IDs)
        valid_refs = sum(1 for sid in supporting if sid in finding_ids)
        if valid_refs == 0:
            # Try fuzzy matching (LLM might prefix differently)
            for sid in supporting:
                for fid in finding_ids:
                    if sid in fid or fid in sid:
                        valid_refs += 1
                        break
        
        if valid_refs == 0:
            logger.warning(f"No valid finding references in theme: {supporting}")
            # Don't fail — LLM synthesis is still valuable
        
        return True

    def generate_theme_narrative(
        self,
        theme: Dict,
        findings: List[Dict],
        objective: str,
    ) -> str:
        """
        Generate a prose narrative for a single theme.
        
        This is used in report generation to create the Discussion section.
        
        Args:
            theme: Theme dictionary from synthesize()
            findings: All findings (to look up supporting ones)
            objective: Research objective
            
        Returns:
            Prose narrative suitable for Discussion section
        """
        # Build supporting findings text
        supporting_ids = set(theme.get("supporting_finding_ids", []))
        supporting_findings = []
        
        for f in findings:
            fid = f.get("id", f.get("finding_id", ""))
            if fid in supporting_ids or any(fid in sid or sid in fid for sid in supporting_ids):
                supporting_findings.append(f)
        
        if not supporting_findings:
            # Fall back to all findings if ID matching failed
            supporting_findings = findings[:5]
        
        supporting_text = self._format_findings_for_prompt(supporting_findings)
        
        # Build narrative prompt
        prompt = THEME_NARRATIVE_PROMPT.format(
            objective=objective,
            theme_title=theme.get("title", "Untitled Theme"),
            synthesis_claim=theme.get("synthesis_claim", ""),
            supporting_findings_text=supporting_text,
            contradictions=theme.get("contradictions") or "None identified",
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="report_writing",
                system="You are a scientific writer crafting a Discussion section.",
                max_tokens=2048,
            )
            
            text = response.content if hasattr(response, "content") else str(response)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Theme narrative generation failed: {e}")
            # Return a basic fallback
            return f"{theme.get('synthesis_claim', 'No synthesis available.')}"
    
    def synthesize_and_narrate(
        self,
        findings: List[Dict],
        objective: str,
        max_themes: int = 5,
    ) -> Dict[str, Any]:
        """
        Complete synthesis pipeline: identify themes and generate narratives.
        
        This is the main entry point for report generation.
        
        Args:
            findings: List of finding dictionaries
            objective: Research objective
            max_themes: Maximum themes to generate
            
        Returns:
            Dictionary with:
            - themes: List of theme dictionaries
            - narratives: Dict mapping theme_id -> prose narrative
            - discussion_text: Combined Discussion section text
        """
        # Step 1: Identify themes
        themes = self.synthesize(findings, objective, max_themes)
        
        if not themes:
            return {
                "themes": [],
                "narratives": {},
                "discussion_text": "Insufficient findings for synthesis.",
            }
        
        # Step 2: Generate narrative for each theme
        narratives = {}
        for theme in themes:
            theme_id = theme.get("theme_id", "unknown")
            narrative = self.generate_theme_narrative(theme, findings, objective)
            narratives[theme_id] = narrative
        
        # Step 3: Combine into Discussion section
        discussion_parts = []
        for theme in themes:
            theme_id = theme.get("theme_id", "unknown")
            title = theme.get("title", "Theme")
            narrative = narratives.get(theme_id, "")
            
            discussion_parts.append(f"### {title}\n\n{narrative}")
        
        discussion_text = "\n\n".join(discussion_parts)
        
        logger.info(f"✅ Generated Discussion with {len(themes)} theme sections")
        
        return {
            "themes": themes,
            "narratives": narratives,
            "discussion_text": discussion_text,
        }
