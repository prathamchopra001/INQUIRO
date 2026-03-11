"""
Domain Anchoring Module for Literature Search Agent

This module provides domain anchor extraction and query validation
to ensure literature searches stay on-topic.

INTEGRATION: Import these functions in src/agents/literature.py
and call them in the _generate_queries() method.

Usage:
    from src.literature.domain_anchoring import DomainAnchorExtractor
    
    extractor = DomainAnchorExtractor(llm_client)
    anchors = extractor.extract_anchors(objective)
    # anchors = {
    #     "primary_domain": "ecology",
    #     "anchor_terms": ["biodiversity", "ecological monitoring", ...],
    #     "avoid_terms": ["framework", "architecture", ...]
    # }
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DomainAnchorExtractor:
    """
    Extracts domain-specific anchor terms from research objectives.
    
    These anchors are used to ensure literature search queries
    stay relevant to the target domain instead of drifting
    to related but off-topic fields.
    """
    
    def __init__(self, llm_client, cache_anchors: bool = True):
        """
        Args:
            llm_client: The LLM client for anchor extraction
            cache_anchors: Whether to cache anchors per objective (default True)
        """
        self.llm_client = llm_client
        self.cache_anchors = cache_anchors
        self._anchor_cache: Dict[str, dict] = {}
    
    def extract_anchors(self, objective: str) -> dict:
        """
        Extract domain anchor terms from a research objective.
        
        Args:
            objective: The research objective string
            
        Returns:
            dict with keys:
                - primary_domain: str
                - anchor_terms: List[str]
                - cross_domain_terms: List[str]
                - avoid_terms: List[str]
        """
        # Check cache first
        cache_key = objective[:200]  # Use first 200 chars as key
        if self.cache_anchors and cache_key in self._anchor_cache:
            logger.debug("Using cached domain anchors")
            return self._anchor_cache[cache_key]
        
        # Try LLM extraction
        try:
            anchors = self._extract_via_llm(objective)
            if self._validate_anchors(anchors):
                if self.cache_anchors:
                    self._anchor_cache[cache_key] = anchors
                return anchors
        except Exception as e:
            logger.warning(f"LLM anchor extraction failed: {e}")
        
        # Fallback to heuristic extraction
        anchors = self._extract_heuristic(objective)
        if self.cache_anchors:
            self._anchor_cache[cache_key] = anchors
        return anchors
    
    def _extract_via_llm(self, objective: str) -> dict:
        """Use LLM to extract domain anchors."""
        from config.prompts.domain_anchors import DOMAIN_ANCHOR_EXTRACTION_PROMPT
        
        # Check if using local model (needs simpler prompt)
        use_local = False
        if hasattr(self.llm_client, 'is_local_tier_for_role'):
            use_local = self.llm_client.is_local_tier_for_role("query_generation")
        
        if use_local:
            # Use simpler prompt for local models
            from config.prompts.lit_agent_local import CORE_TERM_EXTRACTION_PROMPT_SIMPLE
            prompt = CORE_TERM_EXTRACTION_PROMPT_SIMPLE.format(objective=objective)
        else:
            prompt = DOMAIN_ANCHOR_EXTRACTION_PROMPT.format(objective=objective)
        
        # Use complete_for_role if available (LLMClient), otherwise fall back to complete
        if hasattr(self.llm_client, 'complete_for_role'):
            response = self.llm_client.complete_for_role(
                prompt=prompt,
                role="query_generation",  # Use fast tier for this
                max_tokens=500
            )
            text = response.content if hasattr(response, 'content') else str(response)
        else:
            text = self.llm_client.complete(prompt, role="query_generation")
        
        # Parse JSON response
        if use_local:
            # Local model may return just an array of terms
            anchors = self._parse_local_response(text, objective)
        else:
            anchors = self._parse_json_response(text)
        
        return anchors
    
    def _parse_local_response(self, text: str, objective: str) -> dict:
        """
        Parse response from local model which may be simpler format.
        
        Local models often return just an array of terms instead of full dict.
        """
        # Try to find JSON array
        if '[' in text and ']' in text:
            start = text.find('[')
            end = text.rfind(']') + 1
            try:
                terms = json.loads(text[start:end])
                if isinstance(terms, list):
                    # Convert simple array to full anchors dict
                    return {
                        "primary_domain": "general",
                        "anchor_terms": [str(t).strip() for t in terms if t][:8],
                        "cross_domain_terms": [],
                        "avoid_terms": ["framework", "architecture", "system", "analysis"]
                    }
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object
        if '{' in text and '}' in text:
            try:
                return self._parse_json_response(text)
            except ValueError:
                pass
        
        # Fallback: extract quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return {
                "primary_domain": "general",
                "anchor_terms": [q.strip() for q in quoted if len(q) > 2][:8],
                "cross_domain_terms": [],
                "avoid_terms": ["framework", "architecture", "system"]
            }
        
        # Ultimate fallback: use heuristic
        logger.warning("Local model response unparseable, using heuristic")
        return self._extract_heuristic(objective)
    
    def _extract_heuristic(self, objective: str) -> dict:
        """
        Fallback heuristic extraction when LLM fails.
        
        Extracts nouns and noun phrases, filtering out common
        generic terms.
        """
        logger.info("Using heuristic domain anchor extraction")
        
        # Common domain indicators and their associated terms
        domain_indicators = {
            "ecology": ["biodiversity", "ecosystem", "species", "habitat", "wildlife", "ecological"],
            "health": ["disease", "clinical", "patient", "medical", "epidemiological", "health"],
            "economics": ["market", "price", "economic", "fiscal", "monetary", "trade"],
            "climate": ["climate", "carbon", "temperature", "atmospheric", "weather"],
            "agriculture": ["crop", "agricultural", "farm", "soil", "irrigation", "livestock"],
        }
        
        # Generic terms to avoid
        generic_terms = [
            "framework", "architecture", "system", "model", "approach",
            "method", "analysis", "study", "research", "data", "information",
            "process", "technique", "tool", "platform", "solution"
        ]
        
        objective_lower = objective.lower()
        
        # Detect primary domain
        primary_domain = "general"
        max_matches = 0
        anchor_terms = []
        
        for domain, indicators in domain_indicators.items():
            matches = sum(1 for ind in indicators if ind in objective_lower)
            if matches > max_matches:
                max_matches = matches
                primary_domain = domain
                anchor_terms = [ind for ind in indicators if ind in objective_lower]
        
        # Extract capitalized terms (likely proper nouns / frameworks)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', objective)
        anchor_terms.extend([pn.lower() for pn in proper_nouns if len(pn) > 3])
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', objective)
        anchor_terms.extend([q.lower() for q in quoted])
        
        # Deduplicate and limit
        anchor_terms = list(dict.fromkeys(anchor_terms))[:8]
        
        # If we found very few, add some from the objective
        if len(anchor_terms) < 3:
            words = objective_lower.split()
            for word in words:
                if len(word) > 6 and word not in generic_terms and word not in anchor_terms:
                    anchor_terms.append(word)
                    if len(anchor_terms) >= 5:
                        break
        
        return {
            "primary_domain": primary_domain,
            "anchor_terms": anchor_terms[:6],
            "cross_domain_terms": [],
            "avoid_terms": generic_terms[:10]
        }
    
    def _validate_anchors(self, anchors: dict) -> bool:
        """Validate that extracted anchors are usable."""
        if not isinstance(anchors, dict):
            return False
        if "anchor_terms" not in anchors:
            return False
        if not isinstance(anchors.get("anchor_terms"), list):
            return False
        if len(anchors.get("anchor_terms", [])) < 2:
            return False
        return True
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling common issues."""
        # Strip markdown code blocks
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try finding JSON object in text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON from response: {text[:200]}")
    
    def format_anchoring_instruction(self, anchors: dict) -> str:
        """
        Format domain anchors into an instruction string for the query prompt.
        
        Args:
            anchors: The anchors dict from extract_anchors()
            
        Returns:
            Formatted instruction string to insert into QUERY_GENERATION_PROMPT
        """
        from config.prompts.domain_anchors import QUERY_ANCHORING_INSTRUCTION
        
        anchor_list = ", ".join(f'"{t}"' for t in anchors.get("anchor_terms", []))
        avoid_list = ", ".join(f'"{t}"' for t in anchors.get("avoid_terms", [])[:5])
        
        return QUERY_ANCHORING_INSTRUCTION.format(
            primary_domain=anchors.get("primary_domain", "general"),
            anchor_terms=anchor_list,
            avoid_terms=avoid_list
        )


class QueryValidator:
    """
    Validates search queries for domain relevance before execution.
    
    This catches off-topic queries before they waste API calls
    and pollute the RAG system with irrelevant papers.
    """
    
    def __init__(self, anchors: dict, strict_mode: bool = False):
        """
        Args:
            anchors: Domain anchors from DomainAnchorExtractor
            strict_mode: If True, reject queries without domain terms
        """
        self.anchors = anchors
        self.strict_mode = strict_mode
        self.anchor_terms = set(t.lower() for t in anchors.get("anchor_terms", []))
        self.avoid_terms = set(t.lower() for t in anchors.get("avoid_terms", []))
    
    def validate_query(self, query: str) -> Tuple[bool, float, str]:
        """
        Validate a single query for domain relevance.
        
        Args:
            query: The search query string
            
        Returns:
            Tuple of (is_valid, relevance_score, reason)
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Check for domain anchor presence
        anchor_matches = self.anchor_terms & query_words
        has_anchor = len(anchor_matches) > 0
        
        # Check for avoid term dominance
        avoid_matches = self.avoid_terms & query_words
        avoid_ratio = len(avoid_matches) / max(len(query_words), 1)
        
        # Calculate relevance score
        score = 0.0
        
        # Anchor presence: +0.4 per anchor (max 0.8)
        score += min(len(anchor_matches) * 0.4, 0.8)
        
        # Penalize avoid term dominance
        if avoid_ratio > 0.5:
            score -= 0.3
        
        # Bonus for longer, more specific queries
        if len(query_words) >= 4:
            score += 0.1
        
        # Bonus for quoted phrases (exact match intent)
        if '"' in query:
            score += 0.1
        
        score = max(0.0, min(1.0, score))
        
        # Determine validity
        if self.strict_mode:
            is_valid = has_anchor and score >= 0.4
        else:
            is_valid = score >= 0.3 or has_anchor
        
        # Generate reason
        if not has_anchor:
            reason = f"Query lacks domain anchors. Consider adding: {list(self.anchor_terms)[:3]}"
        elif avoid_ratio > 0.5:
            reason = f"Query dominated by generic terms: {list(avoid_matches)}"
        elif score < 0.4:
            reason = "Query may be too generic for the target domain"
        else:
            reason = "Query appears domain-relevant"
        
        return is_valid, score, reason
    
    def validate_queries(self, queries: List[str]) -> List[dict]:
        """
        Validate multiple queries, returning results for each.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of validation result dicts
        """
        results = []
        for query in queries:
            is_valid, score, reason = self.validate_query(query)
            results.append({
                "query": query,
                "is_valid": is_valid,
                "relevance_score": score,
                "reason": reason
            })
        return results
    
    def filter_queries(self, queries: List[str], min_score: float = 0.3) -> List[str]:
        """
        Filter queries to only keep domain-relevant ones.
        
        Args:
            queries: List of query strings
            min_score: Minimum relevance score to keep (default 0.3)
            
        Returns:
            Filtered list of queries
        """
        results = self.validate_queries(queries)
        
        kept = []
        filtered = []
        
        for r in results:
            if r["relevance_score"] >= min_score:
                kept.append(r["query"])
            else:
                filtered.append(r["query"])
        
        if filtered:
            logger.info(f"Filtered {len(filtered)} off-topic queries: {filtered}")
        
        return kept


def improve_query_with_anchors(query: str, anchors: dict) -> str:
    """
    Improve a query by adding domain anchors if missing.
    
    Args:
        query: Original query string
        anchors: Domain anchors dict
        
    Returns:
        Improved query with domain context
    """
    query_lower = query.lower()
    anchor_terms = anchors.get("anchor_terms", [])
    
    # Check if query already has an anchor
    for anchor in anchor_terms:
        if anchor.lower() in query_lower:
            return query  # Already anchored
    
    # Add the most relevant anchor
    if anchor_terms:
        # Pick shortest anchor to minimize query length impact
        best_anchor = min(anchor_terms, key=len)
        return f"{best_anchor} {query}"
    
    return query
