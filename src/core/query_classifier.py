"""
Query Classifier for INQUIRO

Uses the LLM to intelligently classify queries:
1. Simple factual questions → LLM answers directly
2. Math expressions → compute directly  
3. Research objectives → proceed to full pipeline

No hardcoding - leverages the LLM's knowledge.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Classification of user queries."""
    CALCULATION = "calculation"      # Math expression to compute
    SIMPLE_QUESTION = "simple"       # Simple factual question - LLM can answer
    RESEARCH = "research"            # Genuine research objective


@dataclass
class ClassificationResult:
    """Result of query classification."""
    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    direct_answer: Optional[str] = None
    reasoning: str = ""


class QueryClassifier:
    """
    Classifies user input using LLM to determine if it needs research.
    
    Simple questions get answered directly by the LLM.
    Only genuine research objectives proceed to the full pipeline.
    """
    
    # Patterns for math expressions
    MATH_OPERATORS = r'[\+\-\*\/\^\%]'
    MATH_EXPRESSION_PATTERN = re.compile(r'^[\s\d\+\-\*\/\^\%\(\)\.\,xX]+$')
    
    CLASSIFICATION_PROMPT = """You are a query classifier for INQUIRO, an autonomous research system.

Classify this query into ONE of these categories:

1. SIMPLE - A simple factual question that can be answered from general knowledge in 1-3 sentences.
   Examples: "What is the SI unit of energy?", "Who wrote Romeo and Juliet?", "What is photosynthesis?"

2. RESEARCH - A genuine research objective that requires literature review, data analysis, or in-depth investigation.
   Examples: "Investigate factors affecting hospital readmission", "Analyze the relationship between social media and mental health", "Review machine learning approaches for protein folding"

Query: "{query}"

Respond with ONLY a JSON object:
{{"type": "SIMPLE" or "RESEARCH", "reason": "brief explanation"}}

If SIMPLE, also include "answer": "your direct answer to the question"
"""

    def __init__(self, llm_client=None):
        """
        Initialize classifier.
        
        Args:
            llm_client: Optional LLMClient instance. If not provided, will import when needed.
        """
        self._llm_client = llm_client
    
    def _get_llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            from src.utils.llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a user query using the LLM.
        
        Args:
            query: The user's input string
            
        Returns:
            ClassificationResult with type, confidence, and optional direct answer
        """
        query = query.strip()
        
        if not query:
            return ClassificationResult(
                query_type=QueryType.RESEARCH,
                confidence=0.0,
                reasoning="Empty input"
            )
        
        # Check for math expressions first (no LLM needed)
        calc_result = self._check_calculation(query)
        if calc_result:
            return calc_result
        
        # Check for "what is X" style calculations
        what_is_calc = self._check_what_is_calculation(query)
        if what_is_calc:
            return what_is_calc
        
        # Use LLM to classify and potentially answer
        return self._classify_with_llm(query)
    
    def _classify_with_llm(self, query: str) -> ClassificationResult:
        """Use LLM to classify query and answer if simple."""
        try:
            llm = self._get_llm_client()
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)
            
            response = llm.complete_for_role(
                prompt=prompt,
                role="query_classification",
                temperature=0.1
            )
            
            # Extract text content from LLMResponse object
            text = response.content if hasattr(response, 'content') else str(response)
            text = text.strip()
            
            # Parse the JSON response
            import json
            
            # Clean up response - extract JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            # Find JSON object
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                text = text[start:end]
            
            result = json.loads(text)
            
            query_type = result.get("type", "RESEARCH").upper()
            
            if query_type == "SIMPLE" and "answer" in result:
                return ClassificationResult(
                    query_type=QueryType.SIMPLE_QUESTION,
                    confidence=0.95,
                    direct_answer=result["answer"],
                    reasoning=result.get("reason", "LLM classified as simple question")
                )
            else:
                return ClassificationResult(
                    query_type=QueryType.RESEARCH,
                    confidence=0.9,
                    reasoning=result.get("reason", "LLM classified as research objective")
                )
                
        except Exception as e:
            # Safe logging for Windows console encoding issues
            try:
                logger.warning(f"LLM classification failed: {e}, defaulting to research")
            except UnicodeEncodeError:
                logger.warning("LLM classification failed (encoding error), defaulting to research")
            return ClassificationResult(
                query_type=QueryType.RESEARCH,
                confidence=0.5,
                reasoning=f"Classification failed: {e}"
            )
    
    def _check_calculation(self, query: str) -> Optional[ClassificationResult]:
        """Check if the query is a math expression and compute it."""
        normalized = query.replace(' ', '').replace(',', '')
        normalized = re.sub(r'[xX](?=\d)', '*', normalized)
        normalized = re.sub(r'(?<=\d)[xX]', '*', normalized)
        
        if not self.MATH_EXPRESSION_PATTERN.match(normalized):
            return None
        
        if not re.search(self.MATH_OPERATORS, normalized):
            return None
        
        if not re.search(r'\d', normalized):
            return None
        
        try:
            safe_expr = normalized.replace('^', '**')
            if not re.match(r'^[\d\+\-\*\/\%\.\(\)\s]+$', safe_expr):
                return None
            
            result = eval(safe_expr, {"__builtins__": {}}, {})
            
            if isinstance(result, float):
                if result == int(result):
                    result_str = str(int(result))
                else:
                    result_str = f"{result:.10g}"
            else:
                result_str = str(result)
            
            return ClassificationResult(
                query_type=QueryType.CALCULATION,
                confidence=0.99,
                direct_answer=f"{query} = {result_str}",
                reasoning="Math expression computed"
            )
            
        except Exception as e:
            logger.debug(f"Math evaluation failed: {e}")
            return None
    
    def _check_what_is_calculation(self, query: str) -> Optional[ClassificationResult]:
        """Check for 'what is X' style calculations."""
        query_lower = query.lower().strip()
        
        patterns = [
            r"what(?:'s| is| are)\s+(.+?)[\?\s]*$",
            r"how much is\s+(.+?)[\?\s]*$",
            r"calculate\s+(.+?)[\?\s]*$",
            r"compute\s+(.+?)[\?\s]*$",
            r"solve\s+(.+?)[\?\s]*$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                potential_math = match.group(1).strip()
                calc_result = self._check_calculation(potential_math)
                if calc_result:
                    calc_result.direct_answer = f"{query.strip('?')} = {calc_result.direct_answer.split('=')[1].strip()}"
                    return calc_result
        
        return None


def check_before_research(objective: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to check if an objective should proceed to research.
    
    Returns:
        (should_proceed, message)
    """
    classifier = QueryClassifier()
    result = classifier.classify(objective)
    
    if result.query_type == QueryType.RESEARCH:
        return True, None
    
    return False, result.direct_answer


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("Query Classifier Test (LLM-powered)")
    print("=" * 60)
    
    classifier = QueryClassifier()
    
    test_cases = [
        "2+2",
        "what is 5*5",
        "what is the SI unit of energy",
        "who wrote hamlet",
        "Investigate factors affecting hospital readmission",
        "research the impact of climate change on biodiversity",
    ]
    
    for query in test_cases:
        print(f"\nQuery: {query}")
        result = classifier.classify(query)
        print(f"  Type: {result.query_type.value}")
        print(f"  Confidence: {result.confidence:.0%}")
        if result.direct_answer:
            print(f"  Answer: {result.direct_answer}")
        print(f"  Reasoning: {result.reasoning}")
