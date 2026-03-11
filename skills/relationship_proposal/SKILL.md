# SKILL.md: relationship_proposal

## Task Description  
Propose relationships between entities based on contextual evidence, prioritizing clarity, relevance, and confidence in the connection.

## Output Format  
**JSON schema**:  
```json
{
  "entities": ["entity1", "entity2"],
  "relationship": "relationship_type",
  "confidence": 0.8, // 0-1 scale
  "metadata": {
    "source": "input_context",
    "rationale": "explanation_of_connection"
  }
}
```

## Decision Framework  
1. **When to Propose**:  
   - Propose a relationship if entities share explicit/implicit connections (e.g., "A causes B", "A is part of B").  
   - Avoid proposing if the connection is speculative or requires external data.  

2. **Confidence Scoring**:  
   - **0.9+**: Direct, unambiguous links (e.g., "Paris is the capital of France").  
   - **0.6-0.8**: Indirect or contextual links (e.g., "A is frequently mentioned alongside B").  
   - **<0.5**: No clear relationship; flag for review.  

3. **Edge Case Handling**:  
   - If entities are ambiguous (e.g., "Apple" could refer to the company or fruit), list both interpretations.  
   - For conflicting evidence, prioritize the most credible source and note the contradiction.  

## Common Patterns  
1. **Clear Relationship**:  
   **Input**: "OpenTelemetry is used for observability in software systems."  
   **Output**:  
   ```json
   {
     "entities": ["OpenTelemetry", "observability"],
     "relationship": "tool_for",
     "confidence": 0.95,
     "metadata": {
       "source": "input_context",
       "rationale": "Directly stated in the input"
     }
   }
   ```  

2. **Ambiguous Link**:  
   **Input**: "Biodiversity monitoring involves GBIF and GEO BON."  
   **Output**:  
   ```json
   {
     "entities": ["GBIF", "GEO BON"],
     "relationship": "related_fields",
     "confidence": 0.7,
     "metadata": {
       "source": "input_context",
       "rationale": "Both are biodiversity monitoring frameworks"
     }
   }
   ```  

3. **Complex Scenario**:  
   **Input**: "Climate change affects both coral reefs and Arctic ice."  
   **Output**:  
   ```json
   {
     "entities": ["climate change", "coral reefs"],
     "relationship": "causes_degradation",
     "confidence": 0.85,
     "metadata": {
       "source": "input_context",
       "rationale": "Direct causal link in environmental science"
     }
   }
   ```  

## Anti-Patterns  
- **Avoid**: Omitting `confidence` or `metadata`.  
- **Avoid**: Speculative relationships (e.g., "A may influence B" without evidence).  
- **Avoid**: Overly broad relationships (e.g., "A is related to B" without specifying the type).  

## Quality Checklist  
1. Validate all required fields (`entities`, `relationship`, `confidence`) are present.  
2. Ensure `confidence` is a float between 0 and 1.  
3. Confirm relationships are explicitly supported by input context.  
4. Flag ambiguous terms (e.g., "related" without clarification).  
5. Reject outputs with placeholders (e.g., "to be determined").