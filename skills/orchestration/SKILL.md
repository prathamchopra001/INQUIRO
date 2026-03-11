# SKILL.md: Orchestration

## Task Description  
Decompose complex objectives into specific, answerable questions that align with the research domain and methodology. Ensure questions are actionable, focused, and directly address the core objective.

## Output Format  
```json
{
  "questions": ["Specific question 1", "Specific question 2", ...],
  "rationale": "Explanation of how the questions address the objective"
}
```  
- **questions**: Array of 5-10 answerable questions starting with "What", "How", "Why", "Which", or "To what extent".  
- **rationale**: 1-2 sentences explaining the decomposition logic and alignment with the objective.

## Decision Framework  
1. **Identify Core Objective**: Extract the main goal from the input.  
2. **Break into Sub-Questions**: Use "What", "How", "Why" to split the objective into smaller parts.  
3. **Ensure Specificity**: Avoid vague terms; specify variables, methods, or contexts.  
4. **Confidence Scoring**:  
   - Score 1-5: 5 = precise and answerable, 1 = too broad or unclear.  
   - Prioritize questions with higher scores.  
5. **Edge Cases**:  
   - If the objective is ambiguous, ask for clarification.  
   - For domain-specific terms, define them in the rationale.  

## Common Patterns  
**Example 1**  
Input: *"What are the key factors affecting Q-learning convergence?"*  
Output:  
```json
{
  "questions": [
    "How do hyperparameters influence Q-learning convergence rates?",
    "What role does reward function design play in Q-learning stability?",
    "Which environments exhibit the fastest Q-learning convergence?"
  ],
  "rationale": "These questions focus on hyperparameters, reward functions, and environment-specific factors, which are actionable and directly tied to the objective."
}
```  

**Example 2**  
Input: *"How can we improve patient adherence to treatment plans?"*  
Output:  
```json
{
  "questions": [
    "What barriers do patients face when adhering to treatment plans?",
    "How does telehealth impact medication adherence in chronic disease management?",
    "Which interventions show the highest adherence rates in clinical trials?"
  ],
  "rationale": "The questions address barriers, telehealth impact, and effective interventions, providing a structured approach to improving adherence."
}
```  

## Anti-Patterns  
- **Too Broad**: *"What is Q-learning?"* (Not answerable via data analysis).  
- **Unfocused**: *"How can we improve everything?"* (Lacks specificity).  
- **Irrelevant Questions**: *"What is the meaning of life?"* (Off-topic).  
- **Missing Rationale**: Omitting the explanation for decomposition logic.  

## Quality Checklist  
1. Validate JSON structure matches the schema.  
2. Ensure all questions are answerable via literature or data.  
3. Confirm questions are specific, not vague or overly broad.  
4. Verify the rationale explains how the questions align with the objective.  
5. Check for domain-specific terminology and ensure it’s clearly addressed.