# SKILL.md: gap_analysis

## Task Description  
Identifies gaps between research objectives and completed work by comparing stated goals against findings and tasks, categorizing answered questions, open questions, and weak areas.

## Output Format  
**JSON Schema**:  
```json
{
  "answered_questions": ["specific question 1", "specific question 2"],
  "open_questions": ["specific question 3", "specific question 4"],
  "weak_areas": ["area 1", "area 2"]
}
```  
All values are arrays of strings. Use exact phrasing from input.

## Decision Framework  
1. **Parse Objective**: Break down the objective into explicit research questions (e.g., "What are the key metrics for biodiversity monitoring?").  
2. **Cross-Reference Findings**: Match completed tasks/findings to answered questions (e.g., "GBIF data integration" → answers "How can GBIF data be leveraged?").  
3. **Identify Open Questions**: Flag questions not addressed by findings (e.g., "What are the ethical implications of real-time surveillance?").  
4. **Highlight Weak Areas**: Note gaps in methodology, data, or scope (e.g., "Lack of field testing for OpenTelemetry deployment").  
5. **Confidence Scoring**: Rate each question's confidence (0–100) based on evidence. Only include questions with ≥70 confidence in "answered_questions".  
6. **Edge Cases**:  
   - Ambiguous objectives → classify as weak areas.  
   - Incomplete data → flag as weak areas, not open questions.  

## Common Patterns  
### ✅ Example 1  
**Input**:  
Objective: "Investigate how observability engineering principles... can be applied to frontline ecological incident monitoring..."  
**Output**:  
```json
{
  "answered_questions": ["What are the key metrics for biodiversity monitoring?"],
  "open_questions": ["How do OpenTelemetry tools handle real-time ecological data streaming?"],
  "weak_areas": ["Lack of case studies on biodiversity surveillance"]
}
```  
### ✅ Example 2  
**Input**:  
Objective: "Evaluate renewable energy adoption in rural communities."  
**Output**:  
```json
{
  "answered_questions": ["What are the cost barriers for solar adoption in rural areas?"],
  "open_questions": ["How do policy incentives vary across regions?"],
  "weak_areas": ["Insufficient data on long-term maintenance costs"]
}
```  

## Anti-Patterns  
- ❌ Rephrasing completed tasks as open questions (e.g., "GBIF data integration" → "What is GBIF data integration?").  
- ❌ Using vague entries like "Need more research" or "Unknown".  
- ❌ Speculative questions not supported by objective.  
- ❌ Duplicating entries across categories.  
- ❌ Including tasks as open questions if they’re just uncompleted.  

## Quality Checklist  
1. Validate all answered questions are explicitly addressed in findings/tasks.  
2. Ensure open questions are *genuinely unanswered*, not just unexplored.  
3. Flag weak areas for insufficient data/methodology, not missing tasks.  
4. Confirm JSON structure matches schema (no extra fields).  
5. Remove duplicates and ensure phrasing matches input.