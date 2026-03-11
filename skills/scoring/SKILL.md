```markdown
# SKILL: Scoring

## Task Description
This role is responsible for assigning scores or ratings based on predefined criteria. It involves evaluating information and applying a consistent scoring rubric.

## Output Format
The output should be a JSON object with the following structure:

```json
{
  "score": INTEGER,
  "reason": STRING,
  "confidence": FLOAT (0.0 to 1.0)
}
```

## Decision Framework

1. **Identify Criteria:** Understand the specific criteria to be used for scoring. This might involve properties, features, or other measurable aspects.
2. **Evaluate Information:** Analyze the input data or information in relation to the identified criteria.
3. **Assign Score:** Assign a numerical score based on the evaluation. The scoring scale should be clearly defined (e.g., 1-5, 1-10).
4. **Provide Reason:** Concisely explain the rationale behind the assigned score, referencing the specific criteria and evidence.
5. **Assess Confidence:** Determine a confidence score (0.0-1.0) reflecting the certainty in the assigned score. Factors affecting confidence include data quality, clarity of criteria, and ambiguity in the information.
   - **High Confidence (0.8-1.0):** Clear evidence aligns strongly with scoring criteria.
   - **Medium Confidence (0.5-0.7):** Some ambiguity or missing information exists.
   - **Low Confidence (0.0-0.4):** Significant uncertainty or conflicting evidence.
6. **Edge Case Handling:**
   - If information is insufficient to assign a meaningful score, return a score of -1 and explain the reason in the "reason" field. Confidence should be low (<=0.2).
   - If multiple criteria conflict, prioritize the most important criteria based on the context. Justify the prioritization in the "reason" field.

## Common Patterns

**Example 1:**

**Input:** Dataset with high data quality, clear documentation, and strong relevance to the research objective.

**Output:**
```json
{
  "score": 5,
  "reason": "Dataset exhibits high data quality, comprehensive documentation, and strong relevance to the research objective, making it highly suitable.",
  "confidence": 0.9
}
```

**Example 2:**

**Input:** Dataset with moderate data quality, limited documentation, and some relevance to the research objective.

**Output:**
```json
{
  "score": 3,
  "reason": "Dataset exhibits moderate data quality, limited documentation, and some relevance to the research objective, indicating moderate suitability.",
  "confidence": 0.6
}
```

**Example 3:**

**Input:** Dataset with low data quality, missing documentation, and weak relevance to the research objective.

**Output:**
```json
{
  "score": 1,
  "reason": "Dataset exhibits low data quality, missing documentation, and weak relevance to the research objective, indicating low suitability.",
  "confidence": 0.8
}
```

## Anti-Patterns

- **Inconsistent Scoring:** Applying different scoring criteria to similar inputs.
- **Vague Reasons:** Providing explanations that lack specific details or justification.
- **Overconfidence:** Assigning high confidence scores without sufficient evidence.
- **Ignoring Edge Cases:** Failing to handle situations with insufficient or conflicting information.
- **Returning scores outside the defined range.**
- **Forgetting to include `reason` and `confidence` fields.**

## Quality Checklist

- [ ] Score is within the defined range (e.g., 1-5).
- [ ] Reason is clear, concise, and justifies the assigned score.
- [ ] Confidence score accurately reflects the certainty in the assigned score.
- [ ] Edge cases are handled appropriately.
- [ ] Output adheres to the specified JSON format.
```