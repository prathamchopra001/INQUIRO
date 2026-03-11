# completion_check Skill File

## Task Description  
Evaluate whether a research process has gathered sufficient evidence to produce a meaningful discovery report. Determine completion status based on objective clarity, data sufficiency, and methodological rigor.

## Output Format  
```json
{
  "is_complete": boolean,
  "confidence_score": number (0-100),
  "rationale": string
}
```

## Decision Framework  
1. **Check Objective Clarity**:  
   - If the research goal is ambiguous or overly broad → flag for clarification (confidence ≤ 30)  
   - If the objective is specific and measurable → proceed to data assessment  

2. **Assess Data Sufficiency**:  
   - High-quality, relevant data covering all specified areas → high confidence  
   - Missing critical data points (e.g., gaps in biodiversity monitoring) → medium confidence  
   - Irrelevant or insufficient data → low confidence  

3. **Evaluate Methodology**:  
   - Robust methods with clear analysis → high confidence  
   - Weak methodology or unvalidated techniques → medium confidence  
   - No methodology described → low confidence  

4. **Confidence Scoring Guidelines**:  
   - 80-100: All criteria met, no gaps  
   - 50-79: Partial gaps, but sufficient for preliminary conclusions  
   - 0-49: Critical gaps or unclear objectives  

5. **Edge Case Handling**:  
   - Ambiguous data sources → request clarification before scoring  
   - Overly technical jargon without explanation → deduct 20% from confidence score  

## Common Patterns  
**Example 1**  
Input: "Investigate X... Cover six areas: (1) BIODIVERSITY MONITORING..."  
Output:  
```json
{
  "is_complete": false,
  "confidence_score": 45,
  "rationale": "Objective is specific, but lacks data on observability engineering applications. Missing critical areas like data integration frameworks."
}
```  

**Example 2**  
Input: "Analyzed 100+ ecological datasets using OpenTelemetry... Validated against GBIF standards"  
Output:  
```json
{
  "is_complete": true,
  "confidence_score": 92,
  "rationale": "Clear objective met with comprehensive data. Methodology aligns with observability principles and includes validation against established benchmarks."
}
```  

**Example 3**  
Input: "Explored potential applications of observability engineering... No specific datasets mentioned"  
Output:  
```json
{
  "is_complete": false,
  "confidence_score": 28,
  "rationale": "Objective is vague. No data sources or methodology described. Cannot assess sufficiency of evidence."
}
```  

## Anti-Patterns  
- **Vague Rationales**: "Some data is missing" → must specify which areas  
- **Binary Outcomes**: Only "yes/no" without confidence scoring  
- **Ignoring Methodology**: Focusing solely on data volume instead of analytical rigor  
- **Unbounded Confidence**: Assigning 100% confidence without validating all criteria  

## Quality Checklist  
1. Output must strictly follow the JSON schema  
2. Confidence scores must be integers between 0-100  
3. Rationale must explicitly reference the research objective and gaps  
4. Edge cases (ambiguous data, unclear methods) must trigger specific handling rules  
5. Avoid technical jargon without explanation in the rationale