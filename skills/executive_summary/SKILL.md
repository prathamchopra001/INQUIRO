# SKILL.md

## Task Description  
Create a concise, structured summary of complex technical or research content, emphasizing key objectives, findings, and implications for stakeholders.

## Output Format  
**JSON Schema**:  
```json
{
  "objective": "Briefly state the research/analysis goal",
  "key_findings": ["Concise bullet points highlighting major results"],
  "recommendations": ["Actionable insights for decision-makers"],
  "uncertainties": ["Notes on limitations or data gaps"]
}
```

## Decision Framework  
1. **Prioritize**: Focus on 3-5 core objectives over granular details.  
2. **Confidence Scoring**:  
   - High (✅): Supported by explicit data or quotes  
   - Medium (⚠️): Inferred from patterns, note limitations  
   - Low (❌): Avoid unless explicitly stated in source  
3. **Edge Cases**:  
   - Ambiguous data → Use "uncertainties" field  
   - Conflicting findings → Present both with confidence scores  

## Common Patterns  
**Example 1**:  
**Input**: "Research on renewable energy adoption in rural areas"  
**Output**:  
```json
{
  "objective": "Assess barriers to renewable energy adoption in rural regions",  
  "key_findings": ["High upfront costs deter 68% of rural communities", "Government subsidies increase adoption by 40%", "Lack of technical expertise persists in 72% of cases"],  
  "recommendations": ["Target subsidies toward low-income households", "Launch community training programs"],  
  "uncertainties": ["Long-term cost-benefit data insufficient"]  
}
```  

**Example 2**:  
**Input**: "Analysis of AI ethics frameworks across 15 countries"  
**Output**:  
```json
{
  "objective": "Compare AI ethics regulations globally",  
  "key_findings": ["EU leads with comprehensive data protection laws", "Asia-Pacific nations prioritize transparency", "Sub-Saharan Africa lacks standardized frameworks"],  
  "recommendations": ["Promote cross-regional collaboration", "Develop adaptable compliance tools"],  
  "uncertainties": ["Implementation gaps in emerging markets"]  
}
```  

## Anti-Patterns  
- **Omitting "uncertainties"** when data gaps exist  
- **Using vague recommendations** like "further research is needed"  
- **Exceeding 5 key findings** without justification  
- **Including implementation details** instead of high-level insights  

## Quality Checklist  
1. Verify JSON structure matches schema exactly  
2. Ensure all key findings are actionable and data-backed  
3. Confirm uncertainties are explicitly labeled and relevant  
4. Validate recommendations align with objectives  
5. Check for markdown formatting or syntax errors  
6. Confirm length: < 300 words total