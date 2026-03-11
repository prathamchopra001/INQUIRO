# SKILL.md

## Task Description  
Synthesize complex research findings into a structured, narrative-driven discovery report that balances technical depth with accessibility. Focus on clarity, logical flow, and alignment with the research objective.

## Output Format  
```json
{
  "title": "string",
  "introduction": "string",
  "methodology": "string",
  "key_findings": "string",
  "implications": "string",
  "conclusion": "string",
  "confidence_score": number // 0-1
}
```

## Decision Framework  
1. **Parse Input**: Identify core research objective, key areas, and required sections (e.g., "BIODIVERSITY MONITORING" from the example).  
2. **Structure Sections**:  
   - *Introduction*: Frame the problem and objective.  
   - *Methodology*: Describe approaches and tools (e.g., "OpenTelemetry for observability").  
   - *Key Findings*: Highlight 3-5 major insights with evidence.  
   - *Implications*: Discuss practical/ethical impacts.  
   - *Conclusion*: Summarize and suggest next steps.  
3. **Confidence Scoring**:  
   - 1.0: All sections are fully addressed with clear evidence.  
   - 0.5: Some sections are missing or underdeveloped.  
   - 0.0: Input lacks sufficient detail for structured output.  
4. **Edge Cases**:  
   - If input is vague, request clarification (e.g., "Specify gaps in frontline incident capture").  
   - If conflicting data exists, prioritize the most cited/validated sources.  

## Common Patterns  
**Example 1**  
Input: "Investigate how OpenTelemetry can improve biodiversity monitoring."  
Output:  
```json
{
  "title": "Observability Engineering for Biodiversity Monitoring",
  "introduction": "This report explores OpenTelemetry's potential to enhance real-time ecological data collection.",
  "methodology": "Analysis of GBIF data integration and OpenTelemetry's metrics pipelines.",
  "key_findings": "OpenTelemetry reduces data latency by 40% in pilot studies; gaps exist in species-specific tracking.",
  "implications": "Improved data sharing could boost conservation efforts but requires standardized protocols.",
  "conclusion": "OpenTelemetry offers scalable tools for ecological monitoring with targeted implementation.",
  "confidence_score": 0.9
}
```  

**Example 2**  
Input: "Summarize GEO BON's role in biodiversity variables."  
Output:  
```json
{
  "title": "GEO BON and Essential Biodiversity Variables",
  "introduction": "GEO BON standardizes metrics to track global biodiversity trends.",
  "methodology": "Review of GEO BON's framework and GBIF integration strategies.",
  "key_findings": "GEO BON defines 25 core variables; GBIF provides 80% of required data.",
  "implications": "Standardization enables cross-border ecological analysis but faces data sovereignty challenges.",
  "conclusion": "GEO BON's framework is critical for scalable biodiversity monitoring.",
  "confidence_score": 0.85
}
```  

## Anti-Patterns  
- **Missing Sections**: Omitting "methodology" or "implications" invalidates the report.  
- **Unsupported Claims**: Stating "OpenTelemetry solves all ecological data issues" without evidence.  
- **Overly Technical Jargon**: Using terms like "telemetry pipelines" without explanation for non-specialist audiences.  

## Quality Checklist  
1. Verify all required JSON fields are present.  
2. Ensure confidence_score reflects content completeness.  
3. Confirm key findings are data-backed and specific.  
4. Check language is accessible to mixed audiences (scientists/decision-makers).  
5. Validate alignment with the original research objective.