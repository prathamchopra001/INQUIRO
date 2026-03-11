```markdown
# SKILL: Report Writing

## Task Description
This skill focuses on synthesizing research findings into concise, well-structured reports. It involves identifying key themes, patterns, and insights from individual findings, and presenting them in a coherent and informative manner.

## Output Format
The output should be a structured report, formatted as follows:

```json
{
  "report_title": "Title of the Report",
  "executive_summary": "A brief overview of the report's key findings and conclusions (max 150 words).",
  "introduction": "Background information and the purpose of the report (max 100 words).",
  "key_findings": [
    {
      "theme": "Overarching theme or pattern identified",
      "supporting_evidence": [
        {
          "finding_id": "ID of the individual finding",
          "summary": "A concise summary of the finding and its relevance to the theme (max 50 words).",
          "confidence": "Confidence score of the finding (e.g., high, medium, low)."
        }
      ],
      "synthesis": "A brief synthesis of how the findings under this theme connect (max 75 words)."
    }
  ],
  "conclusion": "Summarize the overall implications and draw conclusions based on the key findings (max 100 words).",
  "limitations": "Acknowledge any limitations in the research or findings (max 50 words)."
}
```

## Decision Framework

1. **Identify Themes**: Group individual findings based on common topics, patterns, or relationships. Aim for 2-4 key themes.
2. **Prioritize Findings**: Select the most relevant and high-confidence findings to support each theme. Focus on findings with clear claims and strong evidence.
3. **Summarize Evidence**: Condense each selected finding into a brief summary highlighting its key points and relevance to the theme.
4. **Synthesize Themes**: Explain the connection between the findings within each theme. How do they support or contradict each other? What insights do they provide collectively?
5. **Write Sections**:
    *   **Executive Summary**: Briefly state the main findings and their implications.
    *   **Introduction**: Provide context and state the report's purpose.
    *   **Key Findings**: Detail each theme with supporting evidence and synthesis.
    *   **Conclusion**: Summarize overall implications.
    *   **Limitations**: Acknowledge any weaknesses.
6. **Confidence Scoring**: Use the provided confidence scores. If not provided, assess confidence based on the source type (e.g., peer-reviewed > expert opinion > blog post) and the clarity/strength of the evidence. High, medium, or low are acceptable values.
7. **Edge Case Handling**:
    *   **Conflicting Findings**: Acknowledge conflicting findings within the synthesis and explain potential reasons for the discrepancies.
    *   **Limited Findings**: If there are too few findings to identify clear themes, focus on summarizing the individual findings and highlighting potential areas for further research.

## Common Patterns

**Example 1**

**Input:** (Multiple findings related to the impact of social media on mental health)

**Output:** (A report with themes like "Social Media Addiction and Depression," "Cyberbullying and Anxiety," and "Positive Effects of Online Support Networks.")

**Example 2**

**Input:** (Findings about different renewable energy sources)

**Output:** (A report with themes like "Solar Energy Efficiency," "Wind Power Reliability," and "Geothermal Energy Sustainability.")

## Anti-Patterns

1. **Overly Descriptive**: Avoid simply listing the findings without identifying themes or synthesizing the information.
2. **Ignoring Confidence Scores**: Do not treat all findings as equally valid. Prioritize high-confidence findings.
3. **Missing Synthesis**: Do not present findings in isolation. Always explain how they relate to each other and the overall themes.
4. **Generic Conclusions**: Avoid vague or unsubstantiated conclusions. Ensure conclusions are directly supported by the key findings.
5. **Hallucinated Information**: Do not invent findings or data that are not present in the provided information.

## Quality Checklist

1. Are the key themes clearly identified and well-supported by evidence?
2. Is each finding accurately summarized and its relevance explained?
3. Is the synthesis insightful and does it connect the findings in a meaningful way?
4. Does the executive summary accurately reflect the report's key findings?
5. Are the conclusions justified based on the evidence presented?
6. Is the output in the correct JSON format?
7. Are limitations acknowledged?
```