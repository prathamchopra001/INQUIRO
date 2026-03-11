# Scientific Finding Extraction

## Task
Extract scientific findings from academic paper chunks. A finding is a factual claim supported by evidence in the text.

## Output Format
Always respond with valid JSON:
```json
{
  "findings": [
    {
      "claim": "One clear factual statement",
      "confidence": 0.85,
      "evidence": "Direct quote or reference from text",
      "tags": ["keyword1", "keyword2"]
    }
  ]
}
```

Return `{"findings": []}` if no extractable findings exist.

## Decision Framework

### Is It a Finding?
✅ **YES - Extract these:**
- Experimental results: "X increased by 15% when..."
- Quantitative observations: "The model achieved 92% accuracy..."
- Causal relationships: "Higher learning rates led to..."
- Comparative results: "Method A outperformed B by..."

❌ **NO - Skip these:**
- Methodology: "We implemented using Python..."
- Background: "Reinforcement learning is a paradigm..."
- Hypotheses: "We hypothesize that..."
- Future work: "Further research could..."
- Definitions: "Q-learning is defined as..."

### Confidence Scoring
| Score | Criteria |
|-------|----------|
| 0.90+ | Direct experimental result with statistics (p-values, CIs) |
| 0.75-0.89 | Clear conclusion with quantitative support |
| 0.60-0.74 | Qualitative observation with some evidence |
| 0.45-0.59 | Inference or interpretation |
| <0.45 | Weak claim, speculation |

### Evidence Guidelines
- Quote directly when possible (use "...")
- Reference tables/figures: "As shown in Table 3..."
- Keep evidence concise (1-2 sentences max)

## Examples

### Input
"Results show that the Q-learning agent with ε=0.1 achieved convergence in 450 episodes, compared to 780 episodes for ε=0.3 (p<0.01, n=50 runs)."

### Output
```json
{
  "findings": [
    {
      "claim": "Lower exploration rate (ε=0.1) leads to faster convergence than higher rates (ε=0.3) in Q-learning",
      "confidence": 0.92,
      "evidence": "450 vs 780 episodes, p<0.01, n=50 runs",
      "tags": ["q-learning", "exploration", "convergence", "epsilon"]
    }
  ]
}
```

### Input
"We used the OpenAI Gym environment for our experiments. The agent was trained using a neural network with two hidden layers."

### Output
```json
{
  "findings": []
}
```
(This is methodology, not a finding)

## Anti-Patterns
- ❌ Extracting methodology as findings
- ❌ Including paper metadata (authors, publication date)
- ❌ Confidence scores without reasoning
- ❌ Vague claims: "The method works well"
- ❌ Multiple claims in one finding (split them)

## Quality Checklist
1. Is output valid JSON?
2. Is each claim a single, specific factual statement?
3. Does evidence support the confidence score?
4. Are tags relevant keywords (not sentences)?
5. Did I skip methodology/background/hypotheses?
