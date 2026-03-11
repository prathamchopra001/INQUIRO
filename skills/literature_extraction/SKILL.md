# Literature Finding Extraction

## Task
Extract research findings from academic paper excerpts. Literature findings must ALWAYS be attributed to their source papers using the ACTUAL author names from the context.

## Output Format
Always respond with a valid JSON ARRAY (not wrapped in an object):
```json
[
  {
    "claim": "Prior work by Smith et al. found that...",
    "confidence": 0.85,
    "evidence": "Quote or paraphrase from paper",
    "paper_id": "ID from context",
    "paper_title": "Full paper title",
    "authors": "Smith, Jones, et al.",
    "tags": ["keyword1", "keyword2"]
  }
]
```

Return `[]` (empty array) if no extractable findings exist.

**IMPORTANT:** Return the array directly, NOT wrapped in `{"findings": [...]}`.

## Context Format
The context contains paper information in this format:
```
[Paper: Paper Title Here] [Authors: Smith, Jones, et al.] [DOI: xxx] [Paper ID: abc123]
Text content from the paper...
```

**CRITICAL:** Extract the ACTUAL author names from `[Authors: ...]` - do NOT use placeholder text like "[Authors]".

## Decision Framework

### What Makes a Literature Finding?
✅ **YES - Extract these:**
- Experimental results: "Smith et al. found that X increased by 15%..."
- Empirical observations: "The study showed 92% accuracy on..."
- Comparative results: "Their method outperformed baseline by..."
- Quantitative conclusions: "Results indicate a 2.3x improvement..."

❌ **NO - Skip these:**
- Paper metadata: "This paper was published in 2023..."
- Methodology descriptions: "The authors used PyTorch..."
- Future work: "Further research could..."
- Background/definitions: "Reinforcement learning is..."
- Vague statements: "The method works well"

### Attribution Rules (CRITICAL)
Every claim MUST start with attribution using REAL names:
- "Prior work by Smith et al. found that..."
- "According to 'Convergence of Q-Learning', researchers showed..."
- "Jones and colleagues demonstrated that..."
- "The study by Williams (2023) revealed..."

NEVER write claims like:
- ❌ "Prior work by [Authors] found..." (placeholder!)
- ❌ "X increases convergence speed" (no attribution)
- ❌ "We found that..." (you didn't find it)

### Confidence Scoring
| Score | Criteria |
|-------|----------|
| 0.90+ | Direct experimental result with statistics (p-values, CIs) |
| 0.75-0.89 | Clear conclusion with quantitative support |
| 0.60-0.74 | Qualitative finding with some evidence |
| 0.45-0.59 | Interpretation or inference from results |
| <0.45 | Weak claim, speculation, or review statement |

## Examples

### Input
```
[Paper: Convergence of Q-Learning] [Authors: Smith, Jones, Wang] [DOI: 10.1234/xyz] [Paper ID: sem_abc123]
Results show that the Q-learning agent with ε=0.1 achieved convergence in 450 episodes, compared to 780 episodes for ε=0.3 (p<0.01, n=50 runs).
```

### Output
```json
[
  {
    "claim": "Prior work by Smith et al. found that lower exploration rates (ε=0.1) lead to significantly faster convergence than higher rates (ε=0.3) in Q-learning agents",
    "confidence": 0.92,
    "evidence": "450 vs 780 episodes to convergence, p<0.01, n=50 runs",
    "paper_id": "sem_abc123",
    "paper_title": "Convergence of Q-Learning",
    "authors": "Smith, Jones, Wang",
    "tags": ["q-learning", "exploration", "convergence", "epsilon"]
  }
]
```

### Input (No findings)
```
[Paper: RL Survey] [DOI: N/A] [Paper ID: arxiv_456]
This section provides background on reinforcement learning algorithms.
```

### Output
```json
[]
```

## Anti-Patterns
- ❌ Using "[Authors]" placeholder instead of real names
- ❌ Missing attribution entirely
- ❌ Wrapping in `{"findings": [...]}` instead of just `[...]`
- ❌ Presenting as own discovery: "We found that..."
- ❌ Wrong paper_id: Using paper title instead of ID
- ❌ Extracting methodology as findings

## Quality Checklist
1. Is output a valid JSON array (not wrapped in object)?
2. Does EVERY claim use REAL author names (not "[Authors]")?
3. Is paper_id extracted from [Paper ID: xxx] in context?
4. Are authors extracted from [Authors: xxx] in context?
5. Does evidence support the confidence score?
