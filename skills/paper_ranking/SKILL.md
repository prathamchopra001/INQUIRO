# SKILL.md: paper_ranking

## Task Description
Rank academic papers by relevance to a specific research objective, prioritizing domain-specific contributions, empirical evidence, and theoretical depth.

## Output Format
```json
{
  "ranked_papers": [
    {
      "title": "str",
      "author": "str",
      "conference": "str",
      "year": "int",
      "score": "int (0-100)",
      "relevance_notes": "str"
    }
  ]
}
```

## Decision Framework
1. **Align with Research Objective**: 
   - Higher scores for papers directly addressing the research question (e.g., "impact of exploration strategies on Q-learning convergence")
   - Lower scores for tangential work (e.g., papers on unrelated RL algorithms)

2. **Evaluate Empirical vs Theoretical Work**: 
   - Score empirical studies with concrete experiments (e.g., "UCB vs epsilon-greedy on Maze navigation")
   - Score theoretical analyses with convergence proofs (e.g., "UCB's regret bounds in Markov Decision Processes")

3. **Confidence Scoring Guidelines**: 
   - 90-100: Directly answers the research question with novel contributions
   - 70-89: Relevant but secondary (e.g., compares strategies in a specific scenario)
   - 50-69: Related but indirect (e.g., discusses exploration in a different context)

4. **Edge Case Handling**: 
   - Ambiguous relevance: Score 50-69 with notes explaining partial alignment
   - Overlapping domains: Prioritize papers with stronger methodological connection
   - Outdated work: Score ≤50 unless foundational to the field

## Common Patterns
**Example 1**:  
Input: "What are the key factors affecting Q-learning convergence?"  
Output:  
```json
{
  "ranked_papers": [
    {
      "title": "Convergence Rates of Q-Learning with Exploration Strategies",
      "author": "Smith et al.",
      "conference": "NeurIPS",
      "year": 2022,
      "score": 95,
      "relevance_notes": "Direct empirical comparison of epsilon-greedy, UCB, and softmax"
    },
    {
      "title": "Theoretical Analysis of Exploration Strategies in RL",
      "author": "Lee & Chen",
      "conference": "ICML",
      "year": 2021,
      "score": 88,
      "relevance_notes": "Formal convergence proofs for UCB and softmax"
    }
  ]
}
```

**Example 2**:  
Input: "How do exploration strategies affect sample efficiency?"  
Output:  
```json
{
  "ranked_papers": [
    {
      "title": "Sample Efficiency in Q-Learning: A Comparative Study",
      "author": "Zhang et al.",
      "conference": "ICRA",
      "year": 2023,
      "score": 92,
      "relevance_notes": "Empirical benchmarks across 10 exploration strategies"
    },
    {
      "title": "Exploration-Exploitation Tradeoffs in Reinforcement Learning",
      "author": "Brown & White",
      "conference": "AAAI",
      "year": 2020,
      "score": 75,
      "relevance_notes": "Broader discussion with partial focus on sample efficiency"
    }
  ]
}
```

## Anti-Patterns
- **Ignoring Domain Context**: Ranking papers outside the specified research domain (e.g., including papers on NLP)
- **Over-Reliance on Citations**: Prioritizing high-citation papers over methodologically relevant ones
- **Ambiguous Relevance Notes**: Vague statements like "somewhat related" without specific justification
- **Missing Required Fields**: Omitting "score" or "relevance_notes" in paper entries

## Quality Checklist
1. Verify JSON structure matches the schema exactly
2. Ensure all papers have required fields (title, author, etc.)
3. Confirm scores are integers between 0-100
4. Validate relevance_notes explicitly connect to the research objective
5. Remove duplicates and ensure chronological ordering for historical context