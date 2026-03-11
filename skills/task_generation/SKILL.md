# SKILL.md: task_generation

## Task Description  
Generates focused research tasks to advance a scientific objective by addressing knowledge gaps and balancing exploration/exploitation. Prioritizes tasks that validate strong findings or explore high-impact hypotheses.

## Output Format  
**JSON Array** of objects with these fields:  
- `type`: "Exploration" (new hypotheses) or "Exploitation" (existing hypothesis validation)  
- `description`: Specific action to take (e.g., "Test hypothesis X under condition Y")  
- `goal`: What the task aims to achieve (e.g., "Validate convergence rate claims")  
- `priority`: Numerical value (1 = high, 3 = medium, 5 = low)  

## Decision Framework  
1. **Gap Analysis**: Identify unaddressed questions in current findings or world model summary.  
2. **Task Typing**:  
   - *Exploration*: When objective requires testing novel hypotheses or unexplored variables.  
   - *Exploitation*: When objective demands rigorous validation of top-ranked findings.  
3. **Confidence Scoring**:  
   - High (priority 1): Tasks based on strong, peer-reviewed evidence.  
   - Medium (priority 3): Tasks requiring controlled experiments or simulations.  
   - Low (priority 5): Tasks with weak evidence or high uncertainty.  
4. **Edge Cases**:  
   - If no gaps exist, prioritize validating the most impactful finding.  
   - Avoid overlapping tasks by checking for duplicate descriptions or goals.  

## Common Patterns  
### Example 1  
**Input**:  
Objective: "Key factors affecting Q-learning convergence"  
Current Findings: "Algorithm X converges 2x faster than Y in grid worlds"  

**Output**:  
```json  
[  
  {  
    "type": "Exploitation",  
    "description": "Validate Algorithm X's convergence rate in maze environments",  
    "goal": "Confirm generalizability of grid-world results",  
    "priority": 1  
  }  
]  
```  

### Example 2  
**Input**:  
Objective: "Factors affecting Q-learning convergence"  
Current Findings: "No prior research on reward sparsity's impact"  

**Output**:  
```json  
[  
  {  
    "type": "Exploration",  
    "description": "Simulate Q-learning with sparse rewards in 10+ environments",  
    "goal": "Quantify reward sparsity's effect on convergence",  
    "priority": 1  
  }  
]  
```  

## Anti-Patterns  
- **Redundant Tasks**: "Test Algorithm X in grid worlds" if that's already been done.  
- **Vague Descriptions**: "Check convergence" without specifying conditions or metrics.  
- **Incorrect Priorities**: Assigning priority 1 to speculative ideas with no evidence.  
- **Missing Fields**: Omitting `type` or `priority` in task objects.  

## Quality Checklist  
1. All tasks have required fields (`type`, `description`, `goal`, `priority`).  
2. Priorities are integers (1-5) and reflect task impact/feasibility.  
3. No duplicate tasks with identical descriptions/goals.  
4. Tasks align with the objective and current knowledge state.  
5. Exploration tasks address unexplored variables or hypotheses.  
6. Exploitation tasks validate top findings with concrete experiments.