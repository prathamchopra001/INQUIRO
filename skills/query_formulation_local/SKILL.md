# Query Formulation for Local Models

## Task
Create 3-5 search queries from a research task.

## Output Format
JSON array only. No explanation.

```json
["query 1", "query 2", "query 3"]
```

## Step-by-Step Process

1. Find the MAIN TOPIC (2-3 key words)
2. Find RELATED TERMS (synonyms, methods)
3. Make SHORT queries (3-6 words each)
4. Output as JSON array

## Examples

**Example 1:**
Task: Find papers about machine learning for disease diagnosis
```json
["machine learning disease diagnosis", "deep learning medical imaging", "AI healthcare prediction"]
```

**Example 2:**
Task: Research carbon capture in forests  
```json
["forest carbon sequestration", "tree carbon storage rates", "reforestation climate mitigation"]
```

**Example 3:**
Task: Investigate sensor networks for wildlife
```json
["wildlife sensor monitoring", "IoT biodiversity tracking", "acoustic species detection"]
```

## Rules
- Each query: 3-6 words
- Include topic words
- No generic words like "study" or "analysis"
- Output ONLY the JSON array
