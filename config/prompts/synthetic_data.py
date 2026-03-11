SCHEMA_DESIGN_PROMPT = """You are a research data scientist designing a synthetic dataset.

## Research Objective
{objective}

## Instructions
Design a realistic dataset that would allow meaningful investigation of this objective.
Think carefully about:
- What entities/observations would rows represent?
- What variables (columns) are needed to study this objective?
- What realistic value ranges and distributions should each column have?
- What relationships between variables should exist (to be discoverable)?
- What sample size is appropriate? (aim for 500-5000 rows)

## Output Format
Respond with ONLY valid JSON. No explanation, no markdown.

{{
  "description": "Brief description of what this dataset represents",
  "rows": 1000,
  "columns": [
    {{
      "name": "column_name",
      "dtype": "float",
      "description": "What this column represents",
      "generation": "Brief description of how to generate it (distribution, range, etc.)"
    }}
  ],
  "relationships": [
    "column_a is positively correlated with column_b (r~0.6)",
    "column_c depends on column_d with added noise"
  ],
  "groups": {{
    "column_name": ["group_a", "group_b"],
    "description": "How groups differ"
  }}
}}

Design 8-20 columns. Include a mix of:
- Numeric continuous (float)
- Numeric discrete (int)
- Categorical (string)
- At least one time/period column if relevant
- At least one grouping column

Make the relationships realistic and discoverable through standard statistical analysis."""


CODE_GENERATION_PROMPT = """You are a data scientist writing Python code to generate a synthetic dataset.

## Dataset Schema
{schema_json}

## Research Objective (for context)
{objective}

## Rules
1. First line: # -*- coding: utf-8 -*-
2. Use ONLY: pandas, numpy, scipy.stats (no other libraries)
3. Generate the dataset according to the schema above
4. Plant the specified relationships between columns (with noise so they're not trivially obvious)
5. Save the dataset to: /app/outputs/synthetic_dataset.csv
6. Print the shape, column names, dtypes, and first 5 rows
7. Print summary statistics (df.describe())
8. Print correlation matrix for numeric columns
9. Do NOT use plt.show() or create any plots
10. Wrap everything in try/except and print any errors
11. Use np.random.seed(42) for reproducibility
12. os.makedirs('/app/outputs/', exist_ok=True) before saving
13. IMPORTANT: `lambda` is a Python reserved keyword. Do NOT use it as a variable name
    or parameter name. Use `lam`, `rate`, or `scale` instead.
14. After saving, ALWAYS print this exact line to confirm:
    print('SYNTHETIC_SAVE_OK')
    This is required for verification. Do not skip it.
15. CRITICAL BACKUP: After saving AND printing SYNTHETIC_SAVE_OK, also print the
    entire CSV to stdout between markers so it can be recovered if the file save fails:
    print('===SYNTHETIC_CSV_START===')
    print(df.to_csv(index=False))
    print('===SYNTHETIC_CSV_END===')
    This is mandatory. Do not skip it.

The dataset should be realistic enough that a researcher analyzing it
would discover meaningful patterns, but not so obvious that every
relationship is immediately apparent.

Write ONLY Python code. No markdown, no backticks."""

