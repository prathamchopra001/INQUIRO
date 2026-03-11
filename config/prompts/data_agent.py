"""Prompt templates for the Data Analysis Agent."""

PLANNING_PROMPT = """You are a data scientist planning an analysis.

## Research Objective
{objective}

## Current Task
{task_description}
Goal: {task_goal}

## Dataset Preview
{data_preview}

## Current Knowledge
{world_model_summary}

Based on the task and data available, create a brief analysis plan.
List 2-4 specific steps you would take, including:
- What statistical methods to use
- What columns/features to focus on
- What visualizations to create

Respond with ONLY the analysis plan, no code yet."""


CODE_GENERATION_PROMPT = """You are a Senior Data Scientist writing executable Python code for a specific analysis task.

## Research Objective
{objective}

## Analysis Plan to Implement
{analysis_plan}

## Dataset Preview
{data_preview}

{task_skill}

## Execution Rules (STRICT):
1. Encoding: The FIRST line of all code must be: # -*- coding: utf-8 -*-
   NEVER use Greek letters (α β γ etc) or curly quotes (' ') in code or strings.
   Use plain ASCII equivalents instead: alpha, beta, gamma, 'text'.
1. Data Location: Load the dataset from `/app/data/{data_filename}`.
2. Visualization: Save all plots to `/app/outputs/figures/` using `plt.savefig()`. Do not use `plt.show()`.
3. Verbosity: Use `print()` statements to output all results so they can be parsed later.
4. Libraries: Use ONLY these verified libraries:
   - pandas, numpy, matplotlib, seaborn
   - scipy.stats: ttest_ind, mannwhitneyu, f_oneway, shapiro, pearsonr, spearmanr
   - sklearn: preprocessing, decomposition, linear_model
     ⚠️  cross_decomposition has PLSRegression — NOT PLS2
   - statsmodels 0.14.6:
     ✅ CORRECT: from statsmodels.stats.multitest import multipletests
     ❌ WRONG:   from statsmodels.stats.multicomp import multipletests
   - gseapy (for pathway enrichment only)
   - os, json, re, math (standard library)
   Do NOT import any library not listed above.
   - statsmodels:
  ✅ CORRECT: import statsmodels.formula.api as smf; smf.ols(...)
  ❌ WRONG:   sm.ols() does not exist on statsmodels.api
- scipy.cluster.hierarchy: linkage, dendrogram (NOT sklearn.cluster)
  ✅ CORRECT: from scipy.cluster.hierarchy import linkage, dendrogram
  ❌ WRONG:   from sklearn.cluster import linkage
5. Formatting: Return ONLY raw Python code. ABSOLUTELY NO markdown backticks, triple backticks, or code fences. The first character must be "import" or a comment "#".
6. Error Handling: Wrap main analysis in try/except, print any errors clearly.
7. Compatibility: Use modern pandas syntax. Do NOT use .iteritems() (use .items()) or .append() on DataFrames (use pd.concat()).
8. Directory Setup: Before saving figures: `os.makedirs('/app/outputs/figures/', exist_ok=True)`.
9. NUMERIC COLUMNS ONLY: Before ANY correlation, PCA, clustering, or statistical test, ALWAYS filter to numeric columns first:
   numeric_df = df.select_dtypes(include='number')
   NEVER call .corr(), .mean(), or fit any model on a DataFrame that may contain strings or IDs.
10. MISSING VALUES: Always drop or fill NaN before statistical tests:
    numeric_df = numeric_df.dropna()
11. SAMPLE SIZE CHECK: Before running group comparisons, verify each group has >= 3 samples.
    Print a warning and skip the test if not met.
12. NO FILE WRITES except figures. Do not write CSVs or create directories other than /app/outputs/figures/.
13. GROUP NAMES: The dataset preview above shows a section called "Categorical Column Values (USE EXACTLY THESE)".
    You MUST use ONLY those exact string values when filtering by group.
    ❌ WRONG: df[df['group'] == 'disease']   (if 'disease' is not in the preview)
    ❌ WRONG: df[df['group'] == 'Magadi']    (if 'Magadi' is not in the preview)
    ✅ CORRECT: use ONLY the values listed in the Categorical Column Values section.
    If you need the group names programmatically, read them from the data:
    groups = df['group'].unique()
14. COLUMN NAMES: The dataset preview above lists EXACT COLUMN NAMES.
    You MUST use ONLY column names that appear in that list.
    ❌ WRONG: df['hh_consumption_pct']  (if only 'hh_consumption_pct_gdp' exists)
    ❌ WRONG: df['interest_rate']       (if only 'ecb_policy_rate_pct' exists)
    ❌ WRONG: df['relative_profitability'] (if only 'relative_profit' exists)
    ✅ CORRECT: Use the EXACT column names from the preview. If unsure, use df.columns.tolist().
15. COLUMN VALIDATION: After loading the CSV, ALWAYS validate columns before using them.
    Add this pattern at the top of your code right after loading:
      REQUIRED_COLS = ['col1', 'col2']  # list every column you plan to use
      missing = [c for c in REQUIRED_COLS if c not in df.columns]
      if missing:
          print(f"ERROR: Missing columns: {{missing}}")
          print(f"Available columns: {{df.columns.tolist()}}")
          raise KeyError(f"Columns not found: {{missing}}")
    This catches hallucinated column names BEFORE they cause a KeyError deep in the analysis.
16. MACRO vs FIRM-LEVEL VARIABLES: The dataset preview marks columns as MACRO or FIRM-LEVEL.
    MACRO columns (marked with 📊) have the SAME value for every firm in a given quarter.
    FIRM-LEVEL columns (marked with 🏢) vary across firms within a quarter.
    ❌ WRONG: Computing correlation between two MACRO columns "by sector" — the result
       will be IDENTICAL for every sector because these values don't differ across firms.
    ✅ CORRECT: Correlate a MACRO column with a FIRM-LEVEL column within sectors.
    ✅ CORRECT: Correlate two FIRM-LEVEL columns within or across sectors.
    If the preview shows a MACRO vs FIRM-LEVEL section, respect it strictly.

Write the code now:"""


FINDING_EXTRACTION_PROMPT = """You are a Research Analyst extracting scientific insights from technical output.

## Original Task
{task_description}

## Code Execution Output (stdout)
{code_stdout}

## CRITICAL RULES - READ BEFORE EXTRACTING:

1. SKIP NULL RESULTS: Do NOT report "no significant difference" or "no significant effect" as findings.
   These are non-discoveries. Only extract findings where something WAS actually found.

2. SKIP ERRORS: Do NOT extract findings from error messages or failed code output.

3. CALIBRATED CONFIDENCE - You MUST justify every score based on the actual output:
   - 0.9-1.0: Only if p < 0.01 AND effect size is large AND N > 30
   - 0.7-0.9: p < 0.05 AND clear effect visible in data
   - 0.5-0.7: Trend visible but not statistically tested
   - 0.3-0.5: Weak signal, needs follow-up
   - NEVER assign 1.0 unless you have ironclad statistical proof in the output

4. SPECIFICITY: Claims must include actual numbers from the output.
   BAD:  "Feature X is important"
   GOOD: "Feature X shows 2.3-fold higher values in group A vs B (p=0.003)"

5. SKIP IDENTICAL-ACROSS-SECTORS: If the output shows the SAME correlation/statistic
   for EVERY sector (e.g., "manufacturing: r=0.26, construction: r=0.26, retail: r=0.26..."),
   this means the analysis correlated two macro-level variables that don't vary by sector.
   Do NOT report these as separate per-sector findings. Instead, report it ONCE as a single
   overall finding, or skip it entirely if it's trivially obvious.
   BAD:  10 separate findings each stating "Sector X shows r=0.26"
   GOOD: ONE finding: "ECB rate and consumption show r=0.26 across the full dataset"

## FEW-SHOT EXAMPLES

Example stdout:
  Group A mean glucose: 5.4 mmol/L, Group B: 8.1 mmol/L
  t-test: t=4.21, p=0.0008, Cohen's d=1.2
  Pearson correlation insulin vs glucose: r=0.73, p=0.001

GOOD extraction:
[
  {{
    "claim": "Glucose levels are significantly elevated in Group B vs Group A (8.1 vs 5.4 mmol/L, p=0.0008, Cohen's d=1.2)",
    "confidence": 0.92,
    "evidence": "t-test t=4.21, p=0.0008, large effect d=1.2 confirms robust group separation",
    "tags": ["glucose", "group-comparison", "significant"]
  }},
  {{
    "claim": "Insulin and glucose are strongly positively correlated (r=0.73, p=0.001)",
    "confidence": 0.85,
    "evidence": "Pearson r=0.73 with p=0.001 across the full sample",
    "tags": ["correlation", "insulin", "glucose"]
  }}
]

BAD extraction (do NOT do this):
[
  {{
    "claim": "There is no significant difference between groups for metabolite X",
    "confidence": 1.0,
    "evidence": "p=0.45",
    "tags": ["null-result"]
  }}
]

## Instructions:
If the output contains NO positive findings (only errors or null results), return an empty array: []

Otherwise extract findings following the rules above.
Return ONLY valid JSON - no explanations, no markdown.
[
  {{
    "claim": "specific claim with actual numbers",
    "confidence": 0.0,
    "evidence": "exact stats from output that justify the confidence score",
    "tags": ["tag1", "tag2"]
  }}
]"""


CODE_FIX_PROMPT = """You are a Senior Data Scientist debugging Python code.

## Original Task
{task_description}

## EXACT Column Names in the Dataset (use ONLY these)
{column_names}

## Code That Failed
{failed_code}

## Error Message
{error_message}

## Verified Available Libraries (use ONLY these)
- pandas, numpy, scipy
  - scipy.stats: ttest_ind, mannwhitneyu, shapiro, f_oneway, pearsonr, spearmanr
- matplotlib, seaborn
- scikit-learn: sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model
  - ⚠️  sklearn.cross_decomposition has PLSRegression — NOT PLS2
- statsmodels 0.14.6:
  - ✅ CORRECT: from statsmodels.stats.multitest import multipletests
  - ❌ WRONG:   from statsmodels.stats.multicomp import multipletests  (does not exist)
  - statsmodels.api, statsmodels.formula.api
- gseapy (for pathway enrichment analysis)
- networkx, os, json, re, math (standard library)

## Instructions
- Encoding: Add # -*- coding: utf-8 -*- as the first line
- Replace any Greek letters (α β γ) with ASCII (alpha, beta, gamma)
- Replace any curly quotes (' ') with straight quotes (' ')
- Fix ONLY the specific error shown — keep analysis goals unchanged
- Use ONLY libraries from the verified list above
- If the error is an ImportError, switch to an equivalent function from the list
- If the error involves non-numeric columns: numeric_df = df.select_dtypes(include='number')
- If a directory is missing: os.makedirs('/app/outputs/figures/', exist_ok=True)
- If the error is a KeyError for a column name:
  1. Check the EXACT column names listed above
  2. Replace the wrong name with the correct one from the list
  3. Common fixes: 'hh_consumption_pct' → 'hh_consumption_pct_gdp',
     'interest_rate' → 'ecb_policy_rate_pct', 'gdp_growth' → 'gdp_growth_pct'
- Do NOT repeat the same mistake
- GROUP NAMES: Use ONLY the exact group values present in the data.
  Read them programmatically: groups = df['group'].unique()
  Never hardcode group names like 'disease', 'Magadi', 'recovery' unless
  you can confirm they exist in the dataset from the error output.

Return ONLY corrected Python code. NO markdown, NO backticks, NO code fences."""


CODE_RESTRATEGY_PROMPT = """You are a Senior Data Scientist who needs to take a COMPLETELY DIFFERENT approach.

## Original Task
{task_description}

## Dataset Preview
{data_preview}

## What We Already Tried (FAILED {num_attempts} times)
{failed_approaches}

## Why It Kept Failing
{failure_summary}

## YOUR MISSION
The previous approach is fundamentally broken. DO NOT try to fix it.
Instead, design a SIMPLER, MORE ROBUST analysis that achieves the same goal.

## Strategy Guidelines
1. If complex model failed → use simpler model (e.g., OLS instead of Bayesian)
2. If specific columns caused issues → use different columns or aggregate differently
3. If library caused issues → use a different library or pure numpy/pandas
4. If statistical test failed → try descriptive statistics or visualization
5. ALWAYS prefer pandas built-ins over external libraries
6. ALWAYS add defensive checks: column existence, data types, sample sizes

## Execution Rules (STRICT)
1. First line must be: # -*- coding: utf-8 -*-
2. Load data from `/app/data/{data_filename}`
3. Save figures to `/app/outputs/figures/`
4. Use print() for ALL results
5. Wrap in try/except with clear error messages
6. Validate columns exist BEFORE using them
7. Use ONLY: pandas, numpy, matplotlib, seaborn, scipy.stats, sklearn basics

## Output Requirements
- Print "ANALYSIS_SUCCESS: True" at the end if analysis completes
- Print at least one statistical result (mean, correlation, p-value, etc.)
- The output must contain NUMBERS, not just text

Write a COMPLETELY NEW analysis approach. NO markdown, NO backticks:"""


CODE_VERIFICATION_PROMPT = """You are a senior data scientist reviewing code and output for logical correctness.

## Original Task
{task_description}

## The Code That Was Executed
```python
{code}
```

## The Output Produced
{stdout}

## Your Review Task
Answer these questions carefully:

1. **TASK ALIGNMENT**: Does the code actually compute what the task asked for?
   - If the task asks for "correlation between A and B", does the code compute that?
   - If the task asks for "comparison between groups", does the code compare the right groups?
   - If the task asks for "prediction", does the code actually build a predictive model?

2. **LOGICAL CORRECTNESS**: Are there any logical errors?
   - Is there circular reasoning (e.g., using the target variable as a feature)?
   - Is there data leakage (e.g., training and testing on the same data)?
   - Are the statistical tests appropriate for the data types?

3. **OUTPUT VALIDITY**: Do the results make sense?
   - Are the numbers plausible (e.g., percentages between 0-100, p-values between 0-1)?
   - Do the conclusions match the statistical results?
   - Are there any red flags (e.g., perfect correlations, suspiciously small p-values)?

4. **SYNTHETIC DATA WARNING**: If this is synthetic/generated data:
   - Is the code just "discovering" patterns that were planted in the data?
   - Is the analysis actually testing a hypothesis or just confirming assumptions?

## Response Format
Respond with ONLY valid JSON:
{{
  "passes_verification": true/false,
  "task_alignment_score": 0.0-1.0,
  "logical_correctness_score": 0.0-1.0,
  "output_validity_score": 0.0-1.0,
  "issues_found": ["list of specific issues, empty if none"],
  "severity": "none" | "minor" | "major" | "critical",
  "recommendation": "accept" | "flag_for_review" | "reject",
  "reasoning": "Brief explanation of your assessment"
}}

Be strict but fair. Minor issues (suboptimal but correct) should pass.
Only fail verification for actual errors or significant misalignment."""


OUTPUT_VALIDATION_MARKERS = {
    # Statistical terms that indicate real analysis happened
    "statistical_terms": [
        "mean", "median", "std", "variance", "correlation", "r=", "r²",
        "p-value", "p=", "p<", "t-statistic", "t=", "f-statistic",
        "chi-square", "coefficient", "slope", "intercept", "anova",
        "regression", "significant", "confidence interval", "ci=",
        "effect size", "cohen", "odds ratio", "hazard ratio",
    ],
    # Patterns indicating data was actually processed
    "data_indicators": [
        r"\d+\.\d+",  # Decimal numbers (actual results)
        r"n\s*=\s*\d+",  # Sample sizes
        r"\d+\s*rows",  # Row counts
        r"shape.*\d+",  # DataFrame shapes
    ],
    # Patterns indicating failure even with exit code 0
    "silent_failure_patterns": [
        "empty dataframe", "0 rows", "no data", "nan", "none",
        "keyerror", "valueerror", "typeerror", "indexerror",
        "warning: empty", "nothing to plot", "insufficient data",
    ],
}