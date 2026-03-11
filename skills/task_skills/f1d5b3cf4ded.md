---
task_hash: f1d5b3cf4ded
generated: 2026-03-10T13:19:28.604994
techniques: ['Time series analysis', 'Correlation analysis', 'Regression analysis', 'Input-Output analysis', 'Descriptive statistics', 'Data visualization']
libraries: ['pandas', 'numpy', 'statsmodels', 'scikit-learn', 'matplotlib', 'seaborn']
source: llm_analysis
---

```markdown
# Skill: Italian Macroeconomic Data Analysis

## Task Description
Analyze Italian macroeconomic and sectoral data for ABM calibration and validation.

## Output Format
Markdown document detailing code patterns.

## Required Packages
```bash
pip install pandas numpy statsmodels scipy seaborn
```

## Correct Imports
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
```

## DO NOT USE
*   `pandas.Panel` (deprecated). Use `pandas.DataFrame` instead.
*   Manual loop-based calculations for correlation or regression on large datasets. Use vectorized operations.
*   Incorrectly specified regression models (e.g., omitting a constant term when needed).

## Example Code

```python
# Load data (replace with actual file paths)
gdp_data = pd.read_csv("italian_gdp_quarterly.csv", index_col='Date', parse_dates=True) # Ensure Date column exists
io_table = pd.read_csv("italian_input_output.csv") # Ensure table is correctly formatted
firm_data = pd.read_csv("italian_firm_counts.csv") # Ensure sector mapping

# Example 1: Time series analysis of GDP
gdp_data['GDP'].plot()
plt.title('Italian GDP (Quarterly)')
plt.show()

# Example 2: Correlation analysis
correlation, p_value = pearsonr(gdp_data['GDP'], gdp_data['Investment'])
print(f"Correlation between GDP and Investment: {correlation}, p-value: {p_value}")

# Example 3: Regression analysis
model = smf.ols('GDP ~ Investment + Consumption', data=gdp_data).fit()
print(model.summary())

# Example 4: Descriptive statistics
print(gdp_data.describe())

# Example 5: Basic Data Visualization (scatter plot)
plt.scatter(gdp_data['Investment'], gdp_data['GDP'])
plt.xlabel('Investment')
plt.ylabel('GDP')
plt.title('Investment vs. GDP')
plt.show()
```

## Common Pitfalls
*   **Data Type Errors:** Ensure data types are appropriate (numeric for calculations, datetime for time series). Use `pd.to_numeric()` or `pd.to_datetime()` to convert if needed.
*   **Missing Data:** Handle missing values using `df.dropna()` or `df.fillna()` before analysis.
*   **Incorrect File Paths:** Double-check file paths to avoid `FileNotFoundError`.
*   **Interpreting Correlation vs. Causation:** Remember that correlation does not imply causation.
*   **Multicollinearity:** Check for multicollinearity in regression models (high correlation between independent variables). Consider using VIF (Variance Inflation Factor) to detect it.
*   **Stationarity:** Verify stationarity for time series analysis. Use Augmented Dickey-Fuller test (ADF test) and techniques like differencing if needed.
*   **Index Alignment:** Ensure dataframes are properly aligned on dates or relevant indices before performing calculations.