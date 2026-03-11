---
task_hash: fcff4fbef618
generated: 2026-03-07T18:21:38.076222
techniques: ['Descriptive Statistics', 'Distribution Fitting', 'Market Share Calculation', 'Data Aggregation']
libraries: ['pandas', 'numpy', 'scipy', 'matplotlib/seaborn']
source: llm_analysis
---

```markdown
# Skill: Italian Macroeconomic Data Analysis

## Task Description
Analyze Italian macroeconomic data to determine firm size and market share distributions across sectors.

## Output Format
A Python script that performs descriptive statistics, distribution fitting, market share calculation, and data aggregation.

## Required Packages
```bash
pip install pandas numpy scipy matplotlib
```

## Correct Imports
```python
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, gamma  # Example distributions
import matplotlib.pyplot as plt
```

## DO NOT USE
*   `statsmodels` for basic descriptive statistics (use pandas).
*   Outdated plotting libraries (use `matplotlib` directly).
*   Looping through DataFrames for calculations (use vectorized operations).

## Example Code

```python
import pandas as pd
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# Sample Data (replace with your actual data loading)
data = {'Sector': ['A', 'A', 'B', 'B', 'C'],
        'Firm': ['Firm1', 'Firm2', 'Firm3', 'Firm4', 'Firm5'],
        'Revenue': [100, 150, 200, 250, 300],
        'Employees': [10, 15, 20, 25, 30]}
df = pd.DataFrame(data)

# 1. Data Aggregation (Sector Level)
sector_data = df.groupby('Sector').agg({'Revenue': 'sum', 'Employees': 'sum'})

# 2. Market Share Calculation
total_revenue = df['Revenue'].sum()
df['MarketShare'] = df['Revenue'] / total_revenue

# 3. Firm Size Distribution (example: log-normal)
firm_sizes = df['Employees']
shape, loc, scale = lognorm.fit(firm_sizes)  # Fit log-normal distribution
x = np.linspace(firm_sizes.min(), firm_sizes.max(), 100)
pdf = lognorm.pdf(x, shape, loc, scale)

# Plotting the distribution
plt.hist(firm_sizes, density=True, alpha=0.6, label='Employee Sizes')
plt.plot(x, pdf, 'r-', label='Log-Normal PDF')
plt.xlabel('Number of Employees')
plt.ylabel('Density')
plt.title('Distribution of Firm Sizes')
plt.legend()
plt.show()


print(sector_data)
print(df)
```

## Common Pitfalls
*   **Incorrect Data Types:** Ensure numerical columns are correctly typed (e.g., `float64` or `int64`). Use `df.dtypes` and `df.astype()` to verify and correct.
*   **Zero Division:** Handle cases where total revenue might be zero when calculating market share. Use `np.where` or `.replace()` to avoid errors.
*   **Distribution Fitting Errors:** Ensure that the data is appropriate for the chosen distribution. Check for non-positive values when fitting distributions like log-normal.
*   **Incorrect Grouping:** Verify that the grouping is done correctly to avoid skewed aggregated results.