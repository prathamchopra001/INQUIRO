---
task_hash: 248abce11fde
generated: 2026-03-10T14:00:42.518486
techniques: ['Agent-Based Modeling (ABM)', 'Time series analysis', 'Vector Autoregression (VAR)', 'Performance comparison (e.g., RMSE, MAE)', 'Statistical analysis']
libraries: ['Python', 'Mesa (for ABM)', 'statsmodels (for VAR)', 'NumPy', 'Pandas', 'scikit-learn (for performance metrics)', 'Matplotlib/Seaborn (for visualization)']
source: llm_analysis
---

```markdown
# Skill: ABM Performance Comparison with VAR Benchmark

## Task Description
Compare the forecasting performance of an Agent-Based Model (ABM) with varying proportions of learning firms against a VAR benchmark model for GDP, inflation, investments, household consumption, and government consumption.

## Required Packages
```bash
pip install mesa statsmodels numpy pandas scikit-learn
```

## Correct Imports
```python
import mesa
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
```

## DO NOT USE
*   Incorrect `statsmodels` import paths (e.g., deprecated modules).
*   Manual ABM implementation without using a framework like Mesa.
*   Ignoring stationarity checks for VAR model.

## Example Code

```python
import mesa
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

# Dummy ABM output (replace with actual ABM simulation)
def run_abm(learning_proportion, steps=100):
    # Simulate ABM and return time series data for GDP, inflation, etc.
    # This is a placeholder; implement your actual ABM here.
    data = pd.DataFrame(np.random.rand(steps, 5), columns=['GDP', 'Inflation', 'Investment', 'Consumption', 'GovConsumption'])
    return data

# VAR benchmark model
def run_var(data, forecast_steps=10):
    # Check for stationarity (simplified example)
    for col in data.columns:
        if sm.tsa.adfuller(data[col])[1] > 0.05: # Non-stationary
            data[col] = data[col].diff().dropna()  # Difference to make stationary

    model = VAR(data)
    results = model.fit(maxlags=10, ic='aic') # Select lag order using AIC
    forecast = results.forecast(data.values[-results.k_ar:], steps=forecast_steps)
    return forecast

# Performance comparison
def compare_performance(abm_data, var_forecast, actual_data):
    rmse = {}
    mae = {}
    for col_idx, col in enumerate(abm_data.columns):
        rmse[col] = mean_squared_error(actual_data[col][-len(var_forecast):], var_forecast[:, col_idx], squared=False)
        mae[col] = mean_absolute_error(actual_data[col][-len(var_forecast):], var_forecast[:, col_idx])
    return rmse, mae

# Main execution
learning_proportions = [0.1, 0.25, 0.5, 0.75, 1.0]
results = {}

for prop in learning_proportions:
    abm_data = run_abm(prop) # Get ABM output
    var_forecast = run_var(abm_data.copy()) # Generate VAR forecast using ABM data
    actual_data = abm_data # Assume ABM data is "actual" for now. Replace with real data.
    rmse, mae = compare_performance(abm_data, var_forecast, actual_data)
    results[prop] = {'rmse': rmse, 'mae': mae}

print(results) # Output results
```

## Common Pitfalls
*   **Non-Stationary Data:** VAR models require stationary data.  Always check and difference data if needed.  Use `sm.tsa.adfuller` for testing.
*   **Incorrect Lag Order:** Selecting the appropriate lag order for the VAR model is crucial. Use information criteria (e.g., AIC, BIC) to choose the best lag order.
*   **Data Alignment:** Ensure the ABM output and the actual data (if using separate real-world data) are properly aligned for performance comparison.
*   **Overfitting:**  Avoid overfitting the VAR model by carefully selecting the lag order and using regularization techniques if necessary.
*   **Interpreting ABM Output:** The `run_abm` function should return Pandas DataFrames with appropriate column names for the variables of interest (GDP, Inflation, etc.).