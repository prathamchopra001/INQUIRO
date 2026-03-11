---
task_hash: 8caca9aa18c0
generated: 2026-03-07T22:58:34.937738
techniques: ['Agent-Based Modeling (ABM)', 'Vector Autoregression (VAR)', 'Time Series Analysis', 'Forecasting', 'Performance Comparison', 'Statistical Analysis', 'Model Calibration']
libraries: ['NumPy', 'Pandas', 'Statsmodels', 'Scikit-learn', 'potentially a library for ABM such as Mesa or NetLogo (via rpy2)']
source: llm_analysis
---

```markdown
# Skill: Comparative Forecasting with ABM and VAR

## Task Description
Compare the forecasting performance of an Agent-Based Model (ABM) with varying proportions of learning firms to a Vector Autoregression (VAR) model, both calibrated to Italian macroeconomic data.

## Required Packages
```bash
pip install mesa statsmodels numpy pandas
```

## Correct Imports
```python
import mesa
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
```

## DO NOT USE
*   Manual loop-based forecasting in VAR (use `forecast()` method).
*   Ignoring stationarity checks before VAR modeling.
*   Incorrect exogenous variable handling in VAR.

## Example Code

```python
# Dummy data for ABM (replace with actual ABM simulation results)
abm_forecasts_50 = np.random.rand(20) # 50% learning firms
abm_forecasts_75 = np.random.rand(20) # 75% learning firms

# Example Italian macroeconomic data (replace with actual data)
data = pd.DataFrame({
    'gdp': np.random.rand(100),
    'inflation': np.random.rand(100),
    'interest_rate': np.random.rand(100)
})

# VAR Model
model = VAR(data)
results = model.fit(maxlags=5) # Choose appropriate lag order
var_forecasts = results.forecast(data.values[-5:], steps=20) # Forecast 20 steps ahead

# Convert VAR forecasts to DataFrame for easier handling
var_forecasts_df = pd.DataFrame(var_forecasts, columns=data.columns)

# Evaluation (example using GDP)
actual_gdp = np.random.rand(20) # Replace with actual GDP data for the forecast period
rmse_abm_50 = np.sqrt(mean_squared_error(actual_gdp, abm_forecasts_50))
rmse_abm_75 = np.sqrt(mean_squared_error(actual_gdp, abm_forecasts_75))
rmse_var = np.sqrt(mean_squared_error(actual_gdp, var_forecasts_df['gdp'].values))


print(f"RMSE ABM (50%): {rmse_abm_50}")
print(f"RMSE ABM (75%): {rmse_abm_75}")
print(f"RMSE VAR: {rmse_var}")
```

## Common Pitfalls
*   **Data Preprocessing:** Ensure data is stationary before VAR modeling. Use differencing if needed.  `data = data.diff().dropna()`
*   **Lag Order Selection:**  Use information criteria (AIC, BIC) to select the optimal lag order for the VAR model. `results = model.fit(ic='aic')`
*   **Calibration:** The ABM needs a proper calibration strategy, which is not included here.
*   **Exogenous Variables:**  Handle exogenous variables carefully in VAR. Use the `exog` parameter in the `VAR` constructor and `forecast` method.
*   **Evaluation Metric:** Choose appropriate evaluation metrics (RMSE, MAE, etc.) based on the forecasting goal.
*   **Statistical Significance:**  Test for statistical significance when comparing the performance of the models.