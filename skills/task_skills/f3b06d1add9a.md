---
task_hash: f3b06d1add9a
generated: 2026-03-07T20:23:45.376502
techniques: ['Time series analysis', 'ARIMA modeling', 'Agent-based modeling (ABM)', 'Forecasting', 'Performance comparison', 'Error metrics (e.g., RMSE, MAE)']
libraries: ['statsmodels', 'numpy', 'pandas', 'matplotlib', 'scipy']
source: llm_analysis
---

```markdown
# Skill: Compare ABM and ARIMA Forecasting Performance

## Task Description
Compare the forecasting performance of an Agent-Based Model (ABM) with learning firms to a simple ARIMA model using historical Italian macroeconomic data.

## Output Format
A Python script performing the analysis and printing performance metrics.

## Required Packages
```bash
pip install pandas statsmodels scikit-learn
```

## Correct Imports
```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np # for RMSE calculation
```

## DO NOT USE
*   `statsmodels.tsa.arima_model.ARIMA` (deprecated). Use `statsmodels.tsa.arima.model.ARIMA` instead.
*   Manual implementation of ARIMA if `statsmodels` provides it.
*   Ignoring data stationarity.

## Example Code

```python
# Sample Data (replace with actual Italian macroeconomic data)
data = {'GDP': [100, 102, 105, 103, 106, 108, 110, 109, 112, 115]}
df = pd.DataFrame(data)

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:len(df)]

# ARIMA Modeling
try:
    model = ARIMA(train['GDP'], order=(5,1,0)) # Example ARIMA order
    model_fit = model.fit()
    predictions_arima = model_fit.forecast(steps=len(test))
except Exception as e:
    print(f"ARIMA Model Fitting Error: {e}")
    predictions_arima = [np.nan] * len(test)  # Handle potential errors gracefully

# ABM Forecasts (replace with actual ABM output)
# Assume ABM generates forecasts for the test period
predictions_abm = [116, 118]  # Example ABM forecasts

# Error Metrics
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Ensure predictions_arima is not all NaN before calculating metrics
if all(np.isnan(x) for x in predictions_arima):
  rmse_arima = np.nan
  mae_arima = np.nan
else:
  rmse_arima = calculate_rmse(test['GDP'], predictions_arima)
  mae_arima = mean_absolute_error(test['GDP'], predictions_arima)

rmse_abm = calculate_rmse(test['GDP'], predictions_abm)
mae_abm = mean_absolute_error(test['GDP'], predictions_abm)

print(f'ARIMA RMSE: {rmse_arima}')
print(f'ARIMA MAE: {mae_arima}')
print(f'ABM RMSE: {rmse_abm}')
print(f'ABM MAE: {mae_abm}')

```

## Common Pitfalls
*   **Non-Stationary Data:** ARIMA requires stationary data. Apply differencing or other transformations if necessary. Test for stationarity using Augmented Dickey-Fuller (ADF) test.
*   **Incorrect ARIMA Order:** The `order` parameter (p, d, q) must be chosen carefully. Use techniques like AIC/BIC to optimize.
*   **Data Scaling:**  Consider scaling your data if the values are very large or small, which can affect model convergence.
*   **ABM Integration:** Ensure the ABM provides forecasts in a compatible format with the error metric calculations.  The ABM output should align with the testing period of the historical data.
*   **Handling ARIMA fit errors:** ARIMA model fitting can fail. Wrap the fitting in a `try...except` block and handle potential errors gracefully.
*   **NaN Predictions:** If the ARIMA model fails to fit and produces NaN predictions, the error metrics will also be NaN. Handle this case appropriately in the output.