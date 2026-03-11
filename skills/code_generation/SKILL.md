```markdown
# SKILL: Code Generation for Data Analysis

This skill generates Python code for data analysis tasks, focusing on clarity, executability, and error handling. The code should use pandas, matplotlib, and scipy where appropriate.

## Output Format

Python code as a single string. The code should be well-formatted and include comments explaining each step. The code should be ready to be executed. Always include error handling using `try...except` blocks.

## Decision Framework

1.  **Understand the Objective**: Carefully analyze the task description, dataset information, and research objective to fully grasp the analysis goal.
2.  **Data Loading**: Load the data using `pandas`. Handle potential file errors (e.g., file not found) using `try...except`.
3.  **Data Cleaning**: Based on the objective, clean the data. This may include handling missing values (imputation or removal), removing duplicates, and converting data types.
4.  **Data Analysis**: Perform the core data analysis based on the objective. This might involve:
    *   **Descriptive Statistics**: Calculate means, medians, standard deviations using `pandas`.
    *   **Data Visualization**: Create plots and charts using `matplotlib` to explore data relationships. Choose appropriate plot types (scatter plots, histograms, bar charts, etc.). Label axes and provide titles.
    *   **Statistical Analysis**: Use `scipy` for statistical tests (e.g., t-tests, ANOVA, correlation tests).
5.  **Error Handling**: Wrap potentially problematic sections of code in `try...except` blocks to catch errors and print informative error messages.
6.  **Output**: Ensure the code prints the results of the analysis in a clear and understandable format.
7.  **Confidence Scoring**: High confidence if all steps are implemented correctly, error handling is robust, and the code is well-commented and executable. Medium confidence if some parts are missing or could be improved. Low confidence if the code is incomplete, contains errors, or doesn't address the objective.

## Common Patterns

**Example 1:**

**Input:** "Analyze the 'iris' dataset to show the relationship between sepal length and sepal width using a scatter plot."

**Output:**

```python
import pandas as pd
import matplotlib.pyplot as plt

try:
    # Load the iris dataset
    iris = pd.read_csv('iris.csv')

    # Create a scatter plot of sepal length vs sepal width
    plt.figure(figsize=(8, 6))
    plt.scatter(iris['sepal_length'], iris['sepal_width'])
    plt.title('Sepal Length vs Sepal Width in Iris Dataset')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print("Error: The iris.csv file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
```

**Example 2:**

**Input:** "Calculate the mean and standard deviation of the 'age' column in the 'customers' dataset."

**Output:**

```python
import pandas as pd

try:
    # Load the customers dataset
    customers = pd.read_csv('customers.csv')

    # Calculate the mean and standard deviation of the age column
    mean_age = customers['age'].mean()
    std_age = customers['age'].std()

    # Print the results
    print(f"Mean age: {mean_age:.2f}")
    print(f"Standard deviation of age: {std_age:.2f}")

except FileNotFoundError:
    print("Error: The customers.csv file was not found.")
except KeyError:
    print("Error: The 'age' column was not found in the dataset.")
except Exception as e:
    print(f"An error occurred: {e}")
```

## Anti-Patterns

*   **Not including error handling**: Always include `try...except` blocks to handle potential errors.
*   **Generating non-executable code**: Ensure the code is syntactically correct and can be run without errors.
*   **Not commenting the code**: Add comments to explain each step of the analysis.
*   **Ignoring the objective**: Ensure the code directly addresses the specified analysis objective.
*   **Using hardcoded file paths without checking existence**: Verify that data files exist before attempting to load them.
*   **Creating overly complex or unnecessary code**: Keep the code as simple and straightforward as possible.
*   **Failing to label plots correctly**: Always label axes and provide titles for plots.

## Quality Checklist

*   [ ] The code is executable and produces the expected output.
*   [ ] Error handling is implemented using `try...except` blocks.
*   [ ] The code is well-commented and easy to understand.
*   [ ] The code addresses the specified analysis objective.
*   [ ] The output is clear and understandable.
*   [ ] The code uses pandas, matplotlib, and scipy appropriately.
*   [ ] Plots are labeled correctly.
*   [ ] File paths are handled robustly.
```