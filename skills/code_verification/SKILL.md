```markdown
# SKILL: Code Verification

## Task Description
This role verifies code and its output for logical correctness, potential errors, and adherence to the original task requirements. It focuses on identifying bugs, inconsistencies, and areas for improvement in code and results.

## Output Format
The output should be a structured critique of the code and its output, focusing on logical correctness, potential errors, and adherence to the original task. The critique should be organized into sections:

*   **Overall Assessment:** A brief summary of the code's correctness and potential issues.
*   **Code Review:** Specific comments on the code, highlighting potential bugs, inefficiencies, or style issues. Focus on logical errors first.
*   **Output Verification:** Analysis of the code's output, checking for consistency, expected values, and adherence to the task requirements.
*   **Recommendations:** Actionable suggestions for improving the code and addressing any identified issues.

## Decision Framework

1.  **Understand the Task:** Thoroughly understand the original task requirements and the intended functionality of the code.
2.  **Code Review:**
    *   **Syntax and Style:** Briefly check for syntax errors and code style issues (e.g., PEP 8).
    *   **Logic:** Carefully examine the code's logic, focusing on potential bugs, edge cases, and incorrect assumptions.
    *   **Efficiency:** Identify potential performance bottlenecks or areas for optimization.
3.  **Output Verification:**
    *   **Consistency:** Check if the output is consistent with the code's logic and the task requirements.
    *   **Expected Values:** Verify if the output values are within the expected range and make sense in the context of the task.
    *   **Edge Cases:** Test the code with different inputs, including edge cases, to ensure it handles them correctly.
4.  **Confidence Scoring:**
    *   **High Confidence (90-100%):** The code is logically sound, produces correct output, and adheres to the task requirements. Any suggestions are minor improvements.
    *   **Medium Confidence (60-89%):** The code has some potential issues or areas for improvement, but it generally produces correct output.
    *   **Low Confidence (0-59%):** The code has significant logical errors, produces incorrect output, or does not adhere to the task requirements.

## Common Patterns

**Example 1:**

*   **Input:** Code that calculates the mean of a list of numbers but doesn't handle empty lists.
*   **Output:** "The code does not handle the case where the input list is empty, which will result in a `ZeroDivisionError`. Add a check for an empty list and return 0 in that case."

**Example 2:**

*   **Input:** Code that calculates Spearman's rank correlation but uses the wrong arguments for the `spear` function.
*   **Output:** "The `spear` function is used with default arguments. Consider setting `nan_policy='omit'` to handle potential NaN values in the data, which could lead to incorrect correlation calculations."

## Anti-Patterns

*   **Focusing solely on syntax errors:** Prioritize logical errors and potential bugs over minor syntax or style issues.
*   **Accepting output without verification:** Always verify the code's output against the task requirements and expected values.
*   **Providing vague or unhelpful feedback:** Be specific and actionable in your comments and suggestions.
*   **Ignoring edge cases:** Always consider potential edge cases and how the code handles them.

## Quality Checklist

*   [ ] I have thoroughly understood the original task requirements.
*   [ ] I have carefully reviewed the code for logical errors, potential bugs, and inefficiencies.
*   [ ] I have verified the code's output against the task requirements and expected values.
*   [ ] I have considered potential edge cases and how the code handles them.
*   [ ] I have provided specific and actionable feedback.
*   [ ] I have assigned a confidence score that accurately reflects the code's correctness.
```