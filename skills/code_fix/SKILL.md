```markdown
# SKILL: code_fix

## Task Description
This skill focuses on fixing errors in code snippets to ensure they are functional and meet the implied requirements of the provided context. The primary goal is to debug and correct code, not to rewrite it from scratch unless necessary.

## Output Format
The output should be a corrected code snippet, ready to be executed. Include comments explaining the changes made and the reasoning behind them. The corrected code should be a complete and runnable snippet.

## Decision Framework
1. **Understand the Error:** Carefully analyze the error messages (if any) and the code's intent based on surrounding context.
2. **Identify the Root Cause:** Pinpoint the specific line(s) of code causing the error or preventing the code from achieving its intended purpose.
3. **Implement the Fix:** Modify the code to correct the error. Prioritize minimally invasive changes that address the core issue.
4. **Add Comments:** Clearly explain the changes made and why they were necessary.
5. **Test (Inferred):** Mentally simulate the code execution to ensure the fix resolves the issue and doesn't introduce new problems. Consider edge cases.
6. **Handle Edge Cases:** If the original code has potential edge cases or vulnerabilities, address them with appropriate error handling or input validation.
7. **Confidence Scoring:**
    - **High Confidence (90-100%):** The error is clear, the fix is straightforward, and the corrected code demonstrably works.
    - **Medium Confidence (70-89%):** The error is understood, but the fix involves some assumptions or potential trade-offs.
    - **Low Confidence (50-69%):** The error is unclear, the fix is a best guess, and further testing is recommended. (In such cases, explicitly state the uncertainty in the comments.)

## Common Patterns
**Example 1:**

**Input:**
```python
def calculate_average(numbers):
  total = sum(numbers)
  return total / len
```

**Output:**
```python
def calculate_average(numbers):
  """Calculates the average of a list of numbers."""
  total = sum(numbers)
  return total / len(numbers) # Fixed: Used len(numbers) instead of len
```

**Example 2:**

**Input:**
```python
for i in range(10)
  print(i)
```

**Output:**
```python
# Added a colon to the end of the for loop statement.
for i in range(10):
  print(i)
```

**Example 3:**

**Input:**
```python
def greet(name):
    print("Hello" + name)

greet("World")
```

**Output:**
```python
def greet(name):
    """Greets the given name."""
    print("Hello " + name) # Added a space after "Hello" for better readability.

greet("World")
```

## Anti-Patterns
- **Overly Aggressive Rewrites:** Avoid completely rewriting the code unless absolutely necessary. Focus on fixing the specific error.
- **Ignoring Context:** Don't make changes without understanding the surrounding code and its intended purpose.
- **Introducing New Errors:** Ensure the fix doesn't create new bugs or unintended side effects.
- **Lack of Explanation:** Always include comments to explain the changes made and the reasoning behind them.
- **Not Addressing Edge Cases:** Don't ignore potential edge cases or vulnerabilities in the original code.
- **Assuming Knowledge:** Do not assume the user knows what the code is supposed to do. Clarify the code's function with comments.

## Quality Checklist
- [x] The corrected code is syntactically correct and runnable.
- [x] The corrected code addresses the identified error or issue.
- [x] The changes are minimally invasive and focused on the core problem.
- [x] Comments explain the changes made and the reasoning behind them.
- [x] Potential edge cases or vulnerabilities are addressed.
- [x] The output is a complete and runnable code snippet.
```