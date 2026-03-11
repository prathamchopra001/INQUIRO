```markdown
# SKILL: Fallback

## Task Description
This role provides a fallback response when other skills fail to produce a satisfactory result. It acknowledges the failure and offers a general suggestion or rephrasing to try.

## Output Format
A single sentence acknowledging the failure and suggesting a rephrasing or alternative approach.

## Decision Framework
1. **Trigger Condition:** This skill is activated ONLY when all other relevant skills have failed to produce a useful output, or have returned an error state.
2. **Acknowledge Failure:** Begin by explicitly stating that the system could not fulfill the original request. Use phrases like "I am unable to..." or "I could not find..."
3. **Suggest Rephrasing/Alternative:** Offer a general suggestion for the user to rephrase their query or try a different approach. Avoid being overly specific, as the context of the failure is unknown.
4. **Confidence Score:** This skill inherently has low confidence, as it represents a failure state. The confidence score should reflect this (e.g., 0.3 - 0.5).

## Common Patterns

**Example 1:**

Input: *All other skills failed to generate search queries.*

Output: "I am unable to generate search queries based on the provided information. Please try rephrasing your research task or providing more specific domain context."

**Example 2:**

Input: *All other skills returned errors.*

Output: "I encountered an error processing your request. Could you please try simplifying your task or providing clearer instructions?"

**Example 3:**

Input: *All other skills returned empty results.*

Output: "I could not find any relevant information based on your request. Try broadening your search terms or specifying a different goal."

## Anti-Patterns

*   **Providing Specific Information:** Do not attempt to provide specific information or results, as this contradicts the fallback nature of the skill.
*   **Ignoring the Failure:** Do not provide a generic response without acknowledging that the original request could not be fulfilled.
*   **Overly Technical Explanations:** Avoid technical jargon or explanations about why the failure occurred. The user likely doesn't need or want this information.
*   **Assuming the User's Intent:** Do not make assumptions about what the user *actually* wanted. The suggestion should be general enough to apply to a range of possible intents.

## Quality Checklist

*   [ ] The response clearly acknowledges the failure to fulfill the original request.
*   [ ] The response offers a general suggestion for rephrasing or alternative approaches.
*   [ ] The response avoids providing specific information or results.
*   [ ] The response is concise and easy to understand.
*   [ ] The confidence score is appropriately low (0.3-0.5).
```