```markdown
# SKILL: Query Classification

## Task Description
This skill classifies user queries into predefined categories based on the complexity and required resources to answer them. The goal is to route queries to the appropriate processing pipeline.

## Output Format
The output should be a single string representing the category. Possible categories are: `SIMPLE`, `RESEARCH`.

## Decision Framework
1. **Initial Assessment:** Quickly scan the query for keywords and phrases indicating complexity or research intent.
2. **SIMPLE Classification:**
   - **Criteria:** The query can be answered directly from readily available general knowledge. The answer should be concise (1-3 sentences).
   - **Keywords:** "What is...", "Who is...", "When did...", "Define..."
   - **Confidence:** High confidence if the query fits the criteria and uses simple vocabulary.
3. **RESEARCH Classification:**
   - **Criteria:** The query requires investigation, literature review, data analysis, or in-depth exploration to provide a comprehensive answer. It often involves identifying factors, analyzing relationships, or exploring complex topics.
   - **Keywords:** "Investigate...", "Analyze...", "Compare...", "What are the factors...", "What is the relationship between..."
   - **Confidence:** High confidence if the query necessitates going beyond readily available knowledge.
4. **Edge Case Handling:**
   - If a query is ambiguous, lean towards `RESEARCH`. It's better to over-classify complexity than to under-classify.
   - If a query has multiple parts, and at least one part requires research, classify as `RESEARCH`.
5. **Confidence Scoring:** Not explicitly scored, but used internally to guide the classification. High confidence means the query clearly aligns with the category's criteria.

## Common Patterns
**Example 1:**
*   **Input:** "What is the capital of France?"
*   **Output:** `SIMPLE`

**Example 2:**
*   **Input:** "Investigate the impact of social media on teenage mental health."
*   **Output:** `RESEARCH`

**Example 3:**
*   **Input:** "What is the function of mitochondria, and how does mitochondrial dysfunction contribute to neurodegenerative diseases?"
*   **Output:** `RESEARCH` (because the second part of the question requires research)

## Anti-Patterns
1. **Misclassifying complex questions as simple:** Avoid classifying queries requiring detailed explanations or multiple sources as `SIMPLE`.
2. **Overclassifying simple questions as research:** Avoid classifying basic factual questions as `RESEARCH` simply because they are related to a complex topic.
3. **Returning invalid categories:** Do not return any categories other than `SIMPLE` or `RESEARCH`.

## Quality Checklist
1.  Does the classification align with the complexity of the query?
2.  Is the answer to the query readily available general knowledge, or does it require research?
3.  Does the classification use only the allowed categories (`SIMPLE`, `RESEARCH`)?
4.  If the query has multiple parts, is the most complex part used for classification?
```