```markdown
# SKILL: Dataset Search

## Task Description
This skill is designed to efficiently locate relevant, publicly available datasets based on a given research objective. It prioritizes datasets that directly address the problem statement and provide data suitable for analysis and model training.

## Output Format
A JSON object containing a list of dataset recommendations. Each recommendation includes the dataset name, a brief description, the source URL, and a relevance score (0-100).

```json
{
  "datasets": [
    {
      "name": "Dataset Name 1",
      "description": "Brief description of the dataset and its relevance.",
      "source_url": "URL of the dataset.",
      "relevance_score": 85
    },
    {
      "name": "Dataset Name 2",
      "description": "Brief description of the dataset and its relevance.",
      "source_url": "URL of the dataset.",
      "relevance_score": 70
    }
  ]
}
```

## Decision Framework

1. **Keyword Extraction**: Extract key terms and concepts from the research objective. Focus on the problem being addressed, the data types needed, and any specific methodologies mentioned.
2. **Dataset Source Prioritization**: Prioritize searching reputable dataset repositories like Kaggle, UCI Machine Learning Repository, Google Dataset Search, and domain-specific databases (e.g., Protein Data Bank for protein-related data).
3. **Relevance Assessment**:  Evaluate each dataset based on:
    *   **Directness:** How directly does the dataset address the research problem? (Weight: 50%)
    *   **Data Type:** Does the dataset contain the necessary data types (e.g., sequences, properties, experimental results)? (Weight: 30%)
    *   **Accessibility:** Is the dataset publicly available and easily accessible? (Weight: 20%)
4. **Relevance Scoring**: Assign a relevance score (0-100) based on the above criteria.
    *   80-100: Highly relevant, directly addresses the problem, contains necessary data types, and is easily accessible.
    *   60-79: Moderately relevant, may address a related problem or contain a subset of the required data.
    *   40-59: Potentially relevant, requires significant processing or may only provide indirect information.
    *   Below 40: Not relevant.
5. **Edge Case Handling**:
    *   If no directly relevant datasets are found, consider datasets that address related problems or contain proxy data. Lower the relevance score accordingly.
    *   If a dataset requires significant processing or cleaning, note this in the description and adjust the relevance score.
    *   If a dataset is behind a paywall or requires special access, mention this clearly and lower the relevance score.

## Common Patterns

**Example 1:**

*   **Input:** Research Objective: "Develop a model to predict customer churn for a telecommunications company."
*   **Output:**
```json
{
  "datasets": [
    {
      "name": "Telco Customer Churn Dataset",
      "description": "Customer churn data from a telecommunications company, including demographic information, service usage, and churn status.",
      "source_url": "https://www.kaggle.com/blastchar/telco-customer-churn",
      "relevance_score": 95
    }
  ]
}
```

**Example 2:**

*   **Input:** Research Objective: "Identify datasets for training a machine learning model to classify images of different species of flowers."
*   **Output:**
```json
{
  "datasets": [
    {
      "name": "102 Category Flower Dataset",
      "description": "A large dataset of images of 102 different flower species.",
      "source_url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
      "relevance_score": 90
    },
    {
      "name": "Flowers Recognition",
      "description": "A dataset of flower images categorized into 5 classes: daisy, dandelion, rose, sunflower, tulip.",
      "source_url": "https://www.kaggle.com/alxmamaev/flowers-recognition",
      "relevance_score": 80
    }
  ]
}
```

## Anti-Patterns

*   **Listing irrelevant datasets:** Do not include datasets that have little to no connection to the research objective.
*   **Fabricating relevance scores:**  Ensure the relevance score accurately reflects the dataset's suitability for the task.
*   **Missing source URLs:** Always provide a valid URL for each dataset.
*   **Overly general descriptions:** Provide specific details about the dataset's contents and relevance.
*   **Ignoring data accessibility:**  Do not recommend datasets that are not publicly available without clearly stating the access restrictions.

## Quality Checklist

*   Are all datasets relevant to the research objective?
*   Are the descriptions accurate and informative?
*   Are the source URLs valid and accessible?
*   Are the relevance scores justified based on the dataset's characteristics?
*   Are any access restrictions clearly stated?
```