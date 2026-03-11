```markdown
# SKILL: schema_design

## Task Description
This skill focuses on designing effective and well-structured schemas for various data types, ensuring data integrity and facilitating efficient data processing. It involves defining the structure, data types, and constraints for datasets.

## Output Format
Output should be a well-formatted JSON schema definition, including:
*   `type`: Data type (e.g., "object", "string", "number", "array", "boolean")
*   `properties`: (For objects) A dictionary of property names and their schema definitions.
*   `items`: (For arrays) The schema definition for the array's elements.
*   `required`: (For objects) A list of required property names.
*   `description`: A brief description of the field.
*   `enum`: (Optional) A list of allowed values for a field.
*   `format`: (Optional) A string indicating the expected format (e.g., "date", "email").

```json
{
  "type": "object",
  "properties": {
    "field1": {
      "type": "string",
      "description": "Description of field1"
    },
    "field2": {
      "type": "number",
      "description": "Description of field2"
    }
  },
  "required": ["field1", "field2"]
}
```

## Decision Framework
1.  **Understand the Data**: Analyze the data requirements and the purpose of the schema. Identify key entities, attributes, and relationships.
2.  **Define Data Types**: Choose appropriate data types for each attribute (string, number, boolean, array, object). Consider using more specific formats (e.g., date, email) when applicable.
3.  **Structure the Schema**: Organize the data into a logical structure (e.g., objects with properties, arrays of items). Use nested structures when necessary.
4.  **Define Constraints**: Specify constraints on the data, such as required fields, allowed values (enums), and data validation rules.
5.  **Consider Extensibility**: Design the schema to be flexible and extensible to accommodate future changes and additions.
6.  **Confidence Scoring**: Assign a confidence score based on the completeness, accuracy, and relevance of the schema. Higher scores for schemas that meet all requirements and are well-structured. Lower scores for incomplete or inaccurate schemas.
7.  **Edge Case Handling**: Address potential edge cases and error conditions, such as missing data, invalid data types, and unexpected values. Implement appropriate validation and error handling mechanisms.

## Common Patterns
**Example 1: User Profile Schema**
```json
{
  "type": "object",
  "properties": {
    "userId": {
      "type": "string",
      "description": "Unique identifier for the user"
    },
    "username": {
      "type": "string",
      "description": "User's username"
    },
    "email": {
      "type": "string",
      "format": "email",
      "description": "User's email address"
    },
    "age": {
      "type": "integer",
      "description": "User's age"
    }
  },
  "required": ["userId", "username", "email"]
}
```

**Example 2: Product Schema**
```json
{
  "type": "object",
  "properties": {
    "productId": {
      "type": "string",
      "description": "Unique identifier for the product"
    },
    "name": {
      "type": "string",
      "description": "Product name"
    },
    "description": {
      "type": "string",
      "description": "Product description"
    },
    "price": {
      "type": "number",
      "description": "Product price"
    },
    "category": {
      "type": "string",
      "description": "Product category"
    }
  },
  "required": ["productId", "name", "price"]
}
```

## Anti-Patterns
*   **Oversimplification**: Avoid schemas that are too simple and do not capture the necessary data.
*   **Overcomplication**: Avoid schemas that are too complex and difficult to understand and maintain.
*   **Missing Data Types**: Ensure that all attributes have appropriate data types defined.
*   **Lack of Constraints**: Include necessary constraints, such as required fields and allowed values, to ensure data integrity.
*   **Inconsistent Naming**: Use consistent naming conventions for attributes.
*   **Ignoring Edge Cases**: Fail to consider potential edge cases and error conditions.
*   **Invalid JSON**: Producing invalid JSON schema. Always validate the output.

## Quality Checklist
*   [x] The schema accurately reflects the data requirements.
*   [x] All attributes have appropriate data types defined.
*   [x] Necessary constraints are included to ensure data integrity.
*   [x] The schema is well-structured and easy to understand.
*   [x] The schema is flexible and extensible to accommodate future changes.
*   [x] The schema handles potential edge cases and error conditions.
*   [x] The JSON is valid.
```