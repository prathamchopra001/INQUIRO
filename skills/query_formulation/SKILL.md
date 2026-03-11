# SKILL: Query Formulation for Academic Literature Search

## Task
Convert a research task into 3-5 academic search queries.

## Output Format
Return ONLY a JSON array of strings. No explanation.

```json
["query 1", "query 2", "query 3"]
```

## Few-Shot Examples

### Example 1: Biodiversity Monitoring
**Task:** Find papers about using sensors to monitor wildlife populations
**Domain terms:** biodiversity, wildlife, monitoring, sensors

**Output:**
```json
[
  "biodiversity monitoring sensor networks",
  "wildlife population tracking IoT devices",
  "acoustic sensors species detection",
  "camera traps automated wildlife census",
  "bioacoustic monitoring bird populations"
]
```

### Example 2: Machine Learning in Healthcare
**Task:** Investigate AI techniques for early diabetes detection
**Domain terms:** diabetes, machine learning, detection, healthcare

**Output:**
```json
[
  "machine learning diabetes prediction models",
  "deep learning early diabetes detection",
  "AI glucose monitoring prediabetes",
  "neural network diabetes diagnosis accuracy",
  "random forest diabetes risk factors"
]
```

### Example 3: Distributed Systems
**Task:** Find papers about observability in microservices architectures
**Domain terms:** observability, microservices, distributed tracing

**Output:**
```json
[
  "observability microservices distributed systems",
  "distributed tracing service mesh monitoring",
  "OpenTelemetry microservices instrumentation",
  "log aggregation kubernetes observability",
  "metrics collection cloud native applications"
]
```

### Example 4: Climate Science
**Task:** Research carbon sequestration methods in forests
**Domain terms:** carbon, sequestration, forests, climate

**Output:**
```json
[
  "forest carbon sequestration rates",
  "tree species carbon storage capacity",
  "reforestation climate mitigation",
  "soil carbon forest ecosystems",
  "biomass carbon accounting forestry"
]
```

## Rules

1. **Include domain terms** - Every query must contain at least one domain-specific term
2. **Be specific** - Use academic vocabulary, not conversational language
3. **Vary specificity** - Mix broad and narrow queries
4. **Keep queries short** - Under 8 words each
5. **Use field terminology** - Framework names, method names, standard terms

## Anti-Patterns (AVOID)

❌ `"study of things related to the topic"` - Too vague
❌ `"an investigation into the relationship between factors and outcomes in the domain"` - Too long
❌ `["query"]` - Only one query (need 3-5)
❌ Queries without domain terms
❌ Plain text instead of JSON array

## Quick Reference

Good queries:
- "Q-learning convergence finite time"
- "biodiversity essential variables framework"
- "OpenTelemetry semantic conventions ecology"

Bad queries:
- "find papers about the topic"
- "research on things"
- "study of various factors"
