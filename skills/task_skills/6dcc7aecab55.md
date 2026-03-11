---
task_hash: 6dcc7aecab55
generated: 2026-03-07T17:10:41.414754
techniques: ['Q-learning', 'Sensitivity Analysis', 'Agent-Based Modeling (ABM)', 'Statistical Analysis']
libraries: ['NumPy', 'Pandas', 'Matplotlib/Seaborn', 'potentially custom ABM library (if not built from scratch)']
source: llm_analysis
---

```markdown
# Skill: Q-learning Sensitivity Analysis in ABM

## Task Description
Conduct a sensitivity analysis of Q-learning parameters (learning rate, discount factor) within an Agent-Based Model (ABM) to evaluate convergence and stability of pricing strategies.

## Required Packages
```bash
pip install numpy pandas matplotlib seaborn
```

## Correct Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
```

## DO NOT USE
*   Avoid using `eval()` or `exec()` for ABM logic.
*   Don't hardcode environment parameters within the Q-learning agent.
*   Do not use global variables excessively; encapsulate ABM components within classes.

## Example Code (Minimal Working Example)
```python
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)  # Explore
        else:
            return np.argmax(self.Q[state, :])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state, :])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

# Dummy ABM Environment (replace with your actual ABM)
class DummyEnvironment:
    def __init__(self, n_states=5, n_actions=3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.current_state = 0

    def step(self, action):
        reward = random.uniform(0, 1) # Simulate reward
        self.current_state = random.randint(0, self.n_states-1) # Simulate next state
        return self.current_state, reward

# Sensitivity Analysis Loop
learning_rates = [0.1, 0.3, 0.5]
discount_factors = [0.7, 0.9, 0.99]
n_episodes = 100

results = {}

for lr in learning_rates:
    for gamma in discount_factors:
        env = DummyEnvironment()
        agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=lr, discount_factor=gamma)
        rewards_per_episode = []

        for episode in range(n_episodes):
            state = env.current_state
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            rewards_per_episode.append(reward)

        results[(lr, gamma)] = rewards_per_episode

# Basic plotting example
for params, rewards in results.items():
    plt.plot(rewards, label=f"LR={params[0]}, Gamma={params[1]}")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-learning Sensitivity Analysis")
plt.legend()
plt.show()
```

## Common Pitfalls
*   **Incorrect State/Action Space Definition:** Ensure `n_states` and `n_actions` accurately reflect the ABM.
*   **Exploration-Exploitation Imbalance:** Tune `epsilon` for appropriate exploration.
*   **Non-Stationary Environment:** ABMs can be non-stationary; adaptive learning rates might be required.
*   **Insufficient Episodes:** Ensure enough episodes for convergence. Monitor Q-values for stability.
*   **Ignoring ABM Complexity**: The dummy example must be replaced by a real interaction with the ABM, which will require careful management of information passed to the Q-learning agent, and the correct interpretation of actions to influence the agent's pricing strategy.
*   **Lack of Statistical Analysis**: The provided plotting is a basic example.  Use statistical measures (e.g., mean, variance, confidence intervals) to quantify convergence and stability.