---
task_hash: dcc8f737ea7e
generated: 2026-03-07T20:17:03.429193
techniques: ['Agent-Based Modeling (ABM)', 'Statistical Analysis', 'Simulation']
libraries: ['Mesa', 'NumPy', 'Pandas', 'Matplotlib']
source: llm_analysis
---

```markdown
# Skill: ABM with Coordinated Pricing

## Task Description
Simulate an Agent-Based Model (ABM) where a small percentage of firms coordinate their pricing strategy (e.g., a cartel) and analyze the resulting impact on market prices and profitability of other (learning) firms.

## Required Packages
```bash
pip install mesa numpy pandas matplotlib
```

## Correct Imports
```python
import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## DO NOT USE
*   Avoid overly complex agent logic initially; focus on core ABM structure.
*   Don't use global variables for agent-specific data.
*   Do not use fixed random seeds during parameter sweeps.

## Example Code
```python
import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Firm(mesa.Agent):
    def __init__(self, unique_id, model, coordinated=False):
        super().__init__(unique_id, model)
        self.price = np.random.uniform(10, 20)
        self.coordinated = coordinated
        self.profit = 0

    def step(self):
        if self.coordinated:
            self.price = self.model.cartel_price #Example Cartel Price
        else:
            #Simple random price adjustment
            self.price += np.random.uniform(-1, 1)
            self.price = max(1, self.price) #Ensure price is positive

        #Calculate Profit (Dummy calculation)
        demand = max(0, 100 - self.price)
        self.profit = self.price * demand

class MarketModel(mesa.Model):
    def __init__(self, num_firms, cartel_percentage=0.05, cartel_price = 25):
        self.num_firms = num_firms
        self.cartel_percentage = cartel_percentage
        self.cartel_price = cartel_price
        self.schedule = mesa.space.MultiGrid(width=10, height=10, torus=False) #Correct usage
        self.grid = mesa.time.SimultaneousActivation(self)
        self.running = True

        # Create agents
        num_cartel_firms = int(num_firms * cartel_percentage)
        for i in range(num_firms):
            coordinated = i < num_cartel_firms #First firms are coordinated
            firm = Firm(i, self, coordinated)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(firm, (x,y))
            self.grid.add(firm)

        self.datacollector = mesa.DataCollector(
            model_reporters={"AveragePrice": lambda m: np.mean([a.price for a in m.grid.get_all_cell_contents()])},
            agent_reporters={"Price": "price", "Profit": "profit"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.grid.step() #Correct step
        if self.schedule.steps > 100:
            self.running = False

#Run the model
model = MarketModel(num_firms=20, cartel_percentage=0.2, cartel_price=25)
model.run_model()

# Collect and analyze data
results = model.datacollector.get_model_vars_dataframe()
print(results.head())

agent_results = model.datacollector.get_agent_vars_dataframe()
print(agent_results.head())

#Basic Plotting Example
plt.plot(results["AveragePrice"])
plt.xlabel("Step")
plt.ylabel("Average Price")
plt.title("Average Market Price Over Time")
plt.show()
```

## Common Pitfalls
*   **Incorrect Scheduler:** Using `mesa.time.RandomActivation` or `mesa.time.StagedActivation` when `mesa.time.SimultaneousActivation` is intended.
*   **Data Collection Errors:** Forgetting to use `self.datacollector.collect(self)` in the model's `step` function.
*   **Agent Placement:** Using `model.grid.place_agent` to place agents directly without considering the grid size.
*   **Cartel Logic:**  Failing to update the cartel price and having firms change prices at different times.
*   **Infinite Loops:** Ensure `self.running = False` condition is met to stop the simulation.
*   **Index Errors**: When accessing agents using `model.grid.get_cell_list_contents([(x,y)])` make sure the coordinates exist.
*   **Missing Step**: Forgetting to call `self.grid.step()` or `self.schedule.step()` in the model.