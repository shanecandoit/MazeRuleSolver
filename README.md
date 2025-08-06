# Rule-Based Maze Solver

This project uses an evolutionary algorithm to generate and refine a set of rules for a maze-solving agent. The rules are represented by an ensemble of simple decision trees. The system evaluates these rule sets based on agent performance in a series of mazes and evolves the best-performing rules over time.

A maze tile is either a wall or a free space. The agent can move in four directions: north, south, east, and west. The goal is to navigate from a starting point to an endpoint while avoiding walls. All free spaces are considered unvisited until the agent moves into them.

---

## Components

### 1. Rule Representation (Decision Trees)

Each rule consists of a set of conditions that check the agent's local environment and an associated action.

- **Conditions**: Each rule has a "left side" which consists of two relative positions (e.g., agent's current position, position to the north). Each position is checked for its state (e.g., 'wall', 'free', 'unvisited'). A rule matches if its conditions are met by the agent's current situation.
- **Action**: The action the agent should take if the rule matches (e.g., `north`, `south`, `east`, `west`).

### 2. Ensembles and Voting

An ensemble is a collection of these rules, forming a complete strategy for a single agent.

- **Ensemble**: An agent is equipped with an ensemble of rules.
- **Voting**: When an agent needs to make a move, it evaluates all rules in its ensemble. Each rule that matches the agent's current situation casts a vote for its associated action.
- **Decision**: The action that receives the majority of votes from the matching rules is the action the agent takes. This allows for more complex strategies by combining multiple simple rules.

---

## System Workflow

### 1. Initialization

- A large pool of random rules is created.
- 100 ensembles are formed by selecting rules from this pool. Each ensemble is assigned to a unique agent.
- The initial maze is a 10x10 grid.

### 2. Evaluation

- Each of the 100 agents attempts to solve the maze using its assigned ensemble of rules.
- The performance of each agent is measured by:
  - Average distance from start: A higher value indicates the agent is exploring the maze effectively.
  - Final distance from end: A lower value indicates success in reaching the goal.
- These metrics are combined to create a single performance score for each ensemble.

### 3. Evolution (Blending and Synthesis)

- This is the core of the learning process. The system uses an evolutionary algorithm to improve the rule sets.
- Selection: Ensembles with high performance scores are selected to be "parents" for the next generation.
- Crossover: New "child" ensembles are created by combining rules from two or more parent ensembles. This process blends successful strategies.
- Mutation: Random changes are introduced to individual rules within the ensembles. This can involve altering conditions or actions. Mutation introduces new strategies and prevents the system from getting stuck in local optima.
- Replacement: The worst-performing ensembles are replaced by the new child ensembles, ensuring that the population of rule sets constantly improves.
- Elitism: A small percentage of the best-performing ensembles are retained unchanged to preserve successful strategies.

### 4. Iteration and Scaling

The evaluation and evolution steps are repeated for many generations. As the agents become more proficient at solving the 10x10 mazes, the difficulty increases by making the mazes larger. This forces the system to develop more generalizable rules rather than rules specific to a single maze layout.

## Swarm

A swarm is a collection of agents that can evaluate agent and rules in relative terms.

## License

AGPL-3.0 +

## Rust

This project also has a Rust implementation that uses the same principles but is (will be) designed for performance and efficiency. The Rust version can be found in the `maze-rules-rust` directory.
