# GridWorld Q-Learning Agent

This project implements a Q-Learning agent that learns to navigate through a GridWorld environment, reaching a goal while avoiding obstacles.

## What is Q-Learning?

Q-Learning is a **model-free reinforcement learning algorithm** that enables an agent to learn how to act optimally in an environment by trial and error. The agent doesn't need to know the environment's rules beforehand—it discovers them through experience.

### The Core Idea

The agent maintains a **Q-Table** (quality table) that stores a value for every possible state-action pair. Each Q-value represents:
- **"How good is it to take this action in this state?"**
- The expected cumulative future reward if the agent takes action A in state S and then follows optimal behavior thereafter.

As the agent explores, it continuously updates these Q-values based on rewards it receives, gradually learning which actions lead to the goal.

---

## What Happens During Training?

Let's walk through exactly what the AI agent does step-by-step during training.

### Training Setup

Before training begins, the agent is initialized with:
- A **3×3 grid** (9 possible states)
- **4 possible actions** at each state: up, down, left, right
- An **empty Q-Table** (all values initialized to 0)
  - Shape: 3×3×4 (grid rows × grid columns × actions)

The Q-Table looks like this initially:
```
State (0,0):  [0, 0, 0, 0]  # Actions: up, down, left, right
State (0,1):  [0, 0, 0, 0]
State (0,2):  [0, 0, 0, 0]
... (all 9 states have zeros)
```

All zeros because the agent knows nothing yet—it's a blank slate.

---

### The Training Loop (Episode by Episode)

Each episode represents one complete attempt to reach the goal:

#### Step 1: Reset Environment
The environment places:
- **Agent** at top-left corner: (0, 0)
- **Goal** at bottom-right corner: (2, 2)
- **Walls** in predetermined locations (if maze variant)

```
🤖 . .    Agent at (0,0)
. . .
. . 🏁    Goal at (2,2)
```

#### Step 2: Agent Chooses an Action (Epsilon-Greedy Policy)

The agent uses an **ε-greedy policy** to balance exploration and exploitation:

**With probability ε (10%):**
- Choose a **random action** (exploration)
- Example: Picks "up" even though it might be bad

**With probability 1-ε (90%):**
- Choose the **best known action** (exploitation)
- Look at Q-values for current state, pick action with highest value
- If multiple actions tie, choose randomly among them

**Early in training:**
```
State (0,0): [0, 0, 0, 0]  # All Q-values are equal
Agent picks randomly (all actions equally good)
```

**Later in training:**
```
State (0,0): [-2.5, 5.2, -3.1, 0.4]  # down has highest Q-value
Agent chooses "down" (exploitation)
```

#### Step 3: Execute Action and Observe Result

The agent takes the chosen action in the environment:

**Example:**
- Current state: (0, 0)
- Action: "down"
- New state: (1, 0)
- Reward: -0.1 (small penalty for each step)

The environment provides:
- **Next state**: (1, 0) - where the agent moved
- **Reward**: -0.1 - penalty for taking a step
- **Done**: False - episode continues (not at goal yet)

#### Step 4: Update Q-Table (Learning Happens Here!)

This is where Q-Learning actually happens. The agent updates its Q-value using the **Q-Learning formula**:

```
Q(s, a) = Q(s, a) + α × [r + γ × max(Q(s', a')) - Q(s, a)]
```

**Let's break this down:**

- **Q(s, a)**: Current Q-value for state S, action A (what we currently believe)
- **α (alpha)**: Learning rate (0.1) - how much we update our belief
- **r**: Reward we just received (-0.1)
- **γ (gamma)**: Discount factor (0.9) - how much we value future rewards
- **max(Q(s', a'))**: Best possible Q-value from the NEW state
- **[r + γ × max(Q(s', a'))]**: Our TARGET - what we think the Q-value should be
- **[Target - Q(s, a)]**: The ERROR - how wrong we were
- **Q(s, a) + α × Error**: Update towards target

**Concrete Example:**

```
Current state: (0, 0)
Action taken: down (index 1)
Current Q-value: 0.0
Reward: -0.1
Next state: (1, 0)

Q-values at next state (1, 0): [0, 0, 0, 0]  # All still zeros
max(Q(s', a')) = 0

Target = r + γ × max(Q(s', a'))
Target = -0.1 + 0.9 × 0
Target = -0.1

Error = Target - Current Q-value
Error = -0.1 - 0.0
Error = -0.1

Update = Q(s, a) + α × Error
Update = 0.0 + 0.1 × (-0.1)
Update = -0.01

New Q-value for state (0,0), action "down": -0.01
```

The agent learned: "Taking 'down' from (0,0) isn't great (-0.01)."

---

#### Step 5: Repeat Until Goal or Limit

The agent continues choosing actions, executing them, and learning until either:
- **Reaches the goal** (reward: +10, done: True)
- **Hits step limit** (50 steps, prevents infinite loops)

**When goal is reached:**
```
State: (2, 1)
Action: "right"
New state: (2, 2) - GOAL!
Reward: +10 (large positive reward)
Done: True

Q-values at (2,2) don't matter (terminal state)
Target = 10 (just the goal reward)
Agent learns: "Action 'right' from (2,1) is GREAT!"
```

---

### Episode Completion and Replay

After finishing an episode (success or step limit), the training loop restarts:

**Episode 1:**
- Random exploration, mostly failing to reach goal
- Q-table starts with small negative values
- Agent learns basic patterns (e.g., "moving towards center is better than hitting walls")

**Episode 50:**
- Agent has explored most state-action pairs
- Q-values reflect which actions tend to lead to the goal
- Agent consistently reaches goal in ~5-6 steps

**Episode 200:**
- Agent has converged to optimal policy
- Q-values are stable (not changing much)
- Agent reaches goal in 4 steps (optimal path)

---

## How Q-Table Represents Learned Knowledge

After training, the Q-Table encodes a complete navigation strategy:

**Simple Grid (No Obstacles) - Trained Q-Table:**

```
State (0,0):  [-2.1, 9.4, -2.0, 8.5]  # down is best
State (0,1):  [-1.8, 9.5, 8.5, 9.6]  # right is best
State (0,2):  [9.5, 9.6, 9.5, 10.0]  # down is best
...
```

**Reading the Q-Table:**
- State (0,0): High Q-value for "down" (9.4) - Agent should go down
- State (0,1): High Q-value for "right" (9.6) - Agent should go right
- State (0,2): High Q-value for "down" (9.6) - Agent should go down

These values encode: **"Taking this action from this state will likely lead to the goal with ~9-10 cumulative reward."**

---

## How the Agent Decides After Training

Once trained, the agent uses its Q-Table to navigate:

```
Current state: (0, 0)
Q-values: [-2.1, 9.4, -2.0, 8.5]
Best action: "down" (highest Q-value)

Move to (1, 0)

Current state: (1, 0)
Q-values: [-1.5, 9.3, -2.2, 8.8]
Best action: "down" (highest Q-value)

Move to (2, 0)

Current state: (2, 0)
Q-values: [-1.9, 8.9, -2.1, 9.5]
Best action: "right" (highest Q-value)

Move to (2, 1)

Current state: (2, 1)
Q-values: [-1.8, 9.2, -2.0, 10.0]
Best action: "right" (highest Q-value)

Move to (2, 2) - GOAL!
```

Total path: down → down → right → right (4 steps)
Total reward: 10 - 4×0.1 = 9.6

The agent executes optimal policy every time (no exploration needed).

---

## Key Components Explained

### Q-Table Structure
```
Q-Table: [row][col][action]

Dimensions:
- row: 0 to 2 (3 rows)
- col: 0 to 2 (3 columns)
- action: 0 to 3 (up, down, left, right)

Total entries: 3 × 3 × 4 = 36 Q-values
```

### Learning Rate (α = 0.1)
Controls how quickly the agent updates its beliefs:
- **High α (0.9)**: Learns quickly but may be unstable
- **Low α (0.01)**: Learns slowly but stably
- **Our choice (0.1)**: Balanced approach

### Discount Factor (γ = 0.9)
Controls how much the agent values future rewards:
- **High γ (0.99)**: Agent cares deeply about long-term rewards
- **Low γ (0.1)**: Agent focuses on immediate rewards
- **Our choice (0.9)**: Values future rewards but not exclusively

### Exploration Rate (ε = 0.1)
Controls exploration vs exploitation:
- **10% of the time**: Random action (discover new strategies)
- **90% of the time**: Best known action (use learned knowledge)

---

## Visualizing Learning Progress

### Before Training (Episode 0)
```
🤖 . .
. . .
. . 🏁

Agent's plan: Random guesses
Success rate: ~0%
```

### During Training (Episode 50)
```
🤖 . .
. . .
. . 🏁

Agent's plan: "I know going down and right usually works"
Success rate: ~80%
```

### After Training (Episode 200)
```
🤖 . .
. . .
. . 🏁

Agent's plan: "Down twice, then right twice. Always."
Success rate: 100%
Average steps: 4 (optimal)
```

---

## Why This Works

### Exploration-Exploitation Balance
The agent needs to:
1. **Explore** (try new actions) to discover what works
2. **Exploit** (use known good actions) to achieve goals

ε-greedy provides this balance automatically.

### Temporal Difference Learning
The agent updates Q-values based on immediate rewards AND estimated future rewards. This enables learning without waiting for episodes to complete.

### Convergence Guarantees
For finite Markov Decision Processes (like our GridWorld), Q-Learning is **mathematically guaranteed** to converge to optimal Q-values given sufficient exploration and appropriate learning rates.

---

## Files

- **gridworld.py** - GridWorld environment (the "world" the agent explores)
- **agent.py** - QLearningAgent class (the "brain" that learns)
- **train.py** - Training script (orchestrates learning)
- **maze.py** - Maze variants + comparative experiments
- **example_usage.py** - Demo of basic GridWorld functionality
- **test_agent.py** - Quick test of agent learning
- **EXPERIMENT_RESULTS.md** - Detailed analysis of maze experiments

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simple GridWorld demo
python example_usage.py

# Train agent on 5×5 grid
python train.py

# Run maze experiments
python maze.py

# Quick test of agent
python test_agent.py
```

---

## Learning Resources

**Q-Learning:**
- [Sutton & Barto Book (Chapter 6)](http://www.incompleteideas.net/book/RLbook2020.pdf)
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

**Reinforcement Learning:**
- [David Silver's RL Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [Stanford CS234](http://cs234.stanford.edu/)

---

## Summary

During training, the Q-Learning agent:
1. **Explores** the environment through random actions (ε-greedy)
2. **Experiences** states, actions, rewards, and next states
3. **Updates** its Q-table using the Q-Learning formula
4. **Gradually learns** which actions lead to the goal
5. **Converges** to an optimal policy (4-step path)

The agent starts knowing nothing (Q-table full of zeros) and ends with complete knowledge of how to navigate the grid optimally. This entire learning process is automatic—no human intervention needed!
