# Deep Q-Network (DQN) Design Documentation

## Overview

The Deep Q-Network (DQN) represents a fundamental evolution from tabular Q-Learning to function approximation using neural networks. This document explains the design decisions, architecture, and implementation details.

---

## Why DQN? The Problem with Tabular Q-Learning

### Tabular Q-Learning Limitations

**Memory Explosion:**
- Q-Table size = num_states × num_actions
- For 3×3 GridWorld: 9 × 4 = 36 entries ✓
- For 10×10 GridWorld: 100 × 4 = 400 entries ✓
- For continuous state spaces: Infinite entries ✗

**No Generalization:**
- Each state-action pair must be visited to learn its value
- Visiting state (0,0) doesn't help with state (0,1)
- Must experience every state individually

**Sample Inefficiency:**
- Learning from consecutive samples creates high correlation
- Can get stuck in local optima

---

## DQN Solution: Neural Network Function Approximation

### Core Idea

Instead of storing Q-values in a table, use a neural network to **approximate** the Q-function:

```
Q(s, a) ≈ NeuralNetwork(s)[a]
```

**Key Benefits:**

1. **Generalization:** Network learns patterns (e.g., "moving towards goal is good") that apply to unseen states
2. **Memory Efficiency:** Network weights (thousands of parameters) vs full table (millions of entries for large spaces)
3. **Continuous State Spaces:** Can handle continuous inputs (e.g., robot positions, velocities)
4. **Transfer Learning:** Knowledge can transfer between similar environments

---

## Neural Network Architecture

### Input Layer

**State Representation:**

We encode state as a 4-dimensional vector: `[x, y, gx, gy]`

- **x, y**: Agent's current position (row, column)
- **gx, gy**: Goal's position (row, column)

**Design Choice:** Providing goal position explicitly helps the network learn "distance to goal" concept. Alternative approaches:
- Relative position: `[gx-x, gy-y]` - Distance from goal
- One-hot grid: 9-dimensional vector for 3×3 grid
- Only agent position: `[x, y]` - Network must infer goal through rewards

### Hidden Layers

**Architecture:**
```
Input (4) → FC (64) → ReLU → FC (64) → ReLU → Output (4)
```

**Layer Details:**
- **FC1**: Linear transformation 4 → 64
- **ReLU1**: Rectified Linear Unit activation
- **FC2**: Linear transformation 64 → 64
- **ReLU2**: Rectified Linear Unit activation

**Why 64 Neurons?**
- Small enough to learn quickly
- Large enough to capture complex relationships
- Balances capacity vs training time for this problem

**Why ReLU Activation?**
- Computationally efficient
- Prevents vanishing gradients
- Non-linear (allows learning complex patterns)

### Output Layer

**Output:** 4-dimensional vector `[Q_up, Q_down, Q_left, Q_right]`

- **No activation function** - We want raw Q-values (not probabilities)
- Each neuron corresponds to one action
- Argmax of output gives best action

---

## Key Innovations

### 1. Experience Replay

**Problem:** Learning from consecutive samples creates correlation, leading to unstable training.

**Solution:** Store experiences in a buffer and sample random batches.

**Implementation:**
```python
# Store experience
replay_buffer.push(state, action, reward, next_state, done)

# During training, sample random batch
batch = replay_buffer.sample(batch_size=64)
```

**Benefits:**
- Breaks temporal correlation between samples
- Enables reuse of past experiences
- More data-efficient learning
- More stable training

**Buffer Details:**
- **Capacity**: 10,000 experiences
- **Data Structure**: Deque (removes oldest when full)
- **Sampling**: Uniform random (no priority)

### 2. Target Network

**Problem:** Using the same network to predict target creates "moving target" - Q-values chase their own predictions.

**Solution:** Maintain two identical networks:
- **Policy Network**: Updated every step, used for action selection
- **Target Network**: Frozen copy, used for target Q-value calculation

**Implementation:**
```python
# Use policy network for action selection
action = select_action(policy_net, state)

# Use target network for target calculation
next_q_values = target_net(next_state)
target = reward + gamma * max(next_q_values)

# Update only policy network
update_policy_network(policy_net, target)
```

**Target Network Updates:**
- **Frequency**: Every 100 training steps
- **Method**: Hard copy of policy network weights
- **Alternative**: Soft updates (polyak averaging)

**Benefits:**
- Stabilizes Q-value targets
- Prevents oscillations/divergence
- Enables reliable convergence

---

## DQN Agent Components

### DQNAgent Class Structure

```python
class DQNAgent:
    def __init__(self, state_size, action_size, actions, ...):
        # Policy network (action selection)
        self.policy_net = DQN(state_size, action_size)

        # Target network (stable targets)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters())

        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def choose_action(self, state, training=True):
        # Epsilon-greedy action selection
        # Uses policy network for Q-values
        pass

    def learn(self, batch):
        # Sample batch from replay buffer
        # Calculate target Q-values using target network
        # Train policy network
        pass

    def update_target_network(self):
        # Copy policy weights to target
        pass
```

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|---------|----------|
| `learning_rate` | 0.001 | Controls optimization step size (lower than Q-learning due to batch training) |
| `gamma` | 0.99 | Discount factor (slightly higher than Q-learning's 0.9) |
| `exploration_rate` | 1.0 | Initial epsilon (100% exploration) |
| `exploration_decay` | 0.995 | Epsilon decay per episode |
| `exploration_min` | 0.01 | Minimum exploration (1%) |
| `batch_size` | 64 | Number of experiences per training step |
| `replay_buffer_size` | 10,000 | Maximum stored experiences |
| `target_update_frequency` | 100 | Steps between target network updates |
| `hidden_size` | 64 | Neurons in hidden layers |

---

## Training Algorithm (Not Yet Implemented)

### The Full DQN Training Loop

```python
# For each episode
for episode in range(num_episodes):
    state = env.reset()

    while not done:
        # 1. Choose action (epsilon-greedy)
        action = agent.choose_action(state, training=True)

        # 2. Execute action
        next_state, reward, done = env.step(action)

        # 3. Store experience
        agent.replay_buffer.push(state, action, reward, next_state, done)

        # 4. Train if buffer has enough samples
        if len(agent.replay_buffer) >= agent.batch_size:
            batch = agent.replay_buffer.sample(agent.batch_size)
            agent.learn(batch)

        state = next_state

    # Decay exploration rate
    agent.update_exploration_rate()
```

### The Learn Method (To Be Implemented)

```python
def learn(self, batch):
    # Prepare batch tensors
    states = torch.FloatTensor([e.state for e in batch])
    actions = torch.LongTensor([e.action for e in batch])
    rewards = torch.FloatTensor([e.reward for e in batch])
    next_states = torch.FloatTensor([e.next_state for e in batch])
    dones = torch.FloatTensor([e.done for e in batch])

    # Current Q-values
    current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

    # Target Q-values using target network
    next_q_values = self.target_net(next_states).max(1)[0].detach()
    expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

    # Loss calculation
    loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))

    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Periodically update target network
    self.step_count += 1
    if self.step_count % self.target_update_frequency == 0:
        self.update_target_network()
```

---

## Comparison: Q-Learning vs DQN

| Aspect | Q-Learning (Tabular) | DQN (Neural Network) |
|---------|------------------------|--------------------------|
| **Storage** | Q-Table (num_states × num_actions) | Network weights (fixed size) |
| **State Access** | Exact lookup O(1) | Forward pass O(parameters) |
| **Generalization** | None (must visit each state) | Learns patterns across states |
| **Continuous States** | Not possible | Supported |
| **Training** | Single sample per step | Batch of samples per step |
| **Stability** | Can oscillate | More stable with target network |
| **Memory** | Efficient for small spaces | Efficient for large/continuous spaces |
| **Implementation** | Simple | Complex (requires PyTorch/TF) |

---

## Current Implementation Status

### ✅ Completed

1. **Neural Network Architecture** (`DQN` class)
   - Input: 4-dimensional state
   - Hidden: 64 neurons (2 layers)
   - Output: 4 Q-values

2. **Experience Replay Buffer** (`ReplayBuffer` class)
   - Stores up to 10,000 experiences
   - Random batch sampling
   - Fixed-size deque

3. **Agent Skeleton** (`DQNAgent` class)
   - Policy and target networks initialized
   - Epsilon-greedy action selection
   - Hyperparameters configured
   - Device support (CPU/GPU)

4. **State Encoding** (`StateEncoder` class)
   - Converts tuples to 4D vectors
   - Includes agent and goal positions

5. **Testing**
   - All modules tested successfully
   - Forward pass verified
   - Action selection working

### ⏳ Pending

1. **Learn Method**
   - Batch training implementation
   - Target network integration
   - Loss calculation and optimization

2. **Training Script**
   - Full DQN training loop
   - Integration with GridWorld
   - Hyperparameter tuning

3. **Performance Analysis**
   - Compare DQN vs Q-Learning
   - Analyze generalization capability
   - Visualize learned features

---

## Design Decisions & Rationale

### Why PyTorch?

- **Research Standard**: Widely used in RL research
- **Dynamic Graphs**: Flexible, pythonic API
- **Debugging**: Excellent error messages
- **Community**: Large ecosystem and documentation

### Why Include Goal Position in State?

**Option A:** `[x, y]` - Only agent position
- Network must learn goal location through rewards
- Slower initial learning
- Less explicit

**Option B:** `[x, y, gx, gy]` - Agent + goal positions
- Network knows where goal is
- Faster initial learning
- More explicit state

**Choice:** Option B. For demonstration, explicit goal position helps network learn faster and makes learned features more interpretable.

### Why Separate Target Network?

**Alternative:** Single network (DDQN - Double DQN)
- More complex implementation
- Better for some problems
- Harder to explain/understand

**Choice:** Classic DQN with target network. Simpler, stable, good for demonstration.

---

## Expected Behavior

### Early Training (Episodes 0-100)
- High exploration (ε = 1.0)
- Random actions dominate
- Q-values random/uninformed
- Target network matches policy (early updates)

### Mid Training (Episodes 100-500)
- Exploration decays (ε ≈ 0.6)
- Network starts learning patterns
- Target network stabilizes
- Occasional goal-reaching

### Late Training (Episodes 500+)
- Low exploration (ε ≈ 0.01)
- Network near convergence
- Consistent goal-reaching
- Q-values stable

### After Training (Evaluation)
- Zero exploration (ε = 0)
- Greedy action selection
- Optimal policy execution
- Generalization to unseen states

---

## Next Steps

1. **Implement Learn Method**
   - Batch sampling from replay buffer
   - Target Q-value calculation
   - Loss computation and backpropagation

2. **Create Training Script**
   - Full training loop with GridWorld
   - Metrics tracking (loss, rewards, steps)
   - Checkpoint saving

3. **Compare with Q-Learning**
   - Same environment, different "brains"
   - Learning curves comparison
   - Generalization analysis

4. **Visualization**
   - Plot learning curves
   - Visualize network weights
   - Analyze learned features

---

## Resources

**Original Paper:**
- Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- Introduced DQN to Atari games

**PyTorch RL:**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)

**DQN Variants:**
- Double DQN: Addresses overestimation bias
- Dueling DQN: Separates state value and action advantage
- Prioritized Replay: Samples important experiences more frequently

---

## Summary

DQN represents a fundamental shift from tabular to neural network-based reinforcement learning. By using a neural network to approximate Q-values, implementing experience replay to stabilize training, and maintaining a target network for reliable convergence, DQN enables:

1. **Generalization** across states
2. **Scalability** to large/continuous spaces
3. **Stability** through decorrelated samples
4. **Flexibility** for complex problems

This implementation provides the foundation for training DQN agents on GridWorld environments, with the learn method and training script as the next major milestones.
