"""
Deep Q-Network (DQN) Agent Implementation

This module implements a Deep Q-Network agent using PyTorch, which replaces
the tabular Q-table with a neural network that learns to approximate Q-values.

Key innovations over basic Q-Learning:
1. Experience Replay: Stores experiences in memory and samples random batches for training
2. Target Network: Separate network with frozen weights for stable target computation
3. Function Approximation: Neural network generalizes across states
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple


# Define a named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(nn.Module):
    """Deep Q-Network neural network architecture.

    A simple feed-forward neural network that maps states to Q-values for each action.
    Uses two hidden layers with ReLU activation.

    Architecture:
    Input: 4-dimensional vector [x, y, gx, gy]
    Hidden Layer 1: 64 neurons
    Hidden Layer 2: 64 neurons
    Output: 4-dimensional Q-values (one per action)
    """

    def __init__(self, state_size, action_size, hidden_size=64):
        """Initialize the DQN neural network.

        Args:
            state_size (int): Dimension of state vector (default: 4 for [x, y, gx, gy])
            action_size (int): Number of possible actions (default: 4 for up, down, left, right)
            hidden_size (int): Number of neurons in hidden layers. Default: 64.
        """
        super(DQN, self).__init__()

        # First hidden layer
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()

        # Output layer (Q-values for each action)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input state vector

        Returns:
            torch.Tensor: Q-values for each action [Q_up, Q_down, Q_left, Q_right]
        """
        # First hidden layer
        x = self.fc1(x)
        x = self.relu1(x)

        # Second hidden layer
        x = self.fc2(x)
        x = self.relu2(x)

        # Output layer (no activation - we want raw Q-values)
        x = self.fc3(x)

        return x


class ReplayBuffer:
    """Experience replay buffer for storing and sampling experiences.

    Stores (state, action, reward, next_state, done) tuples and samples
    random batches during training. This breaks correlation between consecutive samples.
    """

    def __init__(self, capacity=10000):
        """Initialize replay buffer.

        Args:
            capacity (int): Maximum number of experiences to store. Default: 10000.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer.

        Args:
            state: Current state (tuple or array)
            action: Action taken (int or str)
            reward: Reward received
            next_state: Next state (tuple or array)
            done: Whether episode terminated
        """
        # Convert tuple states to numpy arrays for consistency
        if isinstance(state, tuple):
            state = np.array(state)
        if isinstance(next_state, tuple):
            next_state = np.array(next_state)

        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        """Sample a random batch of experiences.

        Args:
            batch_size (int): Number of experiences to sample

        Returns:
            list: List of sampled experiences
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return current buffer size."""
        return len(self.memory)


class DQNAgent:
    """Deep Q-Network Agent.

    Uses a neural network to approximate Q-values instead of a tabular Q-table.
    Implements experience replay and target network for stable training.

    Key differences from Q-LearningAgent:
    1. Q-values come from neural network forward pass, not table lookup
    2. Training uses random batches from replay buffer, not single samples
    3. Target network provides stable Q-value estimates for next states
    """

    def __init__(
        self,
        state_size,
        action_size,
        actions,
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        exploration_min=0.01,
        batch_size=64,
        replay_buffer_size=10000,
        target_update_frequency=100,
        device='cpu'
    ):
        """Initialize the DQN Agent.

        Args:
            state_size (int): Dimension of state vector (e.g., 4 for [x, y, gx, gy])
            action_size (int): Number of possible actions (e.g., 4)
            actions (list): List of action names ["up", "down", "left", "right"]
            hidden_size (int): Neurons in hidden layers. Default: 64.
            learning_rate (float): Learning rate for optimizer. Default: 0.001 (lower than Q-learning).
            gamma (float): Discount factor. Default: 0.99.
            exploration_rate (float): Initial epsilon (exploration probability). Default: 1.0.
            exploration_decay (float): Epsilon decay per episode. Default: 0.995.
            exploration_min (float): Minimum exploration rate. Default: 0.01.
            batch_size (int): Batch size for training. Default: 64.
            replay_buffer_size (int): Size of experience replay buffer. Default: 10000.
            target_update_frequency (int): Update target network every N steps. Default: 100.
            device (str): 'cpu' or 'cuda' for GPU acceleration. Default: 'cpu'.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.actions = actions
        self.hidden_size = hidden_size

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.device = device

        # Step counter for target network updates
        self.step_count = 0

        # Initialize policy network (the one we use for action selection)
        self.policy_net = DQN(state_size, action_size, hidden_size).to(device)

        # Initialize target network (frozen copy for stable Q-value targets)
        self.target_net = DQN(state_size, action_size, hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode

        # Optimizer for training
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Loss function (Mean Squared Error)
        self.criterion = nn.MSELoss()

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def state_to_tensor(self, state):
        """Convert state to tensor for network input.

        Args:
            state: State representation (tuple or array)

        Returns:
            torch.Tensor: State as tensor with shape (1, state_size)
        """
        # Convert to numpy array if tuple
        if isinstance(state, tuple):
            state = np.array(state)

        # Add goal coordinates to state
        # For GridWorld, state is (x, y), goal is always (grid_size-1, grid_size-1)
        # This gives the agent knowledge of goal location
        # This is a design choice - could also just use agent position
        # For now, we'll just use agent position and let the network learn through exploration
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return state_tensor

    def choose_action(self, state, training=True):
        """Choose an action using epsilon-greedy policy.

        Args:
            state: Current state (tuple: row, col)
            training (bool): Whether in training mode (enables exploration). Default: True.

        Returns:
            str: The chosen action from the actions list.
        """
        # Epsilon-greedy: explore with probability epsilon
        if training and random.random() < self.exploration_rate:
            return random.choice(self.actions)

        # Exploit: choose action with highest Q-value
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor)

            # Get action with maximum Q-value
            q_values_array = q_values.cpu().data.numpy()[0]
            best_action_index = np.argmax(q_values_array)

            # Handle ties by choosing randomly among max actions
            max_q_value = q_values_array[best_action_index]
            tied_indices = np.where(q_values_array == max_q_value)[0]
            if len(tied_indices) > 1:
                best_action_index = random.choice(tied_indices)

            return self.actions[best_action_index]

    def get_q_values(self, state):
        """Get Q-values for a given state (for debugging/analysis).

        Args:
            state: Current state

        Returns:
            numpy.ndarray: Q-values for all actions
        """
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().data.numpy()[0]

    def update_exploration_rate(self):
        """Decay exploration rate (epsilon) after each episode.

        Gradually reduces exploration to favor exploitation of learned policy.
        """
        self.exploration_rate = max(self.exploration_min,
                                   self.exploration_rate * self.exploration_decay)

    def set_exploration_rate(self, exploration_rate):
        """Manually set exploration rate.

        Useful for evaluation or custom schedules.

        Args:
            exploration_rate (float): New epsilon value.
        """
        self.exploration_rate = max(self.exploration_min, exploration_rate)

    def update_target_network(self):
        """Copy policy network weights to target network.

        Should be called periodically (e.g., every 100 steps) for stable training.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def learn(self, batch):
        """Train policy network on a batch of experiences.

        Implements the DQN learning algorithm:
        1. Sample batch from replay buffer
        2. Compute current Q-values using policy network
        3. Compute target Q-values using target network
        4. Calculate loss (MSE between current and target)
        5. Backpropagate to update policy network
        6. Periodically update target network

        Args:
            batch (list): List of Experience tuples
        """
        if len(batch) < self.batch_size:
            return  # Not enough samples yet

        # Prepare batch tensors
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([self.actions.index(e.action) if isinstance(e.action, str) else e.action
                                          for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([1.0 if e.done else 0.0 for e in batch]).to(self.device)

        # Current Q-values from policy network
        with torch.no_grad():
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].detach()
            expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Loss calculation (Mean Squared Error)
        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.update_target_network()

        return loss.item()


class StateEncoder:
    """Encodes states for neural network input.

    For GridWorld, we need to convert the raw state (x, y position)
    into a format the neural network can process.

    Options for state representation:
    1. Just agent position: [x, y] - Simple, but no goal awareness
    2. Agent + goal position: [x, y, gx, gy] - Gives goal location directly
    3. Relative position: [gx-x, gy-y] - Gives distance to goal
    4. One-hot encoding of entire grid: 9-dimensional vector for 3x3 grid

    For simplicity, we'll start with option 2: [x, y, gx, gy]
    """

    @staticmethod
    def encode(state, goal_position):
        """Encode state and goal into a 4-dimensional vector.

        Args:
            state (tuple): Agent position (row, col)
            goal_position (tuple): Goal position (row, col)

        Returns:
            numpy.ndarray: Encoded state [x, y, gx, gy]
        """
        x, y = state
        gx, gy = goal_position
        return np.array([x, y, gx, gy], dtype=np.float32)


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Deep Q-Network (DQN) Agent - Module Test")
    print("=" * 60)

    # Test 1: Neural Network Architecture
    print("\n1. Testing DQN Neural Network...")
    state_size = 4  # [x, y, gx, gy]
    action_size = 4  # up, down, left, right

    dqn = DQN(state_size, action_size, hidden_size=64)
    print(f"   Network architecture created:")
    print(f"   - Input size: {state_size}")
    print(f"   - Hidden size: 64 (2 layers)")
    print(f"   - Output size: {action_size}")

    # Test forward pass
    test_state = torch.FloatTensor([[0, 0, 2, 2]])  # Agent at (0,0), goal at (2,2)
    q_values = dqn(test_state)
    print(f"\n   Forward pass test:")
    print(f"   - Input state: [0, 0, 2, 2] (agent at (0,0), goal at (2,2))")
    print(f"   - Output Q-values: {q_values.detach().numpy()[0]}")
    print(f"   - Best action: index {np.argmax(q_values.detach().numpy()[0])}")

    # Test 2: Replay Buffer
    print("\n2. Testing Experience Replay Buffer...")
    buffer = ReplayBuffer(capacity=10)

    # Add some experiences
    for i in range(5):
        buffer.push((0, 0), 1, -0.1, (1, 0), False)
        buffer.push((1, 0), 3, -0.1, (1, 1), False)
        buffer.push((1, 1), 1, 10.0, (2, 2), True)

    print(f"   Added 15 experiences to buffer")
    print(f"   Buffer size: {len(buffer)}")

    batch = buffer.sample(batch_size=3)
    print(f"   Sampled batch of {len(batch)} experiences")
    print(f"   First experience: state={batch[0].state}, action={batch[0].action}, reward={batch[0].reward}")

    # Test 3: State Encoding
    print("\n3. Testing State Encoding...")
    agent_pos = (0, 0)
    goal_pos = (2, 2)
    encoded_state = StateEncoder.encode(agent_pos, goal_pos)
    print(f"   Agent position: {agent_pos}")
    print(f"   Goal position: {goal_pos}")
    print(f"   Encoded state: {encoded_state}")

    # Test 4: DQN Agent Initialization
    print("\n4. Testing DQN Agent Initialization...")
    actions = ["up", "down", "left", "right"]
    agent = DQNAgent(
        state_size=4,
        action_size=4,
        actions=actions,
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.99,
        exploration_rate=1.0,
        batch_size=64,
        replay_buffer_size=10000,
        target_update_frequency=100
    )

    print(f"   Agent initialized with:")
    print(f"   - Policy network: {'cuda' if agent.device == 'cuda' else 'cpu'}")
    print(f"   - Learning rate: {agent.learning_rate}")
    print(f"   - Gamma: {agent.gamma}")
    print(f"   - Epsilon: {agent.exploration_rate}")
    print(f"   - Batch size: {agent.batch_size}")

    # Test 5: Action Selection
    print("\n5. Testing Action Selection...")
    test_positions = [(0, 0), (1, 1), (2, 2)]
    goal_position = (2, 2)

    for pos in test_positions:
        # Encode state with goal position
        encoded_state = StateEncoder.encode(pos, goal_position)
        action = agent.choose_action(encoded_state, training=True)
        q_values = agent.get_q_values(encoded_state)
        print(f"   Position {pos}: Action={action}, Q-values={np.round(q_values, 2)}")

    print("\n" + "=" * 60)
    print("All module tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("- Implement learn() method for training")
    print("- Create training script for DQN")
    print("- Compare DQN vs Q-Learning performance")
