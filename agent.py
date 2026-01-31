import numpy as np
import random


class QLearningAgent:
    """Q-Learning Agent for GridWorld environment.

    The agent learns to navigate the grid by maintaining a Q-table that stores
    the expected value of taking each action in each state.
    """

    def __init__(self, grid_size, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        """Initialize the Q-Learning Agent.

        Args:
            grid_size (int): Dimensions of the grid (e.g., 5 for 5x5 grid).
            actions (list): List of possible actions (e.g., ["up", "down", "left", "right"]).
            learning_rate (float): Alpha (α) - How much to update Q-value based on new information.
                Default: 0.1. Higher values learn faster but may be less stable.
            discount_factor (float): Gamma (γ) - How much to value future rewards over immediate ones.
                Default: 0.9. Higher values consider long-term rewards more.
            exploration_rate (float): Epsilon (ε) - Probability of taking random action for exploration.
                Default: 0.1. Higher values explore more, lower values exploit known good actions.
        """
        self.grid_size = grid_size
        self.actions = actions
        self.num_actions = len(actions)

        # Q-Table: grid_size x grid_size x num_actions array of zeros
        # Dimensions: [row][col][action_index]
        self.q_table = np.zeros((grid_size, grid_size, self.num_actions))

        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy.

        Args:
            state (tuple): Current state (row, column).

        Returns:
            str: The chosen action from the actions list.
        """
        row, col = state

        # With probability exploration_rate, take a random action (exploration)
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)

        # Otherwise, take the best known action (exploitation)
        # Get Q-values for all actions in this state
        q_values = self.q_table[row, col, :]

        # Find action(s) with maximum Q-value
        max_q_value = np.max(q_values)
        best_action_indices = np.where(q_values == max_q_value)[0]

        # If multiple actions have the same max Q-value, choose one randomly
        best_action_index = random.choice(best_action_indices)

        return self.actions[best_action_index]

    def learn(self, state, action, reward, next_state):
        """Update Q-table using Q-Learning update rule.

        The Q-Learning update formula:
        Q(s, a) = Q(s, a) + α * [r + γ * max_a'(Q(s', a')) - Q(s, a)]

        Where:
        - Q(s, a) is current Q-value for state-action pair
        - α (alpha) is learning_rate
        - r is reward received
        - γ (gamma) is discount_factor
        - max_a'(Q(s', a')) is maximum Q-value for any action in next_state

        Args:
            state (tuple): Current state (row, column) before taking action.
            action (str): Action taken in this state.
            reward (float): Reward received after taking action.
            next_state (tuple): New state (row, column) after taking action.
        """
        # Get indices for current state and action
        row, col = state
        action_index = self.actions.index(action)

        # Get current Q-value for (state, action)
        current_q_value = self.q_table[row, col, action_index]

        # Get maximum Q-value for the next state
        # This represents the best possible future reward from next_state
        next_row, next_col = next_state
        max_next_q_value = np.max(self.q_table[next_row, next_col, :])

        # Calculate the target Q-value
        # target = immediate reward + discounted best future reward
        target_q_value = reward + self.discount_factor * max_next_q_value

        # Update Q-value using Q-Learning formula
        # Q(s, a) = Q(s, a) + α * (target - Q(s, a))
        self.q_table[row, col, action_index] = current_q_value + self.learning_rate * (target_q_value - current_q_value)

    def set_exploration_rate(self, exploration_rate):
        """Set a new exploration rate (epsilon).

        Useful for implementing epsilon decay during training.

        Args:
            exploration_rate (float): New exploration rate between 0 and 1.
        """
        self.exploration_rate = exploration_rate

    def get_q_values(self, state):
        """Get all Q-values for a given state.

        Useful for debugging and analyzing learned policies.

        Args:
            state (tuple): State (row, column).

        Returns:
            numpy.ndarray: Q-values for all actions in this state.
        """
        row, col = state
        return self.q_table[row, col, :]
