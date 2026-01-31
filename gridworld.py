import random


class GridWorld:
    """A GridWorld reinforcement learning environment.

    The agent navigates a 2D grid to reach a goal while avoiding walls.
    Supports discrete action space and OpenAI Gym-like API conventions.
    """

    def __init__(self, grid_size=5, obstacle_density=0.1, seed=None):
        """Initialize the environment.

        Args:
            grid_size (int): Dimensions of the grid (creates grid_size × grid_size grid).
                Default: 5.
            obstacle_density (float): Percentage (0.0-1.0) of cells that become walls/obstacles.
                Default: 0.1 (10% of cells).
            seed (int, optional): Random seed for reproducibility. If None, uses random seed.

        Raises:
            ValueError: If grid_size is not a positive integer.
        """
        if grid_size <= 0:
            raise ValueError("grid_size must be positive integer")

        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.rng = random.Random(seed)

        # Will be set in reset()
        self.agent_position = None
        self.goal_position = None
        self.obstacles = set()
        self.done = False

        # Define action space
        self.action_space = ["up", "down", "left", "right"]

        # Define observation space (all possible positions)
        self.observation_space = (grid_size, grid_size)

    def reset(self):
        """Reset the environment to initial state and return the initial observation.

        Returns:
            tuple[int, int]: Initial agent position (row, column).

        Raises:
            ValueError: If grid_size is 0 or negative.
        """
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive integer")

        # Clear obstacles
        self.obstacles = set()

        # Generate obstacles based on obstacle_density
        num_obstacles = int(self.grid_size ** 2 * self.obstacle_density)
        all_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]

        # Exclude (0, 0) from possible obstacle locations (agent start position)
        available_cells = [cell for cell in all_cells if cell != (0, 0)]

        if num_obstacles > 0 and available_cells:
            self.obstacles = set(self.rng.sample(available_cells, min(num_obstacles, len(available_cells))))

        # Set agent position (top-left corner)
        self.agent_position = (0, 0)

        # Set goal position (bottom-right corner)
        self.goal_position = (self.grid_size - 1, self.grid_size - 1)

        # Ensure goal position is not an obstacle
        self.obstacles.discard(self.goal_position)

        # Reset done status
        self.done = False

        # Edge case: if agent starts at goal (grid_size = 1)
        if self.agent_position == self.goal_position:
            self.done = True

        return self.agent_position

    def step(self, action):
        """Execute one action in the environment.

        Args:
            action (str or int): Either string ("up", "down", "left", "right") or integer
                (0=up, 1=down, 2=left, 3=right).

        Returns:
            tuple: (observation, reward, done) where:
                - observation (tuple): New agent position (row, column)
                - reward (float): Reward for this step
                - done (bool): True if episode finished, False otherwise

        Raises:
            ValueError: If action is not in action space.
            RuntimeError: If reset() has not been called.
        """
        if self.agent_position is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Map integer actions to string actions
        if isinstance(action, int):
            action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
            if action not in action_map:
                raise ValueError("Invalid action. Must be one of ['up', 'down', 'left', 'right'] or corresponding integers 0-3")
            action = action_map[action]

        # Validate action
        if action not in self.action_space:
            raise ValueError("Invalid action. Must be one of ['up', 'down', 'left', 'right'] or corresponding integers 0-3")

        # Get current position
        row, col = self.agent_position

        # Determine attempted new position based on action
        if action == "up":
            new_row, new_col = row - 1, col
        elif action == "down":
            new_row, new_col = row + 1, col
        elif action == "left":
            new_row, new_col = row, col - 1
        elif action == "right":
            new_row, new_col = row, col + 1
        else:
            raise ValueError("Invalid action. Must be one of ['up', 'down', 'left', 'right'] or corresponding integers 0-3")

        # Check if new position is valid (within bounds and not an obstacle)
        is_valid = (0 <= new_row < self.grid_size and
                    0 <= new_col < self.grid_size and
                    (new_row, new_col) not in self.obstacles)

        # Update position if valid
        if is_valid:
            self.agent_position = (new_row, new_col)

        # Check if agent reached goal
        if self.agent_position == self.goal_position:
            reward = 10
            self.done = True
        else:
            reward = -0.1
            self.done = False

        return self.agent_position, reward, self.done

    def render(self, mode='human', style='emoji'):
        """Render the current grid state to console.

        Args:
            mode (str): Render mode. Only 'human' (console output) is supported.
            style (str): Visual style - 'emoji' or 'ascii'. Default: 'emoji'.

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self.agent_position is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Check mode
        if mode != 'human':
            print("Warning: Only 'human' mode is supported")

        # Determine character set based on style
        if style == 'emoji':
            agent_char = "🤖"
            goal_char = "🏁"
            wall_char = "🧱"
            empty_char = "⬜"
        elif style == 'ascii':
            agent_char = "A"
            goal_char = "G"
            wall_char = "#"
            empty_char = "."
        else:
            print("Warning: Style must be 'emoji' or 'ascii', using emoji")
            agent_char = "🤖"
            goal_char = "🏁"
            wall_char = "🧱"
            empty_char = "⬜"

        # Print empty line for spacing
        print()

        # Iterate through each row
        for row in range(self.grid_size):
            # Iterate through each column
            for col in range(self.grid_size):
                position = (row, col)

                if position == self.agent_position:
                    print(agent_char, end='')
                elif position == self.goal_position:
                    print(goal_char, end='')
                elif position in self.obstacles:
                    print(wall_char, end='')
                else:
                    print(empty_char, end='')

            # Print newline after each row
            print()

        # Print empty line after grid
        print()

    def get_valid_actions(self):
        """Return list of valid actions from current agent position.

        Returns:
            list[str]: Actions that would result in valid movement (within bounds
                and not hitting walls).

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self.agent_position is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        row, col = self.agent_position
        valid_actions = []

        # Check each action
        action_deltas = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }

        for action, (delta_row, delta_col) in action_deltas.items():
            new_row, new_col = row + delta_row, col + delta_col

            # Check if position is valid (within bounds and not an obstacle)
            if (0 <= new_row < self.grid_size and
                0 <= new_col < self.grid_size and
                (new_row, new_col) not in self.obstacles):
                valid_actions.append(action)

        return valid_actions
