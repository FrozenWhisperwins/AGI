"""Maze variant of GridWorld with fixed obstacles."""

from gridworld import GridWorld


class MazeGridWorld(GridWorld):
    """GridWorld with fixed obstacle layout for maze experiments.

    Creates a 3x3 grid with predetermined wall placements to create
    a simple maze that forces the agent to take a longer path.
    """

    def __init__(self, maze_type='simple'):
        """Initialize the maze environment.

        Args:
            maze_type (str): Type of maze layout.
                - 'simple': One wall blocking direct path
                - 'zigzag': Multiple walls creating a zigzag path
                Default: 'simple'
        """
        # Initialize with 3x3 grid, no random obstacles
        super().__init__(grid_size=3, obstacle_density=0.0)

        self.maze_type = maze_type

        # Predefined wall locations for each maze type
        self.maze_layouts = {
            'simple': [(0, 1)],  # Wall blocks direct right path
            'zigzag': [(0, 1), (1, 1)],  # Two walls create zigzag path
        }

    def reset(self):
        """Reset the environment with fixed maze layout.

        Returns:
            tuple[int, int]: Initial agent position (row, column).
        """
        # Clear obstacles
        self.obstacles = set()

        # Add predefined walls based on maze type
        if self.maze_type in self.maze_layouts:
            self.obstacles = set(self.maze_layouts[self.maze_type])
        else:
            print(f"Warning: Unknown maze_type '{self.maze_type}', using simple layout")
            self.obstacles = set(self.maze_layouts['simple'])

        # Set agent position (top-left corner)
        self.agent_position = (0, 0)

        # Set goal position (bottom-right corner)
        self.goal_position = (self.grid_size - 1, self.grid_size - 1)

        # Ensure goal position is not an obstacle (shouldn't happen with predefined layouts)
        self.obstacles.discard(self.goal_position)

        # Reset done status
        self.done = False

        return self.agent_position


class ComparisonExperiment:
    """Run comparative experiments between maze and no-maze environments."""

    def __init__(self, num_episodes=200, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        """Initialize experiment parameters.

        Args:
            num_episodes (int): Number of training episodes. Default: 200.
            learning_rate (float): Alpha for Q-learning. Default: 0.1.
            discount_factor (float): Gamma for Q-learning. Default: 0.9.
            exploration_rate (float): Epsilon for exploration. Default: 0.1.
        """
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        from agent import QLearningAgent
        self.QLearningAgent = QLearningAgent

    def run_single_experiment(self, env_name, env):
        """Run training on a single environment.

        Args:
            env_name (str): Name of the environment for logging.
            env: GridWorld or MazeGridWorld instance.

        Returns:
            tuple: (agent, episode_rewards, avg_steps_to_goal)
        """
        print(f"\n{'=' * 60}")
        print(f"Experiment: {env_name}")
        print(f"{'=' * 60}")

        # Create agent
        actions = ["up", "down", "left", "right"]
        agent = self.QLearningAgent(
            grid_size=3,
            actions=actions,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            exploration_rate=self.exploration_rate
        )

        # Training loop
        episode_rewards = []
        steps_to_goal = []

        for episode in range(self.num_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done and steps < 50:  # Limit steps to prevent infinite loops
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state)

                state = next_state
                episode_reward += reward
                steps += 1

            episode_rewards.append(episode_reward)
            steps_to_goal.append(steps)

            # Print progress
            if (episode + 1) % 25 == 0:
                avg_reward = sum(episode_rewards[-25:]) / 25
                avg_steps = sum(steps_to_goal[-25:]) / 25
                print(f"Episode {episode + 1}/{self.num_episodes} | Avg Reward (last 25): {avg_reward:.2f} | Avg Steps: {avg_steps:.1f}")

        print(f"\nTraining complete for {env_name}!")
        print(f"Final average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
        print(f"Final average steps: {sum(steps_to_goal) / len(steps_to_goal):.1f}")

        return agent, episode_rewards, steps_to_goal

    def evaluate_agent(self, env, agent, num_episodes=10):
        """Evaluate agent without exploration.

        Args:
            env: Environment instance.
            agent: Trained agent.
            num_episodes (int): Number of test episodes. Default: 10.

        Returns:
            list: Rewards for each test episode.
        """
        print(f"\nEvaluating trained agent ({num_episodes} episodes)...")

        # Disable exploration
        original_epsilon = agent.exploration_rate
        agent.set_exploration_rate(0.0)

        test_rewards = []
        test_steps = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done and steps < 50:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)

                state = next_state
                episode_reward += reward
                steps += 1

            test_rewards.append(episode_reward)
            test_steps.append(steps)

        # Restore exploration rate
        agent.set_exploration_rate(original_epsilon)

        avg_reward = sum(test_rewards) / len(test_rewards)
        avg_steps = sum(test_steps) / len(test_steps)
        success_rate = sum(1 for r in test_rewards if r > 0) / len(test_rewards)

        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average steps: {avg_steps:.1f}")
        print(f"Success rate (reached goal): {success_rate:.1%}")

        return test_rewards, test_steps

    def show_learned_path(self, env, agent):
        """Show the learned optimal path.

        Args:
            env: Environment instance.
            agent: Trained agent.
        """
        print(f"\n{'=' * 60}")
        print("Demonstrating Learned Path (No Exploration)")
        print(f"{'=' * 60}")

        # Disable exploration
        original_epsilon = agent.exploration_rate
        agent.set_exploration_rate(0.0)

        state = env.reset()
        print("Initial state:")
        env.render()

        step = 0
        path = []
        done = False

        while not done and step < 10:
            action = agent.choose_action(state)
            path.append(action)
            next_state, reward, done = env.step(action)

            print(f"Step {step + 1}: Action={action}, Reward={reward}")
            env.render()

            state = next_state
            step += 1

        print(f"\nPath: {' → '.join(path)}")
        print(f"Total steps: {step}")
        # Calculate total reward: step_penalty + goal_reward
        total_reward = sum(-0.1 for _ in range(step - 1)) + (10 if done and step > 0 else -0.1)
        print(f"Total reward: {total_reward:.2f}")

        # Restore exploration rate
        agent.set_exploration_rate(original_epsilon)


if __name__ == "__main__":
    print("=" * 60)
    print("Maze Learning Experiment")
    print("=" * 60)
    print("\nComparing agent learning on:")
    print("1. Simple 3x3 grid (no obstacles)")
    print("2. Maze with one wall (simple)")
    print("3. Maze with two walls (zigzag)")

    # Run experiments
    experiment = ComparisonExperiment(num_episodes=200, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1)

    # Experiment 1: No obstacles (baseline)
    from gridworld import GridWorld
    env_simple = GridWorld(grid_size=3, obstacle_density=0.0, seed=42)
    agent_simple, rewards_simple, steps_simple = experiment.run_single_experiment(
        "Simple Grid (No Obstacles)", env_simple
    )

    # Experiment 2: Simple maze (one wall)
    env_maze_simple = MazeGridWorld(maze_type='simple')
    agent_maze_simple, rewards_maze_simple, steps_maze_simple = experiment.run_single_experiment(
        "Simple Maze (1 Wall)", env_maze_simple
    )

    # Experiment 3: Zigzag maze (two walls)
    env_maze_zigzag = MazeGridWorld(maze_type='zigzag')
    agent_maze_zigzag, rewards_maze_zigzag, steps_maze_zigzag = experiment.run_single_experiment(
        "Zigzag Maze (2 Walls)", env_maze_zigzag
    )

    # Evaluate all agents
    print("\n" + "=" * 60)
    print("EVALUATION (No Exploration)")
    print("=" * 60)

    print("\n1. Simple Grid (No Obstacles):")
    experiment.evaluate_agent(env_simple, agent_simple, num_episodes=10)
    experiment.show_learned_path(env_simple, agent_simple)

    print("\n2. Simple Maze (1 Wall):")
    experiment.evaluate_agent(env_maze_simple, agent_maze_simple, num_episodes=10)
    experiment.show_learned_path(env_maze_simple, agent_maze_simple)

    print("\n3. Zigzag Maze (2 Walls):")
    experiment.evaluate_agent(env_maze_zigzag, agent_maze_zigzag, num_episodes=10)
    experiment.show_learned_path(env_maze_zigzag, agent_maze_zigzag)

    # Comparative analysis
    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS")
    print("=" * 60)

    print(f"\nFinal Average Rewards:")
    print(f"  Simple Grid:     {sum(rewards_simple) / len(rewards_simple):.2f}")
    print(f"  Simple Maze:     {sum(rewards_maze_simple) / len(rewards_maze_simple):.2f}")
    print(f"  Zigzag Maze:    {sum(rewards_maze_zigzag) / len(rewards_maze_zigzag):.2f}")

    print(f"\nFinal Average Steps (last 25 episodes):")
    print(f"  Simple Grid:     {sum(steps_simple[-25:]) / 25:.1f}")
    print(f"  Simple Maze:     {sum(steps_maze_simple[-25:]) / 25:.1f}")
    print(f"  Zigzag Maze:    {sum(steps_maze_zigzag[-25:]) / 25:.1f}")

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
