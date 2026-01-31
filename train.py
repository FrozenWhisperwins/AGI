"""Training script for Q-Learning Agent in GridWorld environment."""

from gridworld import GridWorld
from agent import QLearningAgent


def train_agent(
    grid_size=5,
    obstacle_density=0.1,
    num_episodes=500,
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_rate=0.1,
    seed=42
):
    """Train the Q-Learning agent in GridWorld.

    Args:
        grid_size (int): Dimensions of the grid. Default: 5.
        obstacle_density (float): Percentage of cells that are walls. Default: 0.1.
        num_episodes (int): Number of training episodes. Default: 500.
        learning_rate (float): Alpha parameter for Q-learning. Default: 0.1.
        discount_factor (float): Gamma parameter for Q-learning. Default: 0.9.
        exploration_rate (float): Epsilon for epsilon-greedy policy. Default: 0.1.
        seed (int): Random seed for reproducibility. Default: 42.

    Returns:
        tuple: (env, agent, episode_rewards) where:
            - env: The trained GridWorld environment
            - agent: The trained QLearningAgent
            - episode_rewards: List of total rewards for each episode
    """
    # Create environment
    print("Creating GridWorld environment...")
    env = GridWorld(grid_size=grid_size, obstacle_density=obstacle_density, seed=seed)

    # Create agent
    print("Creating Q-Learning agent...")
    actions = ["up", "down", "left", "right"]
    agent = QLearningAgent(
        grid_size=grid_size,
        actions=actions,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate
    )

    # Training loop
    print(f"\nStarting training for {num_episodes} episodes...")
    episode_rewards = []

    for episode in range(num_episodes):
        # Reset environment for new episode
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        # Run episode until done
        while not done:
            # Choose action using epsilon-greedy policy
            action = agent.choose_action(state)

            # Take action in environment
            next_state, reward, done = env.step(action)

            # Learn from this experience
            agent.learn(state, action, reward, next_state)

            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1

        # Record episode reward
        episode_rewards.append(episode_reward)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            print(f"Episode {episode + 1}/{num_episodes} | Steps: {step} | Reward: {episode_reward:.2f} | Avg (last 100): {avg_reward:.2f}")

    print(f"\nTraining complete!")
    print(f"Total episodes: {num_episodes}")
    print(f"Average reward (all episodes): {sum(episode_rewards) / len(episode_rewards):.2f}")

    return env, agent, episode_rewards


def evaluate_agent(env, agent, num_episodes=10, render=False):
    """Evaluate the trained agent's performance.

    Args:
        env: GridWorld environment.
        agent: Trained QLearningAgent.
        num_episodes (int): Number of evaluation episodes. Default: 10.
        render (bool): Whether to render each step. Default: False.

    Returns:
        list: Total rewards for each evaluation episode.
    """
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    evaluation_rewards = []

    # Temporarily disable exploration for evaluation
    original_exploration_rate = agent.exploration_rate
    agent.set_exploration_rate(0.0)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            # Choose best action (no exploration)
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            if render:
                print(f"\nEpisode {episode + 1}, Step {step + 1}")
                print(f"Action: {action}, Reward: {reward}")
                env.render()

            state = next_state
            episode_reward += reward
            step += 1

        evaluation_rewards.append(episode_reward)

        if render:
            print(f"\nEpisode {episode + 1} complete - Total reward: {episode_reward:.2f}")

    # Restore original exploration rate
    agent.set_exploration_rate(original_exploration_rate)

    avg_reward = sum(evaluation_rewards) / len(evaluation_rewards)
    print(f"\nEvaluation complete!")
    print(f"Average reward: {avg_reward:.2f}")

    return evaluation_rewards


def print_learned_policy(agent, env):
    """Print the learned policy from the Q-table.

    Args:
        agent: QLearningAgent with learned Q-table.
        env: GridWorld environment for context.
    """
    print("\n" + "=" * 60)
    print("Learned Policy (Best Action for Each State)")
    print("=" * 60)

    # Action symbols for display
    action_symbols = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→"
    }

    for row in range(agent.grid_size):
        policy_row = []
        for col in range(agent.grid_size):
            state = (row, col)

            # Check if this is a wall
            if state in env.obstacles:
                policy_row.append("#")  # Wall
            elif state == env.goal_position:
                policy_row.append("🏁")  # Goal
            else:
                # Get best action for this state
                q_values = agent.get_q_values(state)
                best_action_index = np.argmax(q_values)
                best_action = agent.actions[best_action_index]
                policy_row.append(action_symbols[best_action])

        print(" ".join(policy_row))

    print("\nLegend: ↑=up, ↓=down, ←=left, →=right, #=wall, 🏁=goal")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("Q-Learning Agent Training")
    print("=" * 60)

    # Train the agent
    env, agent, episode_rewards = train_agent(
        grid_size=5,
        obstacle_density=0.1,
        num_episodes=500,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.1,
        seed=42
    )

    # Print learned policy
    print_learned_policy(agent, env)

    # Evaluate the trained agent
    print("\nRunning evaluation with learned policy (no exploration)...")
    evaluate_agent(env, agent, num_episodes=10, render=False)

    # Show one episode with rendering to demonstrate learned behavior
    print("\n" + "=" * 60)
    print("Demonstration: One Episode with Rendering")
    print("=" * 60)
    evaluate_agent(env, agent, num_episodes=1, render=True)

    print("\nTraining and evaluation complete!")
