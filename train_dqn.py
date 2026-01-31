"""Training script for Deep Q-Network (DQN) Agent."""

from gridworld import GridWorld
from dqn import DQNAgent, StateEncoder


def train_dqn_agent(
    grid_size=3,
    obstacle_density=0.1,
    num_episodes=1000,
    hidden_size=64,
    learning_rate=0.001,
    gamma=0.99,
    exploration_rate=1.0,
    exploration_decay=0.995,
    exploration_min=0.01,
    batch_size=64,
    replay_buffer_size=10000,
    target_update_frequency=100,
    seed=42
):
    """Train DQN agent on GridWorld.

    Args:
        grid_size (int): Dimensions of the grid. Default: 3.
        obstacle_density (float): Percentage of cells that are walls. Default: 0.1.
        num_episodes (int): Number of training episodes. Default: 1000.
        hidden_size (int): Neurons in hidden layers. Default: 64.
        learning_rate (float): Learning rate for optimizer. Default: 0.001.
        gamma (float): Discount factor. Default: 0.99.
        exploration_rate (float): Initial epsilon. Default: 1.0.
        exploration_decay (float): Epsilon decay per episode. Default: 0.995.
        exploration_min (float): Minimum exploration rate. Default: 0.01.
        batch_size (int): Batch size for training. Default: 64.
        replay_buffer_size (int): Size of experience replay buffer. Default: 10000.
        target_update_frequency (int): Update target network every N steps. Default: 100.
        seed (int): Random seed for reproducibility. Default: 42.

    Returns:
        tuple: (agent, episode_rewards, episode_steps, exploration_rates)
    """
    print("=" * 60)
    print("Deep Q-Network (DQN) Training")
    print("=" * 60)

    # Create environment
    print(f"\n1. Creating GridWorld environment (grid_size={grid_size}, obstacle_density={obstacle_density})...")
    env = GridWorld(grid_size=grid_size, obstacle_density=obstacle_density, seed=seed)

    # Create agent
    print("2. Creating DQN Agent...")
    actions = ["up", "down", "left", "right"]
    state_size = 4  # [x, y, gx, gy]
    action_size = len(actions)

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        actions=actions,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_rate=exploration_rate,
        exploration_decay=exploration_decay,
        exploration_min=exploration_min,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        target_update_frequency=target_update_frequency
    )

    # Training metrics
    episode_rewards = []
    episode_steps = []
    exploration_rates = []
    losses = []

    # Training loop
    print(f"\n3. Starting training for {num_episodes} episodes...")
    print(f"   Initial epsilon: {exploration_rate}")
    print(f"   Final epsilon: {exploration_min}")
    print(f"   Decay rate: {exploration_decay}")

    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        goal_position = env.goal_position

        # Encode state with goal position
        encoded_state = StateEncoder.encode(state, goal_position)

        episode_reward = 0
        step = 0
        done = False
        episode_losses = []

        # Run episode until done
        while not done and step < 50:  # Limit steps to prevent infinite loops
            # Choose action (epsilon-greedy)
            action = agent.choose_action(encoded_state, training=True)

            # Execute action
            next_state, reward, done = env.step(action)

            # Encode next state
            encoded_next_state = StateEncoder.encode(next_state, goal_position)

            # Store experience in replay buffer
            agent.replay_buffer.push(encoded_state, action, reward, encoded_next_state, done)

            # Train if enough experiences
            if len(agent.replay_buffer) >= batch_size:
                batch = agent.replay_buffer.sample(batch_size)
                loss = agent.learn(batch)
                episode_losses.append(loss)

            # Update state and reward
            encoded_state = encoded_next_state
            episode_reward += reward
            step += 1

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        exploration_rates.append(agent.exploration_rate)

        # Decay exploration rate
        agent.update_exploration_rate()

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            avg_steps = sum(episode_steps[-100:]) / 100
            avg_epsilon = exploration_rates[-1]
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Steps: {avg_steps:.1f} | "
                  f"Epsilon: {avg_epsilon:.3f} | "
                  f"Avg Loss: {avg_loss:.4f}")
            episode_losses = []  # Reset for next batch

    print(f"\n4. Training complete!")
    print(f"   Total episodes: {num_episodes}")
    print(f"   Average reward (all): {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"   Average steps (all): {sum(episode_steps) / len(episode_steps):.1f}")
    print(f"   Final epsilon: {exploration_rates[-1]:.3f}")
    print(f"   Target network updates: {agent.step_count}")

    return agent, episode_rewards, episode_steps, exploration_rates


def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate trained DQN agent without exploration.

    Args:
        env: GridWorld environment.
        agent: Trained DQN Agent.
        num_episodes (int): Number of test episodes. Default: 10.

    Returns:
        tuple: (rewards, steps)
    """
    print(f"\nEvaluating agent ({num_episodes} episodes, no exploration)...")

    # Disable exploration
    original_epsilon = agent.exploration_rate
    agent.set_exploration_rate(0.0)

    test_rewards = []
    test_steps = []

    for episode in range(num_episodes):
        state = env.reset()
        goal_position = env.goal_position
        encoded_state = StateEncoder.encode(state, goal_position)

        episode_reward = 0
        steps = 0
        done = False

        while not done and steps < 50:
            action = agent.choose_action(encoded_state, training=False)
            next_state, reward, done = env.step(action)

            encoded_state = StateEncoder.encode(next_state, goal_position)
            episode_reward += reward
            steps += 1

        test_rewards.append(episode_reward)
        test_steps.append(steps)

    # Restore exploration rate
    agent.set_exploration_rate(original_epsilon)

    avg_reward = sum(test_rewards) / len(test_rewards)
    avg_steps = sum(test_steps) / len(test_steps)
    success_rate = sum(1 for r in test_rewards if r > 0) / len(test_rewards)

    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Average steps: {avg_steps:.1f}")
    print(f"   Success rate: {success_rate:.1%}")

    return test_rewards, test_steps


def show_learned_policy(env, agent):
    """Show a demonstration episode with learned policy.

    Args:
        env: GridWorld environment.
        agent: Trained DQN Agent.
    """
    print(f"\n{'=' * 60}")
    print("Demonstrating Learned Policy (No Exploration)")
    print(f"{'=' * 60}")

    # Disable exploration
    original_epsilon = agent.exploration_rate
    agent.set_exploration_rate(0.0)

    state = env.reset()
    goal_position = env.goal_position
    encoded_state = StateEncoder.encode(state, goal_position)

    print("\nInitial state:")
    env.render()

    step = 0
    path = []
    done = False

    while not done and step < 10:
        action = agent.choose_action(encoded_state, training=False)
        path.append(action)

        next_state, reward, done = env.step(action)
        encoded_state = StateEncoder.encode(next_state, goal_position)

        print(f"\nStep {step + 1}: Action={action}, Reward={reward}")
        env.render()

        step += 1

        if done:
            print("   ✓ Goal reached!")

    print(f"\nPath: {' → '.join(path)}")
    print(f"Total steps: {step}")

    # Restore exploration rate
    agent.set_exploration_rate(original_epsilon)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("DQN Agent Training and Evaluation")
    print("=" * 60)

    # Train agent
    agent, rewards, steps, epsilons = train_dqn_agent(
        grid_size=3,
        obstacle_density=0.1,
        num_episodes=1000,
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        exploration_min=0.01,
        batch_size=64,
        replay_buffer_size=10000,
        target_update_frequency=100,
        seed=42
    )

    # Create a fresh environment for evaluation
    from gridworld import GridWorld
    env_eval = GridWorld(grid_size=3, obstacle_density=0.1, seed=42)

    # Evaluate agent
    test_rewards, test_steps = evaluate_agent(env_eval, agent, num_episodes=10)

    # Show demonstration
    show_learned_policy(env_eval, agent)

    # Plot learning curves
    print(f"\n{'=' * 60}")
    print("Learning Curves")
    print(f"{'=' * 60}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Reward per episode
    axes[0].plot(rewards)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Reward per Episode')
    axes[0].grid(True, alpha=0.3)

    # Moving average of rewards
    window = 100
    if len(rewards) > window:
        moving_avg = [sum(rewards[max(0, i-window):i+1]) / window
                   for i in range(len(rewards))]
        axes[0].plot(range(window, len(rewards)), moving_avg[window:], 'r-', linewidth=2, label='Moving Avg (100)')
        axes[0].legend()

    # Steps per episode
    axes[1].plot(steps)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Steps per Episode')
    axes[1].grid(True, alpha=0.3)

    # Moving average of steps
    if len(steps) > window:
        moving_avg = [sum(steps[max(0, i-window):i+1]) / window
                   for i in range(len(steps))]
        axes[1].plot(range(window, len(steps)), moving_avg[window:], 'r-', linewidth=2, label='Moving Avg (100)')
        axes[1].legend()

    # Epsilon over time
    axes[2].plot(epsilons)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Epsilon (Exploration Rate)')
    axes[2].set_title('Exploration Rate Decay')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/cml24sugk0001irps1i59garq/AGI/dqn_learning_curves.png', dpi=150)
    print(f"\nLearning curves saved to: AGI/dqn_learning_curves.png")

    plt.show()

    print(f"\n{'=' * 60}")
    print("Training and evaluation complete!")
    print(f"{'=' * 60}")
    print("\nObservations:")
    print(f"- Final average reward: {sum(rewards[-100:]) / 100:.2f}")
    print(f"- Final average steps: {sum(steps[-100:]) / 100:.1f}")
    print(f"- Final exploration rate: {epsilons[-1]:.3f}")
    print(f"- Success rate (evaluation): {sum(1 for r in test_rewards if r > 0) / len(test_rewards):.1%}")
