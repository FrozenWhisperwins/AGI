"""Quick test script for Q-Learning Agent."""

from gridworld import GridWorld
from agent import QLearningAgent


def test_agent():
    """Test the Q-Learning agent with a quick training run."""
    print("=" * 60)
    print("Q-Learning Agent Test")
    print("=" * 60)

    # Create environment
    print("\n1. Creating GridWorld environment...")
    env = GridWorld(grid_size=5, obstacle_density=0.1, seed=42)

    # Create agent
    print("2. Creating Q-Learning agent...")
    actions = ["up", "down", "left", "right"]
    agent = QLearningAgent(
        grid_size=5,
        actions=actions,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.1
    )

    # Quick training
    print(f"\n3. Training for 50 episodes...")
    episode_rewards = []

    for episode in range(50):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 100:  # Limit steps per episode
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)

            state = next_state
            episode_reward += reward
            step += 1

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(f"   Episode {episode + 1}: Steps={step}, Reward={episode_reward:.2f}, Avg (last 10)={avg_reward:.2f}")

    print(f"\n4. Training complete!")
    print(f"   Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")

    # Test learned policy
    print("\n5. Testing learned policy (no exploration)...")
    agent.set_exploration_rate(0.0)

    test_rewards = []
    for i in range(5):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 50:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward
            step += 1

        test_rewards.append(episode_reward)
        print(f"   Test episode {i+1}: Steps={step}, Reward={episode_reward:.2f}, Done={done}")

    print(f"\n   Average test reward: {sum(test_rewards) / len(test_rewards):.2f}")

    # Show a sample episode with rendering
    print("\n6. Demonstrating one episode with rendering...")
    agent.set_exploration_rate(0.0)
    state = env.reset()
    print("   Initial state:")
    env.render()

    step = 0
    done = False
    while not done and step < 10:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        print(f"   Step {step+1}: Action={action}, Reward={reward}")
        env.render()
        state = next_state
        step += 1

        if done:
            print("   Goal reached!")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_agent()
