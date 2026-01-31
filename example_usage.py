"""Demonstration script for GridWorld environment."""

import random
from gridworld import GridWorld


def main():
    print("=" * 60)
    print("GridWorld Environment Demonstration")
    print("=" * 60)

    # 1. Basic environment initialization
    print("\n1. Initializing environment (grid_size=5, obstacle_density=0.15, seed=42)...")
    env = GridWorld(grid_size=5, obstacle_density=0.15, seed=42)

    # 2. Reset environment
    print("\n2. Resetting environment...")
    state = env.reset()
    print(f"   Initial state: {state}")

    # 3. Render initial state
    print("\n3. Rendering initial state (emoji style):")
    env.render(style='emoji')

    # 4. Take random steps for demonstration
    print("\n4. Taking random steps for demonstration...")
    actions = ["up", "down", "left", "right"]
    # Or use integers: actions_int = [0, 1, 2, 3]

    for step in range(10):
        action = random.choice(actions)
        next_state, reward, done = env.step(action)
        print(f"   Step {step}: Action={action}, State={next_state}, Reward={reward}, Done={done}")
        env.render(style='emoji')

        if done:
            print("   Goal reached!")
            break

    # 5. Demonstrate get_valid_actions()
    print("5. Valid actions from current position:")
    valid = env.get_valid_actions()
    print(f"   {valid}")

    # 6. Demonstrate both render styles
    print("\n6. Demonstrating both render styles:")

    print("\n   Emoji style:")
    env.render(style='emoji')

    print("\n   ASCII style:")
    env.render(style='ascii')

    # 7. Demonstrate deterministic behavior with seed
    print("\n7. Testing reproducibility with same seed...")
    env2 = GridWorld(grid_size=5, obstacle_density=0.15, seed=42)
    state2 = env2.reset()
    print("   Same seed - should be identical layout:")
    env2.render(style='ascii')

    # Verify layouts are identical
    if state == state2 and env.obstacles == env2.obstacles:
        print("   ✓ Layouts are identical (reproducibility confirmed)")
    else:
        print("   ✗ Layouts differ (unexpected)")

    # 8. Demonstrate integer action format
    print("\n8. Demonstrating integer action format (0=up, 1=down, 2=left, 3=right)...")
    env3 = GridWorld(grid_size=4, obstacle_density=0.1, seed=100)
    env3.reset()
    print("   Initial state:")
    env3.render(style='ascii')

    # Test integer actions
    print("\n   Testing integer actions:")
    actions_int = [0, 1, 2, 3]
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}

    for i, action_int in enumerate(actions_int):
        next_state, reward, done = env3.step(action_int)
        print(f"   Step {i}: Action={action_int} ({action_names[action_int]}), State={next_state}, Reward={reward}")
        env3.render(style='ascii')

    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
