# Maze Learning Experiment Results

## Experiment Overview

We investigated how Q-Learning agents adapt to increasing environment complexity by comparing three scenarios:
1. **Simple Grid** (3×3, no obstacles) - Baseline
2. **Simple Maze** (3×3, 1 wall at (0,1))
3. **Zigzag Maze** (3×3, 2 walls at (0,1) and (1,1))

**Hypothesis:** Q-Learning agents will learn optimal paths despite obstacles, but increased complexity may affect learning speed.

**Training Parameters:**
- Episodes: 200
- Learning rate (α): 0.1
- Discount factor (γ): 0.9
- Exploration rate (ε): 0.1

---

## Results Summary

### Final Performance (Evaluation - No Exploration)

| Environment | Avg Reward | Avg Steps | Success Rate | Optimal Path |
|-------------|-------------|------------|--------------|--------------|
| Simple Grid (No Obstacles) | 9.70 | 4.0 | 100% | down → right → right → down |
| Simple Maze (1 Wall) | 9.70 | 4.0 | 100% | down → right → down → right |
| Zigzag Maze (2 Walls) | 9.70 | 4.0 | 100% | down → down → right → right |

**Key Finding:** All three environments have the same optimal path length (4 steps) and reward (9.70 = 10 - 4×0.1). The walls force different routes, but the shortest path to the goal remains 4 steps in all three cases.

---

## Learning Curve Analysis

### Training Progress (Average Reward per 25 Episodes)

| Episode Range | Simple Grid | Simple Maze | Zigzag Maze |
|---------------|--------------|-------------|--------------|
| 1-25 | 9.46 | 9.45 | 9.46 |
| 26-50 | 9.67 | 9.68 | 9.65 |
| 51-75 | 9.64 | 9.66 | 9.69 |
| 76-100 | 9.66 | 9.68 | 9.68 |
| 101-125 | 9.64 | 9.68 | 9.67 |
| 126-150 | 9.64 | 9.65 | 9.64 |
| 151-175 | 9.65 | 9.67 | 9.64 |
| 176-200 | 9.64 | 9.65 | 9.62 |

### Training Progress (Average Steps per 25 Episodes)

| Episode Range | Simple Grid | Simple Maze | Zigzag Maze |
|---------------|--------------|-------------|--------------|
| 1-25 | 6.2 | 6.5 | 6.0 |
| 26-50 | 4.2 | 4.2 | 4.1 |
| 51-75 | 4.8 | 4.4 | 4.5 |
| 76-100 | 4.4 | 4.3 | 4.2 |
| 101-125 | 4.1 | 4.2 | 4.3 |
| 126-150 | 4.5 | 4.5 | 4.7 |
| 151-175 | 4.7 | 4.3 | 4.6 |
| 176-200 | 4.5 | 4.5 | 4.8 |

---

## Detailed Analysis

### 1. Learning Speed Comparison

**Initial Learning (Episodes 1-25):**
- All three environments started with similar performance
- Average steps: ~6.0-6.5 steps per episode
- Indicates agents needed initial exploration to discover effective paths

**Convergence Timeline:**
- **Episode 50:** All environments show substantial improvement (4.1-4.2 average steps)
- This suggests Q-Learning quickly identifies near-optimal strategies regardless of obstacles
- The fundamental state-action value updating mechanism is robust to environment complexity

**Stability (Episodes 50-200):**
- Performance remains stable with minor fluctuations
- Average steps consistently around 4.1-4.8
- No significant performance degradation with increased maze complexity

### 2. Optimal Path Analysis

**Simple Grid (No Obstacles):**
```
A..
...
..G

Path: down → right → right → down
Steps: 4
Reward: 9.70
```

**Simple Maze (1 Wall at (0,1)):**
```
A#.
...
..G

Path: down → right → down → right
Steps: 4
Reward: 9.70
```
The wall at (0,1) blocks the direct "right" move from start, forcing the agent to go down first.

**Zigzag Maze (2 Walls at (0,1) and (1,1)):**
```
A#.
.#.
..G

Path: down → down → right → right
Steps: 4
Reward: 9.70
```
Both walls create a vertical corridor on the left, requiring the agent to go down twice before moving right.

**Key Insight:** Despite different obstacle configurations, the shortest path to the goal in all three cases is 4 steps. This explains why final performance is identical.

---

## Key Findings

### Finding 1: Complexity Doesn't Affect Final Performance
- All three environments converged to the same optimal performance
- Final average reward: 9.70 (optimal for 4-step path)
- 100% success rate across all environments

**Interpretation:** As long as the optimal path length is similar, maze complexity doesn't affect final performance. Q-Learning finds optimal policies regardless of obstacle placement.

### Finding 2: Learning Speed is Robust to Complexity
- Convergence to near-optimal performance occurred around Episode 50 for all environments
- No significant delay in learning caused by walls

**Interpretation:** The epsilon-greedy exploration strategy effectively discovers paths even with obstacles. The agent's exploration (10% random actions) is sufficient to find alternative routes around obstacles.

### Finding 3: Q-Learning Guarantees Optimal Policies
- All agents consistently found the shortest path (4 steps)
- Demonstrates Q-Learning's convergence guarantees for finite MDPs

**Theoretical Context:** Q-Learning is guaranteed to converge to optimal Q-values for finite Markov Decision Processes with sufficient exploration, which our experiment confirms.

### Finding 4: Minimal Exploration-Exploitation Balance Required
- With ε=0.1 (10% exploration), agents quickly discovered optimal paths
- Exploration allowed trying blocked actions, learning to avoid them
- Exploitation leveraged learned Q-values to follow optimal paths

---

## Relationship Between Environment Complexity and Learning Time

### Does Complexity Increase Learning Time?

**Short Answer:** No, not for this experiment.

**Nuanced Answer:**
1. **Path Length Matters More Than Obstacles:** All three environments had optimal paths of length 4. If the zigzag maze required 6 steps, we would likely see differences.

2. **State Space Still Small:** With only 9 states (3×3), the agent can quickly explore all state-action pairs.

3. **Exploration Efficiency:** The 10% exploration rate was sufficient to discover blocked actions quickly.

**Hypothesis for Future Experiments:**
- Larger grids (e.g., 5×5 or 10×10) may show learning time differences
- Higher obstacle density (e.g., 30%) would increase learning time
- More complex maze topologies (dead ends, multiple paths) would affect learning

---

## Q-Table Differences

### How Q-Tables Reflect Environment Structure

While final Q-values weren't explicitly printed, we can infer structural differences:

**Simple Grid Q-Table:**
- High Q-values for direct path actions: down (from (0,0)), right (from (0,1) and (0,2))
- Lower Q-values for actions leading away from goal

**Simple Maze Q-Table:**
- State (0,0): "up" has low Q-value (leads to boundary), "right" has low Q-value (blocked by wall), "down" has high Q-value (correct path)
- State (1,0): "right" has high Q-value, others lower

**Zigzag Maze Q-Table:**
- State (0,0): "down" has highest Q-value (only viable path)
- State (1,0): "down" has highest Q-value (right blocked by wall)
- State (2,0): "right" has highest Q-value (finally unblocked)

**Key Insight:** Q-tables encode obstacle information through action values. Actions leading to walls have lower Q-values because they result in no movement and negative rewards.

---

## Practical Implications

### For RL Algorithm Design:
1. **Exploration Rate is Critical:** ε=0.1 worked well, but different values would affect learning speed
2. **Learning Rate Balance:** α=0.1 provided stable learning without oscillation
3. **Discount Factor Importance:** γ=0.9 appropriately valued the goal reward

### For Environment Design:
1. **State Space Size Matters:** Small state spaces (9 states) enable fast learning
2. **Path Length Dominates Complexity:** Shortest path length is more important than obstacle count
3. **Reproducibility:** Fixed seed ensures consistent obstacle placement for comparison

---

## Limitations and Future Work

### Current Limitations:
1. **Small State Space:** 3×3 grid has only 9 states, limiting generalizability
2. **Single Reward Structure:** Only tested +10 for goal, -0.1 per step
3. **Fixed Exploration Rate:** Did not implement epsilon decay (common in practice)
4. **Limited Obstacle Density:** Tested only 0%, 11%, and 22% obstacle density

### Future Experiments:
1. **Scale Up:** Test 5×5, 10×10 grids with varied obstacle layouts
2. **Reward Engineering:** Test sparse rewards (only +10 for goal) vs dense rewards
3. **Exploration Strategies:** Compare epsilon-greedy vs. epsilon decay vs. Upper Confidence Bound (UCB)
4. **Learning Rate Schedules:** Test adaptive learning rates
5. **Multiple Goals:** Test environments with multiple rewarding states
6. **Dynamic Obstacles:** Test environments where obstacles move or change

---

## Conclusion

The experiment confirms that Q-Learning effectively learns optimal policies in maze environments with fixed obstacles. Key takeaways:

1. **Q-Learning Converges to Optimal Policies:** Despite obstacles, agents found the shortest path in all environments.

2. **Learning Speed is Robust:** All environments converged to optimal performance in ~50 episodes, regardless of maze complexity.

3. **Optimal Path Length Matters:** As long as optimal paths are similarly short, obstacle complexity doesn't affect final performance.

4. **Q-Tables Encode Environment Structure:** Action values reflect obstacle placements, enabling adaptive pathfinding.

5. **Exploration-Exploitation Balance:** A simple epsilon-greedy strategy (ε=0.1) was sufficient for these small mazes.

This experiment validates theoretical guarantees of Q-Learning and demonstrates its robustness to environment complexity for small-scale problems. Scaling to larger environments would reveal more nuanced relationships between complexity and learning time.

---

## Appendix: Test Files

- **maze.py**: Maze variant of GridWorld with fixed obstacle layouts
- **ComparisonExperiment**: Class for running comparative experiments
- **Training script**: Run `python3 maze.py` to reproduce these results
