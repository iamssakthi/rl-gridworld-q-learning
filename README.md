# Intelligent Grid World Navigation Using Q-Learning

An undergraduate-level Reinforcement Learning (RL) mini-project implementing tabular Q-learning to navigate an autonomous agent through a 5×5 Grid World with static obstacles to an optimal goal.

---

## 1. Project Abstract
Reinforcement Learning enables an agent to learn optimal decision-making policies through trial-and-error interactions with an environment. In this project, a **tabular Q-learning agent** is trained to navigate a discrete $5 \times 5$ Grid World containing stationary obstacles. The agent receives numerical feedback (+10 for reaching the goal, -1 for each step, and -5 for hitting boundaries or obstacles). Over 1,000 training episodes, the agent transitions from random exploration to deterministic exploitation, discovering the mathematical shortest obstacle-free trajectory (8 steps, cumulative reward $+3.00$) with a **100% success rate**.

---

## 2. Problem Statement & Objectives
Autonomous navigation in constrained environments requires path planning without prior hard-coded maps or external supervision. The objectives of this project are:
1. Formulate a discrete Grid World navigation task as a Markov Decision Process (MDP).
2. Implement tabular Q-learning and $\epsilon$-greedy exploration from scratch using NumPy and the Python standard library.
3. Train an agent to find the optimal path to the goal while evading obstacles.
4. Evaluate performance rigorously against an untrained baseline across 100 greedy evaluation episodes.
5. Provide automated test verification and visualization tools suitable for academic presentation.

---

## 3. Reinforcement Learning Formulation

### Environment Specifications:
- **Grid Dimensions**: $5 \times 5$ matrix ($25$ discrete cells).
- **Coordinate System**: `(row, col)` with $0 \le \text{row} \le 4$ and $0 \le \text{col} \le 4$.
- **Start State**: Cell `(0, 0)` (top-left).
- **Goal State**: Cell `(4, 4)` (bottom-right).
- **Obstacles**: Static hazards located at `(1, 1)`, `(1, 2)`, and `(3, 2)`.

### State Space ($\mathcal{S}$):
The state space consists of 25 discrete positions mapped to unique scalar indices:
$$\text{State Index} = \text{row} \times 5 + \text{col} \quad \in \{0, 1, \dots, 24\}$$

### Action Space ($\mathcal{A}$):
The agent has four discrete directional movement actions:
- `0 = UP`    $(\Delta\text{row} = -1, \Delta\text{col} = 0)$
- `1 = DOWN`  $(\Delta\text{row} = +1, \Delta\text{col} = 0)$
- `2 = LEFT`  $(\Delta\text{row} = 0, \Delta\text{col} = -1)$
- `3 = RIGHT` $(\Delta\text{row} = 0, \Delta\text{col} = +1)$

### Reward Function ($\mathcal{R}$):
| Event | Reward | Next Agent State | Termination |
| :--- | :---: | :---: | :---: |
| **Normal valid move** | `-1.0` | Target cell | No |
| **Obstacle collision** | `-5.0` | Remains in current cell | No |
| **Grid boundary collision** | `-5.0` | Remains in current cell | No |
| **Goal reached** | `+10.0` | Goal cell `(4, 4)` | **Yes (Episode Ends)** |

Each episode has a strict cutoff of **50 maximum steps** to prevent infinite loops.

---

## 4. Q-Learning Algorithm & Exploration Strategy

> **Note on Architecture**: This project implements **Tabular Q-Learning** (Model-Free, Off-Policy RL). It does **not** use deep neural networks, external APIs, or GPU acceleration. The entire policy is stored in a $25 \times 4$ numerical Q-table.

### Bellman Optimality Equation:
The state-action value $Q(s, a)$ is updated according to the temporal difference (TD) rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]$$

For terminal states (goal reached), future discounted reward is zero:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r - Q(s, a) \Big]$$

- **Learning Rate ($\alpha = 0.1$)**: Controls the rate of value updates.
- **Discount Factor ($\gamma = 0.95$)**: Balances immediate versus future discounted rewards.

### $\epsilon$-Greedy Action Selection:
During training, the agent balances exploration and exploitation:
- With probability $\epsilon$, select a uniform random action $a \in \mathcal{A}$.
- With probability $1 - \epsilon$, select greedy action $a^* = \arg\max_{a} Q(s, a)$.

### Exploration Decay Schedule:
- **Initial Epsilon ($\epsilon_0$)**: $1.00$ (100% exploration).
- **Decay Rate**: $\epsilon \leftarrow \max(\epsilon_{\min}, \epsilon \times 0.995)$ after each episode.
- **Minimum Epsilon ($\epsilon_{\min}$)**: $0.05$ (retains 5% background exploration).

---

## 5. Project Structure

```
RL_GridWorld_Project/
├── app.py               # Streamlit interactive learning lab and demonstration UI
├── main.py              # Environment, Q-learning agent, training & evaluation loops
├── visualization.py     # High-resolution plotting for training curve & grid navigation
├── test_project.py      # Automated unit tests (11 test cases)
├── requirements.txt     # Minimal dependencies (numpy, matplotlib, streamlit)
├── README.md            # Comprehensive project documentation
└── results/
    ├── training_curve.png   # Episode rewards and 50-episode moving average
    ├── evaluation.png       # 5x5 Grid World visualization with greedy trajectory
    └── metrics.csv          # Episode logs & evaluation summary table
```

---

## 6. Setup and Execution Instructions

### Prerequisites:
- Python 3.9+
- Standard virtual environment or pip installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Automated Unit Tests
Verify environment physics, boundary constraints, and Q-update mathematics:
```bash
python -m unittest test_project.py
```
*Expected output: `Ran 11 tests in ~0.25s ... OK`*

### Step 3: Train and Evaluate the Agent (CLI)
Execute complete training (1,000 episodes), evaluation (100 greedy episodes), and visualization generation:
```bash
python main.py
```
Generated plots and CSV will be saved into the `results/` folder.

### Step 4: Launch Interactive Web UI
Launch the interactive Streamlit dashboard for college viva/demo:
```bash
streamlit run app.py
```

---

## 7. Experimental Results & Performance Comparison

The trained Q-learning agent was evaluated across **100 consecutive greedy episodes** ($\epsilon = 0$, exploration disabled) and compared directly against a random action baseline:

| Evaluation Metric | Trained Q-Learning Agent | Random Action Baseline |
| :--- | :---: | :---: |
| **Goal Success Rate** | **100.0%** | 29.0% |
| **Mean Cumulative Reward** | **+3.00 ± 0.00** | -110.52 ± 35.77 |
| **Mean Steps per Episode** | **8.00** | 45.11 |
| **Minimum Steps to Goal** | **8 (Optimal)** | 12 |

### Learned Optimal Route:
The agent learned the following 8-step path:
$$\mathbf{(0,0) \xrightarrow{DOWN} (1,0) \xrightarrow{DOWN} (2,0) \xrightarrow{RIGHT} (2,1) \xrightarrow{RIGHT} (2,2) \xrightarrow{RIGHT} (2,3) \xrightarrow{DOWN} (3,3) \xrightarrow{DOWN} (4,3) \xrightarrow{RIGHT} (4,4)}$$

#### Mathematical Verification of Reward:
- 7 normal movements: $7 \times (-1.0) = -7.0$
- 1 goal arrival step: $+10.0$
- Total cumulative episode reward: $-7.0 + 10.0 = \mathbf{+3.00}$ (Global Optimum).

---

## 8. Interpreting the Generated Visualizations

### 1. Training Curve (`results/training_curve.png`)
- **Raw Episode Rewards (Light Gray)**: High variance during initial episodes ($\epsilon \approx 1.0$) as the agent explores obstacles and boundaries.
- **50-Episode Moving Average (Solid Blue)**: Shows rapid convergence from initial negative penalties ($\sim -130$) to near-optimal rewards ($\sim +3.00$) between episodes 300 and 500.
- **Epsilon Decay (Dotted Amber)**: Demonstrates the gradual reduction of random exploration towards exploitation.

### 2. Grid World Evaluation (`results/evaluation.png`)
- **Green Cell `(0,0)`**: Starting position.
- **Gold Cell `(4,4)`**: Goal state with $+10$ reward.
- **Red Cells `(1,1), (1,2), (3,2)`**: Obstacles with $-5$ penalty.
- **Blue Arrows & Numbered Nodes (0 to 8)**: The exact sequence of decisions chosen by the greedy policy, seamlessly navigating through the narrow corridor between obstacles without collision.

---

## 9. Viva / College Presentation Cheat-Sheet (5-Minute Guide)

When presenting this project to an examiner or audience, structure your talk as follows:

1. **Introduction (1 min)**:
   - State the project title: *"Intelligent Grid World Navigation Using Q-Learning."*
   - Define RL: An agent learns mapping from states to actions to maximize cumulative reward through trial-and-error.

2. **Environment & Formulation (1 min)**:
   - $5 \times 5$ discrete grid, 25 states, 4 actions (UP, DOWN, LEFT, RIGHT).
   - Penalties for collisions (-5) and steps (-1), positive reward (+10) for reaching the goal.
   - Mention that the environment enforces the rules; the path is **not** pre-programmed.

3. **Algorithm & Training (1.5 min)**:
   - Explain tabular Q-learning with a $25 \times 4$ matrix.
   - Mention the Bellman update equation and how temporal difference (TD) error drives learning.
   - Highlight the $\epsilon$-greedy schedule: starting with 100% exploration and exponentially decaying to 5%.

4. **Results & Baseline Comparison (1 min)**:
   - Compare the trained agent (100% success rate, 8 steps, +3.00 reward) with the random baseline (29% success, 45 steps, -110 reward).
   - Show `evaluation.png` and trace the 8-step optimal path avoiding obstacles.

5. **Key Takeaway & Conclusion (30 sec)**:
   - Tabular Q-learning successfully converges to the globally optimal policy in finite discrete MDPs without requiring complex neural networks or external supervision.

---

## 10. Limitations & Future Extensions
- **Scalability**: Tabular representations become intractable in continuous or large state spaces (curse of dimensionality); Deep Q-Networks (DQN) would be needed for larger grids.
- **Static vs. Dynamic Obstacles**: Current obstacles are fixed; future work could introduce moving hazards using stochastic transitions.
- **Multi-Goal or Variable Starts**: The formulation can be extended to randomize start/goal states across episodes to test generalized policy learning.
