"""
Intelligent Grid World Navigation Using Q-Learning
===================================================
A complete, self-contained Reinforcement Learning project demonstrating tabular
Q-learning in a discrete 5x5 Grid World environment with obstacles.

Author: Academic RL Mini-Project
"""

import os
import csv
import numpy as np

# Configurable Hyperparameters and Simulation Constants
GRID_SIZE = 5
START_STATE = (0, 0)
GOAL_STATE = (4, 4)
OBSTACLES = ((1, 1), (1, 2), (3, 2))

# Movement Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

# Rewards
REWARD_STEP = -1.0
REWARD_OBSTACLE = -5.0
REWARD_BOUNDARY = -5.0
REWARD_GOAL = 10.0

# Q-Learning Hyperparameters
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 50
ALPHA = 0.1             # Learning rate
GAMMA = 0.95            # Discount factor
EPSILON_START = 1.0     # Initial exploration rate
EPSILON_MIN = 0.05      # Minimum exploration rate
EPSILON_DECAY = 0.995   # Multiplicative decay per episode
RANDOM_SEED = 42


class GridWorld:
    """
    5x5 Grid World Environment.
    
    Coordinates: (row, col) where row in [0, 4] and col in [0, 4].
    - Start: (0, 0)
    - Goal: (4, 4)
    - Obstacles: (1, 1), (1, 2), (3, 2)
    
    Reward Rules:
    - Normal valid movement: -1.0
    - Attempted obstacle collision: -5.0 (agent stays in place)
    - Attempted out-of-bounds move: -5.0 (agent stays in place)
    - Reaching the goal: +10.0 (episode terminates)
    """

    def __init__(self, grid_size=GRID_SIZE, start=START_STATE, goal=GOAL_STATE,
                 obstacles=OBSTACLES, max_steps=MAX_STEPS_PER_EPISODE):
        self.grid_size = grid_size
        self.start_state = start
        self.goal_state = goal
        self.obstacles = set(obstacles)
        self.max_steps = max_steps
        
        self.state = self.start_state
        self.step_count = 0
        self.done = False

    def reset(self):
        """Resets the environment to initial state and returns start state."""
        self.state = self.start_state
        self.step_count = 0
        self.done = False
        return self.state

    def step(self, action):
        """
        Executes an action in the environment.
        
        Args:
            action (int): 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is already finished. Call reset() before stepping again.")

        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}. Must be one of {list(ACTION_DELTAS.keys())}")

        self.step_count += 1
        delta_r, delta_c = ACTION_DELTAS[action]
        curr_r, curr_c = self.state
        next_r, next_c = curr_r + delta_r, curr_c + delta_c
        target = (next_r, next_c)

        # Boundary collision check
        if not (0 <= next_r < self.grid_size and 0 <= next_c < self.grid_size):
            reward = REWARD_BOUNDARY
            info = {"event": "boundary_collision"}
        # Obstacle collision check
        elif target in self.obstacles:
            reward = REWARD_OBSTACLE
            info = {"event": "obstacle_collision"}
        # Reaching goal check
        elif target == self.goal_state:
            self.state = target
            reward = REWARD_GOAL
            self.done = True
            info = {"event": "reached_goal"}
            return self.state, reward, self.done, info
        # Normal valid move
        else:
            self.state = target
            reward = REWARD_STEP
            info = {"event": "valid_move"}

        # Check maximum steps timeout
        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            info["timeout"] = True

        return self.state, reward, self.done, info

    def state_to_index(self, state):
        """Converts (row, col) coordinate tuple to flat integer state index: row * 5 + col."""
        return state[0] * self.grid_size + state[1]

    def index_to_state(self, idx):
        """Converts flat integer state index back to (row, col) coordinate tuple."""
        return (idx // self.grid_size, idx % self.grid_size)


class QLearningAgent:
    """
    Tabular Q-Learning Agent.
    
    Maintains a 2D Q-table of size (num_states, num_actions) initialized to 0.
    Selects actions using epsilon-greedy exploration during training,
    and pure greedy action selection during evaluation.
    """

    def __init__(self, num_states=25, num_actions=4, alpha=ALPHA, gamma=GAMMA,
                 epsilon=EPSILON_START, min_epsilon=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY, seed=RANDOM_SEED):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        # Initialize tabular Q-table with zeros
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        self.rng = np.random.RandomState(seed)

    def choose_action(self, state_idx, training=True):
        """
        Chooses an action using epsilon-greedy policy.
        
        Args:
            state_idx (int): State index in [0, 24].
            training (bool): If True, applies epsilon exploration. If False, greedy action.
            
        Returns:
            int: Selected action index.
        """
        if training and (self.rng.rand() < self.epsilon):
            # Exploration: Uniform random action
            return self.rng.randint(self.num_actions)
        else:
            # Exploitation: Greedy action with tie-breaking
            q_values = self.q_table[state_idx]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            # Break ties uniformly among actions with identical optimal Q-value
            return self.rng.choice(best_actions)

    def update(self, state_idx, action, reward, next_state_idx, done):
        """
        Updates Q-value using the standard Bellman Optimality Equation:
        Q(s, a) <- Q(s, a) + alpha * [ Target - Q(s, a) ]
        
        where:
        Target = reward + gamma * max_a' Q(s', a')   (if state is non-terminal)
        Target = reward                              (if state is terminal)
        """
        current_q = self.q_table[state_idx, action]
        
        if done:
            # Terminal state: No future discounted return
            target = reward
        else:
            # Non-terminal state: Bootstrap from optimal next-state action value
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])
            
        temporal_difference_error = target - current_q
        self.q_table[state_idx, action] += self.alpha * temporal_difference_error

    def decay_epsilon(self):
        """Multiplicative exponential decay of epsilon bounded below by min_epsilon."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_agent(env, agent, num_episodes=NUM_EPISODES):
    """
    Trains the Q-learning agent through repeated environment interaction.
    
    Returns:
        dict: Training history including rewards, steps, goals reached, and epsilons.
    """
    history = {
        "episode": [],
        "reward": [],
        "steps": [],
        "reached_goal": [],
        "epsilon": []
    }

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        reached_goal = False
        done = False

        while not done:
            state_idx = env.state_to_index(state)
            action = agent.choose_action(state_idx, training=True)
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            # Tabular Q-update
            agent.update(state_idx, action, reward, next_state_idx, done)

            state = next_state
            total_reward += reward
            steps += 1
            if info.get("event") == "reached_goal":
                reached_goal = True

        current_eps = agent.epsilon
        agent.decay_epsilon()

        history["episode"].append(ep)
        history["reward"].append(total_reward)
        history["steps"].append(steps)
        history["reached_goal"].append(1 if reached_goal else 0)
        history["epsilon"].append(current_eps)

    return history


def evaluate_agent(env, agent, num_episodes=100, seed=123):
    """
    Evaluates the trained agent with exploration disabled (epsilon = 0).
    
    Returns:
        dict: Summary metrics across all evaluation episodes.
    """
    # Create evaluation RNG for deterministic, reproducible tie-breaking
    eval_rng = np.random.RandomState(seed)
    
    rewards = []
    steps_list = []
    successes = []

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        reached_goal = False
        done = False

        while not done:
            state_idx = env.state_to_index(state)
            # Greedy action without exploration
            q_values = agent.q_table[state_idx]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            action = eval_rng.choice(best_actions)

            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
            if info.get("event") == "reached_goal":
                reached_goal = True

        rewards.append(total_reward)
        steps_list.append(steps)
        successes.append(1 if reached_goal else 0)

    successful_steps = [s for s, succ in zip(steps_list, successes) if succ == 1]
    min_steps = int(np.min(successful_steps)) if successful_steps else None

    return {
        "num_episodes": num_episodes,
        "success_rate": float(np.mean(successes) * 100.0),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "min_steps": min_steps,
    }


def evaluate_random_baseline(env, num_episodes=100, seed=999):
    """
    Evaluates an untrained random action baseline for performance comparison.
    
    Returns:
        dict: Baseline summary metrics.
    """
    rng = np.random.RandomState(seed)
    rewards = []
    steps_list = []
    successes = []

    for _ in range(num_episodes):
        env.reset()
        total_reward = 0.0
        steps = 0
        reached_goal = False
        done = False

        while not done:
            action = rng.randint(4)
            _, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if info.get("event") == "reached_goal":
                reached_goal = True

        rewards.append(total_reward)
        steps_list.append(steps)
        successes.append(1 if reached_goal else 0)

    successful_steps = [s for s, succ in zip(steps_list, successes) if succ == 1]
    min_steps = int(np.min(successful_steps)) if successful_steps else None

    return {
        "num_episodes": num_episodes,
        "success_rate": float(np.mean(successes) * 100.0),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "min_steps": min_steps,
    }


def get_greedy_path(env, agent):
    """
    Traces the deterministic greedy trajectory of the trained agent from start to goal.
    
    Returns:
        list of tuples: Coordinate trajectory [(0,0), ...]
    """
    state = env.reset()
    path = [state]
    done = False
    visited = set([state])
    steps = 0

    while not done and steps < env.max_steps:
        state_idx = env.state_to_index(state)
        # Deterministic greedy choice (pick the first max_q action for reproducibility)
        q_vals = agent.q_table[state_idx]
        best_action = int(np.argmax(q_vals))

        next_state, _, done, _ = env.step(best_action)
        path.append(next_state)
        state = next_state
        steps += 1
        if next_state in visited and next_state != env.goal_state:
            # Detected loop (safeguard)
            break
        visited.add(next_state)

    return path


def save_metrics_to_csv(training_history, eval_metrics, baseline_metrics, csv_path):
    """
    Saves episode-by-episode training data and evaluation summaries into a CSV file.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Section 1: Summary Evaluation Metrics
        writer.writerow(["# EVALUATION SUMMARY (100 EPISODES)"])
        writer.writerow(["Metric", "Trained Q-Learning Agent", "Random Action Baseline"])
        writer.writerow(["Success Rate (%)", f"{eval_metrics['success_rate']:.2f}", f"{baseline_metrics['success_rate']:.2f}"])
        writer.writerow(["Mean Reward", f"{eval_metrics['mean_reward']:.2f}", f"{baseline_metrics['mean_reward']:.2f}"])
        writer.writerow(["Std Reward", f"{eval_metrics['std_reward']:.2f}", f"{baseline_metrics['std_reward']:.2f}"])
        writer.writerow(["Mean Steps", f"{eval_metrics['mean_steps']:.2f}", f"{baseline_metrics['mean_steps']:.2f}"])
        writer.writerow(["Min Steps to Goal", str(eval_metrics['min_steps']), str(baseline_metrics['min_steps'])])
        writer.writerow([])
        
        # Section 2: Episode-level Training Log
        writer.writerow(["# TRAINING EPISODE LOG"])
        writer.writerow(["episode", "total_reward", "steps", "reached_goal", "epsilon"])
        for i in range(len(training_history["episode"])):
            writer.writerow([
                training_history["episode"][i],
                f"{training_history['reward'][i]:.2f}",
                training_history["steps"][i],
                training_history["reached_goal"][i],
                f"{training_history['epsilon'][i]:.4f}",
            ])


def main():
    print("=" * 60)
    print(" Intelligent Grid World Navigation Using Q-Learning")
    print("=" * 60)
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Start: {START_STATE} | Goal: {GOAL_STATE}")
    print(f"Obstacles: {OBSTACLES}")
    print(f"Hyperparameters: Episodes={NUM_EPISODES}, Alpha={ALPHA}, Gamma={GAMMA}, Decay={EPSILON_DECAY}")
    print("-" * 60)

    # 1. Initialize Environment and Agent
    env = GridWorld()
    agent = QLearningAgent()

    # 2. Train Agent
    print(f"[*] Training agent for {NUM_EPISODES} episodes...")
    training_history = train_agent(env, agent, num_episodes=NUM_EPISODES)
    print("[+] Training completed successfully.")

    # 3. Evaluate Trained Agent
    print("[*] Evaluating trained agent across 100 greedy episodes (exploration disabled)...")
    eval_metrics = evaluate_agent(env, agent, num_episodes=100)
    print(f"    - Success Rate: {eval_metrics['success_rate']:.1f}%")
    print(f"    - Mean Reward:  {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}")
    print(f"    - Mean Steps:   {eval_metrics['mean_steps']:.2f}")
    print(f"    - Min Steps:    {eval_metrics['min_steps']}")

    # 4. Evaluate Baseline
    print("[*] Evaluating random baseline agent for 100 episodes...")
    baseline_metrics = evaluate_random_baseline(env, num_episodes=100)
    print(f"    - Baseline Success Rate: {baseline_metrics['success_rate']:.1f}%")
    print(f"    - Baseline Mean Reward:  {baseline_metrics['mean_reward']:.2f}")
    print(f"    - Baseline Mean Steps:   {baseline_metrics['mean_steps']:.2f}")

    # 5. Extract Greedy Path
    greedy_path = get_greedy_path(env, agent)
    print(f"[*] Optimal learned greedy trajectory ({len(greedy_path) - 1} steps):")
    print("    " + " -> ".join([str(p) for p in greedy_path]))

    # 6. Save Metrics to CSV
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "metrics.csv")
    save_metrics_to_csv(training_history, eval_metrics, baseline_metrics, csv_path)
    print(f"[+] Saved metrics CSV to {csv_path}")

    # 7. Generate Visualizations
    from visualization import plot_training_curve, plot_grid_and_route
    curve_path = os.path.join(results_dir, "training_curve.png")
    grid_path = os.path.join(results_dir, "evaluation.png")
    
    print("[*] Generating visualizations...")
    plot_training_curve(training_history, curve_path)
    plot_grid_and_route(env, greedy_path, grid_path, eval_summary=eval_metrics)
    print(f"[+] Saved training curve to {curve_path}")
    print(f"[+] Saved grid evaluation to {grid_path}")
    print("=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
