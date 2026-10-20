"""
Comprehensive Unit Tests for RL Grid World Project
===================================================
Tests all core components:
1. Environment initialization and reset
2. Valid movement mechanics
3. Boundary collision handling
4. Obstacle collision handling
5. Goal arrival and termination enforcement
6. Tabular Q-table shape and state indexing
7. Bellman Q-learning update equation
8. Terminal-state transition handling (no future bootstrapping)
9. Epsilon decay and lower-bound enforcement
10. End-to-end execution of a short training session
"""

import unittest
import numpy as np
from main import (
    GridWorld,
    QLearningAgent,
    train_agent,
    evaluate_agent,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    REWARD_STEP,
    REWARD_OBSTACLE,
    REWARD_BOUNDARY,
    REWARD_GOAL,
)


class TestGridWorld(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld()

    def test_environment_initialization_and_reset(self):
        """Test that reset returns (0,0) and clears step counts and termination status."""
        state = self.env.reset()
        self.assertEqual(state, (0, 0))
        self.assertEqual(self.env.step_count, 0)
        self.assertFalse(self.env.done)

    def test_valid_movement(self):
        """Test valid movement inside open grid cells."""
        self.env.reset()
        # Move DOWN from (0, 0) -> (1, 0)
        next_state, reward, done, info = self.env.step(ACTION_DOWN)
        self.assertEqual(next_state, (1, 0))
        self.assertEqual(reward, REWARD_STEP)
        self.assertFalse(done)
        self.assertEqual(self.env.step_count, 1)

    def test_boundary_handling(self):
        """Test out-of-bounds attempts yield -5 penalty and keep agent in place."""
        self.env.reset()
        # Attempt UP from (0, 0) -> out of bounds
        next_state, reward, done, info = self.env.step(ACTION_UP)
        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, REWARD_BOUNDARY)
        self.assertFalse(done)
        self.assertEqual(info.get("event"), "boundary_collision")

        # Attempt LEFT from (0, 0) -> out of bounds
        next_state, reward, done, info = self.env.step(ACTION_LEFT)
        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, REWARD_BOUNDARY)

    def test_obstacle_handling(self):
        """Test obstacle collisions yield -5 penalty and prevent entry."""
        self.env.reset()
        # Obstacle at (1, 1). Move to (0, 1) first.
        s1, r1, _, _ = self.env.step(ACTION_RIGHT)
        self.assertEqual(s1, (0, 1))

        # Attempt moving DOWN into (1, 1)
        next_state, reward, done, info = self.env.step(ACTION_DOWN)
        self.assertEqual(next_state, (0, 1))  # Stays at (0, 1)
        self.assertEqual(reward, REWARD_OBSTACLE)
        self.assertFalse(done)
        self.assertEqual(info.get("event"), "obstacle_collision")

    def test_goal_reward_and_termination(self):
        """Test entering goal yields +10 reward, sets done=True, and stops further steps."""
        self.env.reset()
        # Place agent right next to goal (3, 4)
        self.env.state = (3, 4)
        next_state, reward, done, info = self.env.step(ACTION_DOWN)
        self.assertEqual(next_state, (4, 4))
        self.assertEqual(reward, REWARD_GOAL)
        self.assertTrue(done)
        self.assertEqual(info.get("event"), "reached_goal")

        # Further steps without reset should raise RuntimeError
        with self.assertRaises(RuntimeError):
            self.env.step(ACTION_RIGHT)

    def test_state_indexing_bijection(self):
        """Test state to index conversion matches row * 5 + col."""
        for r in range(5):
            for c in range(5):
                idx = self.env.state_to_index((r, c))
                self.assertEqual(idx, r * 5 + c)
                self.assertEqual(self.env.index_to_state(idx), (r, c))


class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QLearningAgent(num_states=25, num_actions=4, alpha=0.1, gamma=0.95,
                                    epsilon=1.0, min_epsilon=0.05, epsilon_decay=0.995, seed=42)

    def test_q_table_shape_and_initialization(self):
        """Verify Q-table has shape (25, 4) and starts with all zeros."""
        self.assertEqual(self.agent.q_table.shape, (25, 4))
        self.assertTrue(np.all(self.agent.q_table == 0.0))

    def test_q_learning_update_equation(self):
        """
        Verify Bellman optimality update:
        Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
        """
        state_idx = 0
        action = ACTION_DOWN
        reward = -1.0
        next_state_idx = 5
        
        # Manually set next state maximum Q-value
        self.agent.q_table[next_state_idx, :] = [1.0, 2.0, 0.5, 0.0]
        max_q_next = 2.0
        
        expected_target = reward + self.agent.gamma * max_q_next  # -1.0 + 0.95 * 2.0 = 0.90
        expected_q = 0.0 + self.agent.alpha * (expected_target - 0.0)  # 0.1 * 0.90 = 0.09
        
        self.agent.update(state_idx, action, reward, next_state_idx, done=False)
        self.assertAlmostEqual(self.agent.q_table[state_idx, action], expected_q, places=5)

    def test_terminal_state_update_no_bootstrapping(self):
        """Verify terminal state updates do not bootstrap future returns."""
        state_idx = 23  # (4, 3)
        action = ACTION_RIGHT
        reward = 10.0
        next_state_idx = 24  # (4, 4)
        
        # Even if next_state had non-zero values, done=True ignores them
        self.agent.q_table[next_state_idx, :] = [100.0, 100.0, 100.0, 100.0]
        
        expected_target = reward  # 10.0
        expected_q = 0.0 + self.agent.alpha * (expected_target - 0.0)  # 1.0
        
        self.agent.update(state_idx, action, reward, next_state_idx, done=True)
        self.assertAlmostEqual(self.agent.q_table[state_idx, action], expected_q, places=5)

    def test_epsilon_decay_and_minimum_bound(self):
        """Verify epsilon decays exponentially and never drops below min_epsilon."""
        initial_eps = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertAlmostEqual(self.agent.epsilon, initial_eps * 0.995, places=5)

        # Decay many times
        for _ in range(2000):
            self.agent.decay_epsilon()
            
        self.assertAlmostEqual(self.agent.epsilon, self.agent.min_epsilon, places=5)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.min_epsilon)


class TestTrainingIntegration(unittest.TestCase):
    def test_small_training_and_evaluation_run(self):
        """Verify complete training loop executes without error and records valid history."""
        env = GridWorld()
        agent = QLearningAgent(seed=42)
        
        history = train_agent(env, agent, num_episodes=50)
        self.assertEqual(len(history["episode"]), 50)
        self.assertEqual(len(history["reward"]), 50)
        self.assertEqual(len(history["steps"]), 50)
        self.assertEqual(len(history["reached_goal"]), 50)
        self.assertEqual(len(history["epsilon"]), 50)

        # Ensure Q-table has been modified from pure zeros
        self.assertTrue(np.any(agent.q_table != 0.0))

        # Test evaluation function (verify Q-table is not modified during evaluation)
        q_table_before = agent.q_table.copy()
        eval_metrics = evaluate_agent(env, agent, num_episodes=10)
        self.assertTrue(np.array_equal(agent.q_table, q_table_before))
        self.assertIn("success_rate", eval_metrics)
        self.assertIn("mean_reward", eval_metrics)
        self.assertIn("mean_steps", eval_metrics)


if __name__ == "__main__":
    unittest.main()
