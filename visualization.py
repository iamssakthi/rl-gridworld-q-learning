"""
Visualization Module for RL Grid World Project
==============================================
Generates publication-quality figures:
1. results/training_curve.png: Episode rewards and moving average over time.
2. results/evaluation.png: 5x5 Grid World visualization with obstacles and learned path.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def compute_moving_average(data, window_size=50):
    """Computes simple moving average for smoothing noisy episode reward curves."""
    if len(data) < window_size:
        return data
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode="valid")
    # Pad beginning so that array lengths match
    pad = np.full(window_size - 1, smoothed[0])
    return np.concatenate((pad, smoothed))


def plot_training_curve(training_history, save_path):
    """
    Plots the raw episode rewards alongside a moving average and exploration decay.
    """
    episodes = training_history["episode"]
    rewards = training_history["reward"]
    epsilons = training_history["epsilon"]

    moving_avg_rewards = compute_moving_average(rewards, window_size=50)

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Plot raw rewards with light transparency
    ax1.plot(episodes, rewards, color="#94a3b8", alpha=0.35, linewidth=0.8, label="Raw Episode Reward")
    
    # Plot moving average with prominent dark blue line
    ax1.plot(episodes, moving_avg_rewards, color="#1e3a8a", linewidth=2.2, label="50-Episode Moving Average")
    
    ax1.set_xlabel("Training Episode", fontsize=12, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Total Reward per Episode", fontsize=12, fontweight="bold", color="#1e3a8a", labelpad=8)
    ax1.tick_params(axis="y", labelcolor="#1e3a8a")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Twin axis for Epsilon decay
    ax2 = ax1.twinx()
    ax2.plot(episodes, epsilons, color="#d97706", linestyle=":", linewidth=1.8, label="Epsilon (Exploration Rate)")
    ax2.set_ylabel("Exploration Rate (ε)", fontsize=12, fontweight="bold", color="#d97706", labelpad=8)
    ax2.tick_params(axis="y", labelcolor="#d97706")
    ax2.set_ylim(-0.02, 1.05)

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.92, fontsize=10)

    plt.title("Q-Learning Convergence: Training Reward & Epsilon Decay", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_grid_and_route(env, path, save_path, eval_summary=None):
    """
    Renders the 5x5 Grid World environment showing:
    - Start cell (0, 0)
    - Goal cell (4, 4)
    - Obstacle cells (1, 1), (1, 2), (3, 2)
    - Optimal trajectory with arrows and step markers
    - Key evaluation metrics banner
    """
    grid_size = env.grid_size
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # Draw grid background and cells
    for r in range(grid_size):
        for c in range(grid_size):
            coord = (r, c)
            
            # Determine cell color
            if coord == env.start_state:
                facecolor = "#dcfce7"  # Light green
                edgecolor = "#15803d"
            elif coord == env.goal_state:
                facecolor = "#fef08a"  # Light yellow/gold
                edgecolor = "#a16207"
            elif coord in env.obstacles:
                facecolor = "#fecaca"  # Light red/obstacle
                edgecolor = "#b91c1c"
            else:
                facecolor = "#f8fafc"  # Neutral cell
                edgecolor = "#cbd5e1"

            # Draw cell rectangle (Note: row goes down, col goes right)
            # Matplotlib coordinates: x = col, y = (grid_size - 1 - r)
            x_pos = c
            y_pos = grid_size - 1 - r

            rect = patches.Rectangle(
                (x_pos, y_pos), 1, 1,
                linewidth=1.5,
                edgecolor=edgecolor,
                facecolor=facecolor
            )
            ax.add_patch(rect)

            # Cell coordinate label
            ax.text(
                x_pos + 0.08, y_pos + 0.88,
                f"({r},{c})",
                fontsize=8,
                color="#64748b",
                fontfamily="monospace"
            )

            # Special cell labels
            if coord == env.start_state:
                ax.text(
                    x_pos + 0.5, y_pos + 0.5, "START\n(0,0)",
                    ha="center", va="center", fontsize=11, fontweight="bold", color="#166534"
                )
            elif coord == env.goal_state:
                ax.text(
                    x_pos + 0.5, y_pos + 0.5, "GOAL\n(+10)",
                    ha="center", va="center", fontsize=11, fontweight="bold", color="#854d0e"
                )
            elif coord in env.obstacles:
                ax.text(
                    x_pos + 0.5, y_pos + 0.5, "OBSTACLE\n(-5)",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="#991b1b"
                )

    # Draw learned route path with arrows
    if path and len(path) > 1:
        for idx in range(len(path) - 1):
            r1, c1 = path[idx]
            r2, c2 = path[idx + 1]

            x1 = c1 + 0.5
            y1 = grid_size - 1 - r1 + 0.5
            x2 = c2 + 0.5
            y2 = grid_size - 1 - r2 + 0.5

            # Arrow vector
            dx = (x2 - x1) * 0.75
            dy = (y2 - y1) * 0.75

            ax.annotate(
                "",
                xy=(x1 + dx, y1 + dy),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.4,head_length=0.6",
                    lw=2.8,
                    color="#2563eb"
                )
            )

            # Step number badge
            ax.scatter([x1], [y1], color="#1d4ed8", s=130, zorder=5)
            ax.text(
                x1, y1, str(idx),
                ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=6
            )

        # Final marker at goal
        r_last, c_last = path[-1]
        x_last = c_last + 0.5
        y_last = grid_size - 1 - r_last + 0.5
        ax.scatter([x_last], [y_last], color="#15803d", s=150, zorder=5)
        ax.text(
            x_last, y_last, str(len(path) - 1),
            ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=6
        )

    # Axes configuration
    ax.set_xlim(-0.05, grid_size + 0.05)
    ax.set_ylim(-0.05, grid_size + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title & evaluation summary box
    summary_text = "Trained Policy: Optimal Route Found"
    if eval_summary:
        summary_text = (
            f"Evaluation (100 Greedy Episodes):  "
            f"Success: {eval_summary['success_rate']:.0f}%  |  "
            f"Mean Reward: {eval_summary['mean_reward']:.1f}  |  "
            f"Optimal Steps: {eval_summary['min_steps']}"
        )

    plt.suptitle("5x5 Grid World Environment & Learned Optimal Path", fontsize=14, fontweight="bold", y=0.96)
    ax.set_title(summary_text, fontsize=10, fontweight="medium", color="#334155", pad=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
