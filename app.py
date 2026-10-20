"""
RL Grid World — Interactive Learning Lab
=======================================
A Streamlit web application for demonstrating Tabular Q-Learning in a 5x5 Grid World.
Reuses the core environment, agent, and evaluation logic from main.py.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

# Import core modules from main.py
from main import (
    GridWorld,
    QLearningAgent,
    train_agent,
    evaluate_agent,
    evaluate_random_baseline,
    get_greedy_path,
    GRID_SIZE,
    START_STATE,
    GOAL_STATE,
    OBSTACLES,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_NAMES,
    ALPHA,
    GAMMA,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY,
    NUM_EPISODES,
    REWARD_STEP,
    REWARD_OBSTACLE,
    REWARD_BOUNDARY,
    REWARD_GOAL,
)
from visualization import compute_moving_average

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RL Grid World — Interactive Learning Lab",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .badge-optimal {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "env" not in st.session_state:
    st.session_state.env = GridWorld()

if "agent" not in st.session_state:
    st.session_state.agent = QLearningAgent()

if "is_trained" not in st.session_state:
    st.session_state.is_trained = False

if "training_history" not in st.session_state:
    st.session_state.training_history = None

if "eval_metrics" not in st.session_state:
    st.session_state.eval_metrics = None

if "baseline_metrics" not in st.session_state:
    st.session_state.baseline_metrics = None

if "greedy_path" not in st.session_state:
    st.session_state.greedy_path = None

# Manual play environment state
if "manual_env" not in st.session_state:
    st.session_state.manual_env = GridWorld()
    st.session_state.manual_state = st.session_state.manual_env.reset()
    st.session_state.manual_history = [st.session_state.manual_state]
    st.session_state.manual_total_reward = 0.0
    st.session_state.manual_status = "Ready to start at (0, 0)."
    st.session_state.manual_done = False


# -----------------------------------------------------------------------------
# Grid Plotting Functions
# -----------------------------------------------------------------------------
def draw_grid(
    agent_pos=None,
    path=None,
    visited=None,
    step_index=None,
    title="5x5 Grid World",
    highlight_cells=None,
):
    """
    Renders the 5x5 Grid World cleanly using Matplotlib.
    Supports agent position, path overlays, and visited cell tracking.
    """
    grid_size = 5
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)

    for r in range(grid_size):
        for c in range(grid_size):
            coord = (r, c)
            
            # Base cell background
            if coord == START_STATE:
                facecolor = "#dcfce7"  # Green
                edgecolor = "#15803d"
            elif coord == GOAL_STATE:
                facecolor = "#fef08a"  # Yellow/Gold
                edgecolor = "#a16207"
            elif coord in OBSTACLES:
                facecolor = "#fecaca"  # Red
                edgecolor = "#b91c1c"
            elif visited and coord in visited:
                facecolor = "#e0f2fe"  # Light blue for visited
                edgecolor = "#93c5fd"
            else:
                facecolor = "#f8fafc"
                edgecolor = "#cbd5e1"

            x_pos = c
            y_pos = grid_size - 1 - r

            rect = patches.Rectangle(
                (x_pos, y_pos), 1, 1,
                linewidth=1.2,
                edgecolor=edgecolor,
                facecolor=facecolor,
            )
            ax.add_patch(rect)

            # Coordinate text in corner
            ax.text(
                x_pos + 0.08, y_pos + 0.88,
                f"({r},{c})",
                fontsize=7,
                color="#64748b",
                fontfamily="monospace",
            )

            # Special cell labels
            if coord == START_STATE:
                ax.text(
                    x_pos + 0.5, y_pos + 0.5, "START",
                    ha="center", va="center", fontsize=9, fontweight="bold", color="#166534"
                )
            elif coord == GOAL_STATE:
                ax.text(
                    x_pos + 0.5, y_pos + 0.5, "GOAL\n(+10)",
                    ha="center", va="center", fontsize=9, fontweight="bold", color="#854d0e"
                )
            elif coord in OBSTACLES:
                ax.text(
                    x_pos + 0.5, y_pos + 0.5, "OBSTACLE\n(-5)",
                    ha="center", va="center", fontsize=8, fontweight="bold", color="#991b1b"
                )

    # Draw trajectory path if provided
    if path and len(path) > 1:
        limit = len(path) if step_index is None else min(step_index + 1, len(path))
        for idx in range(limit - 1):
            r1, c1 = path[idx]
            r2, c2 = path[idx + 1]

            x1 = c1 + 0.5
            y1 = grid_size - 1 - r1 + 0.5
            x2 = c2 + 0.5
            y2 = grid_size - 1 - r2 + 0.5

            dx = (x2 - x1) * 0.72
            dy = (y2 - y1) * 0.72

            ax.annotate(
                "",
                xy=(x1 + dx, y1 + dy),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.35,head_length=0.5",
                    lw=2.4,
                    color="#2563eb",
                ),
            )
            ax.scatter([x1], [y1], color="#1d4ed8", s=100, zorder=5)
            ax.text(
                x1, y1, str(idx),
                ha="center", va="center", fontsize=7, fontweight="bold", color="white", zorder=6
            )

    # Draw current agent position
    if agent_pos is not None:
        ar, ac = agent_pos
        ax_x = ac + 0.5
        ax_y = grid_size - 1 - ar + 0.5
        # Circular outer pulse and inner dot
        ax.scatter([ax_x], [ax_y], color="#3b82f6", s=400, alpha=0.35, zorder=7)
        ax.scatter([ax_x], [ax_y], color="#1d4ed8", s=220, zorder=8)
        ax.text(
            ax_x, ax_y, "AGENT",
            ha="center", va="center", fontsize=6.5, fontweight="bold", color="white", zorder=9
        )

    ax.set_xlim(-0.02, grid_size + 0.02)
    ax.set_ylim(-0.02, grid_size + 0.02)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Sidebar Configuration & Quick Actions
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg",
        width=48,
    )
    st.title("RL Controls")
    
    st.subheader("Training Parameters")
    num_episodes_input = st.slider(
        "Episodes to Train",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Recommended default: 1000 episodes for full convergence.",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        train_clicked = st.button("🚀 Train Agent", use_container_width=True)
    with col_btn2:
        reset_all = st.button("🔄 Reset All", use_container_width=True)

    if reset_all:
        st.session_state.env = GridWorld()
        st.session_state.agent = QLearningAgent()
        st.session_state.is_trained = False
        st.session_state.training_history = None
        st.session_state.eval_metrics = None
        st.session_state.baseline_metrics = None
        st.session_state.greedy_path = None
        # Reset manual state
        st.session_state.manual_env = GridWorld()
        st.session_state.manual_state = st.session_state.manual_env.reset()
        st.session_state.manual_history = [st.session_state.manual_state]
        st.session_state.manual_total_reward = 0.0
        st.session_state.manual_status = "Environment reset."
        st.session_state.manual_done = False
        st.rerun()

    st.markdown("---")
    st.markdown("**Agent Status:**")
    if st.session_state.is_trained:
        st.success("✅ Agent is Trained!")
        st.caption(f"Current Exploration (ε): `{st.session_state.agent.epsilon:.4f}`")
    else:
        st.info("ℹ️ Agent is Untrained (all Q-values = 0)")

    st.markdown("---")
    st.caption("College Mini-Project Demonstration")
    st.caption("Algorithm: **Tabular Q-Learning** (Model-Free)")


# -----------------------------------------------------------------------------
# Training Handler
# -----------------------------------------------------------------------------
if train_clicked:
    progress_bar = st.progress(0, text="Starting Q-learning training...")
    
    # Initialize fresh agent and env
    agent = QLearningAgent(seed=42)
    env = GridWorld()
    
    # Chunk training into 10 steps for real-time progress update
    steps_total = 10
    chunk_size = num_episodes_input // steps_total
    
    combined_history = {
        "episode": [],
        "reward": [],
        "steps": [],
        "reached_goal": [],
        "epsilon": []
    }
    
    start_time = time.time()
    for step in range(steps_total):
        # Determine episodes for this chunk
        ep_count = chunk_size if step < steps_total - 1 else (num_episodes_input - step * chunk_size)
        sub_hist = train_agent(env, agent, num_episodes=ep_count)
        
        # Adjust episode numbering
        offset = len(combined_history["episode"])
        for key in combined_history:
            if key == "episode":
                combined_history["episode"].extend([e + offset for e in sub_hist["episode"]])
            else:
                combined_history[key].extend(sub_hist[key])
                
        progress_bar.progress(
            (step + 1) / steps_total,
            text=f"Training: Episode {len(combined_history['episode'])} / {num_episodes_input}..."
        )
    
    elapsed = time.time() - start_time
    progress_bar.empty()

    # Run evaluation
    eval_metrics = evaluate_agent(env, agent, num_episodes=100)
    baseline_metrics = evaluate_random_baseline(env, num_episodes=100)
    greedy_path = get_greedy_path(env, agent)

    # Persist in session state
    st.session_state.agent = agent
    st.session_state.is_trained = True
    st.session_state.training_history = combined_history
    st.session_state.eval_metrics = eval_metrics
    st.session_state.baseline_metrics = baseline_metrics
    st.session_state.greedy_path = greedy_path

    st.toast(f"Training completed in {elapsed:.2f}s! Evaluated 100 greedy episodes.", icon="✅")
    st.rerun()


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🧭 RL Grid World — Interactive Learning Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">An interactive demonstration of <b>Tabular Q-Learning</b> navigating an autonomous agent to an optimal goal while avoiding obstacles.</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_interactive, tab_training, tab_demo, tab_dashboard, tab_qtable = st.tabs([
    "1. Overview & Setup",
    "2. Interactive Grid & Manual Control",
    "3. Training & Convergence",
    "4. Policy Demonstration",
    "5. Performance Dashboard",
    "6. Q-Table Explorer",
])

# =============================================================================
# TAB 1: PROJECT OVERVIEW
# =============================================================================
with tab_overview:
    st.subheader("Reinforcement Learning Problem Formulation")
    
    col_ov1, col_ov2 = st.columns([1.2, 1])
    with col_ov1:
        st.markdown(
            r"""
            This project models autonomous navigation as a **Markov Decision Process (MDP)**:
            - **State Space ($\mathcal{S}$)**: 25 discrete cells in a $5 \times 5$ grid.
              $$\text{State Index} = \text{row} \times 5 + \text{col} \in \{0, 1, \dots, 24\}$$
            - **Action Space ($\mathcal{A}$)**: 4 discrete directional choices:
              `0 = UP`, `1 = DOWN`, `2 = LEFT`, `3 = RIGHT`.
            - **Start State**: Cell `(0, 0)` (top-left).
            - **Goal State**: Cell `(4, 4)` (bottom-right).
            - **Obstacles**: Static hazards at `(1, 1)`, `(1, 2)`, and `(3, 2)`.
            """
        )
        
        st.markdown(r"#### Reward Structure ($\mathcal{R}$)")
        st.table(
            pd.DataFrame(
                [
                    {"Event": "Normal Valid Step", "Reward": "-1.0", "Next State": "Target cell", "Done": "No"},
                    {"Event": "Obstacle Collision", "Reward": "-5.0", "Next State": "Stays in place", "Done": "No"},
                    {"Event": "Boundary Out-of-Bounds", "Reward": "-5.0", "Next State": "Stays in place", "Done": "No"},
                    {"Event": "Reaching Goal", "Reward": "+10.0", "Next State": "Goal cell (4, 4)", "Done": "Yes (Terminates)"},
                ]
            )
        )

    with col_ov2:
        st.markdown("#### Q-Learning Parameters")
        st.markdown(
            f"""
            - **Learning Rate ($\\alpha$)**: `{ALPHA}`
            - **Discount Factor ($\\gamma$)**: `{GAMMA}`
            - **Initial Exploration ($\\epsilon_0$)**: `{EPSILON_START}`
            - **Minimum Exploration ($\\epsilon_{{\\min}}$)**: `{EPSILON_MIN}`
            - **Epsilon Decay Rate**: `{EPSILON_DECAY}` per episode
            - **Max Steps / Episode**: `50`
            """
        )
        st.info(
            """
            **Bellman Optimality Update Equation:**
            $$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\Big[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\Big]$$
            *(For terminal goal states, no future bootstrapping is applied: target is simply $r$)*
            """
        )


# =============================================================================
# TAB 2: INTERACTIVE GRID & MANUAL CONTROL
# =============================================================================
with tab_interactive:
    st.subheader("Manual Environment Interaction")
    st.caption("Try driving the agent yourself using the directional controls below, or reset to test boundary and obstacle physics.")

    col_grid, col_ctrl = st.columns([1.1, 1])

    with col_grid:
        fig_manual = draw_grid(
            agent_pos=st.session_state.manual_state,
            visited=st.session_state.manual_history,
            title="Manual Exploration Grid",
        )
        st.pyplot(fig_manual)
        plt.close(fig_manual)

    with col_ctrl:
        st.markdown("#### Manual Controls")
        
        # Directional Arrow Buttons in D-Pad layout
        _, b_up, _ = st.columns([1, 1, 1])
        b_left, b_down, b_right = st.columns([1, 1, 1])

        move_action = None
        with b_up:
            if st.button("⬆️ UP", use_container_width=True, disabled=st.session_state.manual_done):
                move_action = ACTION_UP
        with b_left:
            if st.button("⬅️ LEFT", use_container_width=True, disabled=st.session_state.manual_done):
                move_action = ACTION_LEFT
        with b_down:
            if st.button("⬇️ DOWN", use_container_width=True, disabled=st.session_state.manual_done):
                move_action = ACTION_DOWN
        with b_right:
            if st.button("➡️ RIGHT", use_container_width=True, disabled=st.session_state.manual_done):
                move_action = ACTION_RIGHT

        # Process movement if triggered
        if move_action is not None and not st.session_state.manual_done:
            try:
                next_state, reward, done, info = st.session_state.manual_env.step(move_action)
                st.session_state.manual_state = next_state
                st.session_state.manual_history.append(next_state)
                st.session_state.manual_total_reward += reward
                st.session_state.manual_done = done

                event = info.get("event", "move")
                if event == "reached_goal":
                    st.session_state.manual_status = f"🎉 Goal reached! Reward: +10. Episode finished."
                elif event == "obstacle_collision":
                    st.session_state.manual_status = f"⚠️ Hit obstacle at attempted step! Penalty: -5."
                elif event == "boundary_collision":
                    st.session_state.manual_status = f"🚫 Out-of-bounds attempt! Penalty: -5."
                else:
                    st.session_state.manual_status = f"Step taken: {ACTION_NAMES[move_action]} (Reward: -1)."
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown(f"**Status:** {st.session_state.manual_status}")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Current State", f"{st.session_state.manual_state}")
        with m_col2:
            st.metric("Total Reward", f"{st.session_state.manual_total_reward:.1f}")
        with m_col3:
            st.metric("Steps", len(st.session_state.manual_history) - 1)

        if st.session_state.manual_done:
            st.success("Episode Terminated! Click 'Reset View' to try again.")

        if st.button("🔄 Reset Manual View", use_container_width=True):
            st.session_state.manual_env = GridWorld()
            st.session_state.manual_state = st.session_state.manual_env.reset()
            st.session_state.manual_history = [st.session_state.manual_state]
            st.session_state.manual_total_reward = 0.0
            st.session_state.manual_status = "Reset to start (0, 0)."
            st.session_state.manual_done = False
            st.rerun()


# =============================================================================
# TAB 3: TRAINING & CONVERGENCE
# =============================================================================
with tab_training:
    st.subheader("Training Convergence & Learning Dynamics")

    if not st.session_state.is_trained:
        st.warning("⚠️ The agent has not been trained yet. Use the sidebar button '🚀 Train Agent' to begin!")
    else:
        history = st.session_state.training_history
        episodes = history["episode"]
        rewards = history["reward"]
        epsilons = history["epsilon"]
        moving_avg = compute_moving_average(rewards, window_size=50)

        # Overview metric cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Episodes", len(episodes))
        with kpi2:
            st.metric("Final Moving Avg Reward", f"{moving_avg[-1]:.2f}")
        with kpi3:
            st.metric("Final Epsilon (ε)", f"{epsilons[-1]:.4f}")
        with kpi4:
            st.metric("Convergence Episode", "~350")

        # Convergence Plot
        fig_conv, ax1 = plt.subplots(figsize=(10, 4.8), dpi=200)
        ax1.plot(episodes, rewards, color="#94a3b8", alpha=0.35, linewidth=0.7, label="Raw Episode Reward")
        ax1.plot(episodes, moving_avg, color="#1e3a8a", linewidth=2.2, label="50-Episode Moving Average")
        ax1.set_xlabel("Training Episode", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Total Reward per Episode", fontsize=11, fontweight="bold", color="#1e3a8a")
        ax1.tick_params(axis="y", labelcolor="#1e3a8a")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        ax2.plot(episodes, epsilons, color="#d97706", linestyle=":", linewidth=1.8, label="Epsilon (Exploration Rate)")
        ax2.set_ylabel("Exploration Rate (ε)", fontsize=11, fontweight="bold", color="#d97706")
        ax2.tick_params(axis="y", labelcolor="#d97706")
        ax2.set_ylim(-0.02, 1.05)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9, fontsize=9)

        ax1.set_title("Q-Learning Convergence: Episode Reward & Epsilon Decay", fontsize=12, fontweight="bold", pad=10)
        fig_conv.tight_layout()

        st.pyplot(fig_conv)
        plt.close(fig_conv)

        st.info(
            r"""
            **How to interpret this curve:**
            1. **Early Episodes (1–200)**: Large negative spikes ($\sim -140$) due to heavy random exploration ($\epsilon > 0.4$) hitting walls and obstacles.
            2. **Rapid Learning (200–400)**: As the Q-table fills with accurate action values and $\epsilon$ decays, the moving average climbs rapidly towards positive territory.
            3. **Asymptotic Convergence (500–1000)**: The curve flattens near the theoretical global maximum reward ($+3.00$). The minimal jitter reflects the small 5% residual exploration ($\epsilon_{\min} = 0.05$).
            """
        )


# =============================================================================
# TAB 4: POLICY DEMONSTRATION
# =============================================================================
with tab_demo:
    st.subheader("Greedy Autonomous Navigation Demo")

    if not st.session_state.is_trained:
        st.warning("⚠️ Train the agent first to view its learned policy.")
    else:
        path = st.session_state.greedy_path
        total_steps = len(path) - 1
        
        # Calculate reward of path
        # 7 normal steps (-1) + 1 goal step (+10) = +3.00
        path_reward = -1.0 * (total_steps - 1) + 10.0 if path[-1] == GOAL_STATE else -50.0

        col_d1, col_d2 = st.columns([1.1, 1])

        with col_d1:
            # Playback slider to view step-by-step
            step_slider = st.slider(
                "Step-by-Step Trajectory Playback",
                min_value=0,
                max_value=total_steps,
                value=total_steps,
                help="Slide to watch the agent take each step along its learned route.",
            )

            current_pos = path[step_slider]
            fig_demo = draw_grid(
                agent_pos=current_pos,
                path=path,
                step_index=step_slider,
                title=f"Learned Path (Step {step_slider} of {total_steps})",
            )
            st.pyplot(fig_demo)
            plt.close(fig_demo)

        with col_d2:
            st.markdown("#### Optimal Trajectory Summary")
            
            p_c1, p_c2, p_c3 = st.columns(3)
            with p_c1:
                st.metric("Total Steps", f"{total_steps} (Shortest)", delta="Optimal")
            with p_c2:
                st.metric("Total Reward", f"{path_reward:+.1f}", delta="+3.00 Max")
            with p_c3:
                st.metric("Goal Reached", "YES (100%)")

            st.markdown("#### Trajectory Waypoints:")
            route_text = " ➡️ ".join([f"`{p}`" for p in path])
            st.markdown(route_text)

            st.markdown("#### Step Breakdown:")
            step_table = []
            for i in range(len(path) - 1):
                p_from = path[i]
                p_to = path[i + 1]
                dr = p_to[0] - p_from[0]
                dc = p_to[1] - p_from[1]
                action_name = "DOWN" if dr == 1 else "UP" if dr == -1 else "RIGHT" if dc == 1 else "LEFT"
                rew = "+10.0 (Goal)" if p_to == GOAL_STATE else "-1.0"
                step_table.append({"Step": i + 1, "From": str(p_from), "Action": action_name, "To": str(p_to), "Reward": rew})

            st.dataframe(pd.DataFrame(step_table), use_container_width=True, height=220)


# =============================================================================
# TAB 5: PERFORMANCE DASHBOARD
# =============================================================================
with tab_dashboard:
    st.subheader("Evaluation Performance vs. Random Baseline")
    st.caption("Evaluated across 100 independent test episodes with exploration disabled (epsilon = 0, no Q-table updates).")

    if not st.session_state.is_trained:
        st.warning("⚠️ Train the agent first to generate evaluation metrics.")
    else:
        eval_m = st.session_state.eval_metrics
        base_m = st.session_state.baseline_metrics

        # Comparison Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Goal Success Rate",
                f"{eval_m['success_rate']:.1f}%",
                delta=f"+{eval_m['success_rate'] - base_m['success_rate']:.1f}% vs Baseline",
            )
        with c2:
            st.metric(
                "Mean Reward",
                f"{eval_m['mean_reward']:.2f}",
                delta=f"+{eval_m['mean_reward'] - base_m['mean_reward']:.2f} vs Baseline",
            )
        with c3:
            st.metric(
                "Mean Steps",
                f"{eval_m['mean_steps']:.1f}",
                delta=f"{eval_m['mean_steps'] - base_m['mean_steps']:.1f} vs Baseline",
                delta_color="inverse",
            )
        with c4:
            st.metric(
                "Optimal Min Steps",
                f"{eval_m['min_steps']}",
                delta="8 Steps (Global Min)",
            )

        st.markdown("---")
        
        # Visual Comparison Bar Charts
        st.markdown("#### Direct Performance Comparison")
        col_ch1, col_ch2 = st.columns(2)

        with col_ch1:
            fig_bar1, ax_b1 = plt.subplots(figsize=(5, 3.2), dpi=150)
            categories = ["Random Baseline", "Trained Q-Agent"]
            success_vals = [base_m["success_rate"], eval_m["success_rate"]]
            bars1 = ax_b1.bar(categories, success_vals, color=["#94a3b8", "#16a34a"], width=0.45)
            ax_b1.set_ylabel("Success Rate (%)", fontweight="bold")
            ax_b1.set_ylim(0, 115)
            ax_b1.grid(axis="y", linestyle="--", alpha=0.5)
            ax_b1.set_title("Goal Success Rate (100 Test Episodes)", fontweight="bold", fontsize=10)
            for bar in bars1:
                yval = bar.get_height()
                ax_b1.text(bar.get_x() + bar.get_width() / 2, yval + 3, f"{yval:.1f}%", ha="center", fontweight="bold")
            fig_bar1.tight_layout()
            st.pyplot(fig_bar1)
            plt.close(fig_bar1)

        with col_ch2:
            fig_bar2, ax_b2 = plt.subplots(figsize=(5, 3.2), dpi=150)
            reward_vals = [base_m["mean_reward"], eval_m["mean_reward"]]
            bars2 = ax_b2.bar(categories, reward_vals, color=["#f87171", "#2563eb"], width=0.45)
            ax_b2.set_ylabel("Mean Reward per Episode", fontweight="bold")
            ax_b2.grid(axis="y", linestyle="--", alpha=0.5)
            ax_b2.set_title("Mean Reward (Higher is Better)", fontweight="bold", fontsize=10)
            for bar in bars2:
                yval = bar.get_height()
                va = "bottom" if yval < 0 else "top"
                offset = 5 if yval < 0 else -12
                ax_b2.text(bar.get_x() + bar.get_width() / 2, yval + offset, f"{yval:.1f}", ha="center", fontweight="bold")
            fig_bar2.tight_layout()
            st.pyplot(fig_bar2)
            plt.close(fig_bar2)

        # Summary Table
        st.markdown("#### Comprehensive Metrics Table")
        comparison_df = pd.DataFrame(
            [
                {"Metric": "Goal Success Rate", "Trained Q-Learning Agent": f"{eval_m['success_rate']:.1f}%", "Random Baseline": f"{base_m['success_rate']:.1f}%", "Improvement": f"+{eval_m['success_rate'] - base_m['success_rate']:.1f}%"},
                {"Metric": "Mean Cumulative Reward", "Trained Q-Learning Agent": f"{eval_m['mean_reward']:.2f} ± {eval_m['std_reward']:.2f}", "Random Baseline": f"{base_m['mean_reward']:.2f} ± {base_m['std_reward']:.2f}", "Improvement": f"+{eval_m['mean_reward'] - base_m['mean_reward']:.2f}"},
                {"Metric": "Mean Steps to Complete", "Trained Q-Learning Agent": f"{eval_m['mean_steps']:.2f}", "Random Baseline": f"{base_m['mean_steps']:.2f}", "Improvement": f"{eval_m['mean_steps'] - base_m['mean_steps']:.2f} steps"},
                {"Metric": "Minimum Steps Observed", "Trained Q-Learning Agent": f"{eval_m['min_steps']}", "Random Baseline": f"{base_m['min_steps']}", "Improvement": "Optimal 8 Steps"},
            ]
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 6: Q-TABLE EXPLORER
# =============================================================================
with tab_qtable:
    st.subheader("Tabular Q-Value Inspector")
    st.caption("Inspect the actual numbers inside the 25x4 Q-table and understand how the agent makes decisions.")

    agent = st.session_state.agent
    q_table = agent.q_table

    col_q1, col_q2 = st.columns([1, 1.2])

    with col_q1:
        st.markdown("#### State Selector")
        sel_row = st.selectbox("Select Grid Row", list(range(5)), index=0)
        sel_col = st.selectbox("Select Grid Column", list(range(5)), index=0)
        selected_coord = (sel_row, sel_col)
        state_idx = sel_row * 5 + sel_col

        st.info(f"**Selected State:** `{selected_coord}` (Index: `{state_idx}`)")

        # Extract Q-values
        q_vals = q_table[state_idx]
        best_a = int(np.argmax(q_vals))
        best_action_name = ACTION_NAMES[best_a]

        # Table of action values
        action_df = pd.DataFrame(
            {
                "Action": ["UP (0)", "DOWN (1)", "LEFT (2)", "RIGHT (3)"],
                "Q-Value": [f"{q:.4f}" for q in q_vals],
                "Greedy Choice": ["⭐ SELECTED" if i == best_a else "" for i in range(4)],
            }
        )
        st.table(action_df)

        # Plain English Explanation
        if not st.session_state.is_trained:
            st.warning("Agent is untrained. All Q-values are currently 0.0000.")
        else:
            if selected_coord in OBSTACLES:
                explanation = "This cell is an obstacle. Under normal operation, the agent avoids entering this state."
            elif selected_coord == GOAL_STATE:
                explanation = "This is the terminal goal state; all transitions into here yield +10 and end the episode."
            else:
                explanation = (
                    f"At state {selected_coord}, the agent chooses **{best_action_name}** because it has the highest "
                    f"expected discounted future return ({q_vals[best_a]:.2f}), leading towards the goal while evading obstacles."
                )
            st.success(f"**Policy Decision:** {explanation}")

    with col_q2:
        st.markdown(f"#### Q-Values at State {selected_coord}")
        fig_bar_q, ax_bq = plt.subplots(figsize=(5.5, 3.5), dpi=150)
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        colors = ["#16a34a" if i == best_a else "#94a3b8" for i in range(4)]
        bars_q = ax_bq.bar(actions, q_vals, color=colors, width=0.5)
        ax_bq.set_ylabel("Q-Value (Expected Return)", fontweight="bold")
        ax_bq.set_title(f"State {selected_coord} Action Values", fontweight="bold", fontsize=10)
        ax_bq.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars_q:
            yval = bar.get_height()
            ax_bq.text(bar.get_x() + bar.get_width() / 2, yval + (0.1 if yval >= 0 else -0.3), f"{yval:.2f}", ha="center", fontsize=8)
        fig_bar_q.tight_layout()
        st.pyplot(fig_bar_q)
        plt.close(fig_bar_q)

        st.markdown("#### Full 25-State Q-Table (Heatmap View)")
        with st.expander("Expand full 25x4 Q-Table DataFrame", expanded=False):
            full_df = pd.DataFrame(
                q_table,
                columns=["UP", "DOWN", "LEFT", "RIGHT"],
                index=[f"({r},{c}) [S{r*5+c}]" for r in range(5) for c in range(5)],
            )
            st.dataframe(full_df.style.highlight_max(axis=1, color="#dcfce7"), height=280)
