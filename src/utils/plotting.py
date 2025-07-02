import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from agents.dqn import act as dqn_act

# Cache for terrain to ensure consistency across frames
_terrain_cache = {}

def plot_gridworld(env, agent=None, fig=None, ax=None, terrain_seed=42, show_q_heatmap=True, trajectory=None, episode=None, step=None, cbar_ax=None):
    if not hasattr(env, 'config'):
        raise ValueError("Environment must have config attribute")
    required_keys = {'grid_size', 'goal_pos', 'wind', 'stochastic_terrain'}
    if not required_keys.issubset(env.config.keys()):
        raise ValueError(f"Environment config missing required keys: {required_keys - env.config.keys()}")
    global _terrain_cache
    grid_height, grid_width = env.config["grid_size"]
    cache_key = (terrain_seed, grid_height, grid_width, str(env.config.get("stochastic_terrain", {})))
    # Precompute terrain map for consistency
    if cache_key not in _terrain_cache:
        rng = np.random.RandomState(terrain_seed)
        ice_map = rng.rand(grid_height, grid_width) < env.config["stochastic_terrain"].get("ice", 0)
        swamp_map = rng.rand(grid_height, grid_width) < env.config["stochastic_terrain"].get("swamp", 0)
        _terrain_cache[cache_key] = (ice_map, swamp_map)
    else:
        ice_map, swamp_map = _terrain_cache[cache_key]
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        close_fig = True
    else:
        ax.clear()
        close_fig = False
    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(-0.5, grid_height - 0.5)
    ax.set_xticks(range(grid_width))
    ax.set_yticks(range(grid_height))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    # Draw goal first (background)
    goal_row, goal_col = env.config["goal_pos"]
    ax.add_patch(patches.Rectangle((goal_col-0.3, goal_row-0.3), 0.6, 0.6, color='green', label='Goal', zorder=1))
    # Draw terrain (ice and swamp) using cached maps
    for i in range(grid_height):
        for j in range(grid_width):
            if ice_map[i, j]:
                ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, color='#b3e0ff', alpha=0.5, zorder=2))
            if swamp_map[i, j]:
                ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, color='#145a32', alpha=0.3, zorder=2))
    # Draw wind strength indicators
    for col, strength in enumerate(env.config["wind"]):
        if strength > 0:
            ax.annotate(f'↑{strength}', (col, -0.7), ha='center', fontsize=12, color='navy', zorder=10)
    # Draw Q-value heatmap if agent is provided
    if agent is not None and hasattr(agent, 'Q') and show_q_heatmap:
        q_max = np.max(agent.Q, axis=2)
        im = ax.imshow(q_max, cmap='YlOrRd', alpha=0.3, origin='upper', extent=[-0.5, grid_width-0.5, grid_height-0.5, -0.5], zorder=3)
        if cbar_ax is not None:
            cbar_ax.clear()
            plt.colorbar(im, cax=cbar_ax)
            cbar_ax.set_ylabel('Max Q-value')
    # Draw agent trajectory if provided
    if trajectory is not None and len(trajectory) > 1:
        traj = np.array(trajectory)
        ax.plot(traj[:,1], traj[:,0], color='magenta', linewidth=2, alpha=0.7, label='Trajectory', zorder=11)
    # Draw policy arrows if agent is not None and has Q
    if agent is not None and hasattr(agent, 'Q'):
        for i in range(grid_height):
            for j in range(grid_width):
                if (i, j) != tuple(env.config["goal_pos"]):
                    action = np.argmax(agent.Q[i, j])
                    dx, dy = 0, 0
                    if action == 0: dy = 0.3  # up
                    elif action == 1: dy = -0.3  # down
                    elif action == 2: dx = -0.3  # left
                    elif action == 3: dx = 0.3  # right
                    ax.arrow(j, i, dx, dy, head_width=0.18, head_length=0.18, fc='blue', ec='blue', alpha=0.6, zorder=8, length_includes_head=True)
    # Draw agent (on top)
    agent_row, agent_col = env.agent_pos
    ax.add_patch(patches.Circle((agent_col, agent_row), 0.3, color='orange', ec='black', lw=1.5, label='Agent', zorder=12))
    # Remove duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    # Add episode/step info
    if episode is not None and step is not None:
        ax.set_title(f"GridWorld | Episode {episode+1} | Step {step+1}")
    else:
        ax.set_title("GridWorld Environment")
    plt.tight_layout()
    if close_fig:
        plt.pause(0.1)
        plt.close(fig)
    else:
        fig.canvas.draw_idle()
        plt.pause(0.01)

def update_metric_plots(metrics, r_ax, s_ax, succ_ax):
    """Update the metrics plot axes with current training metrics."""
    r_ax.clear()
    s_ax.clear()
    succ_ax.clear()
    
    # Plot rewards
    r_ax.plot(metrics["rewards"], 'b-')
    r_ax.set_title("Rewards per Episode")
    r_ax.set_xlabel("Episode")
    r_ax.set_ylabel("Total Reward")
    r_ax.grid(True)
    
    # Plot steps
    s_ax.plot(metrics["steps"], 'r-')
    s_ax.set_title("Steps per Episode")
    s_ax.set_xlabel("Episode")
    s_ax.set_ylabel("Steps")
    s_ax.grid(True)
    
    # Plot success rate
    if metrics["successes"]:
        window = min(10, len(metrics["successes"]))
        rate = np.convolve(metrics["successes"], np.ones(window)/window, mode='valid')
        succ_ax.plot(rate, 'g-')
        succ_ax.set_ylim(-0.1, 1.1)
        succ_ax.set_title("Success Rate")
        succ_ax.set_xlabel("Episode")
        succ_ax.set_ylabel("Success Rate (10-ep avg)")
        succ_ax.grid(True)

def plot_final_metrics(metrics):
    """Create a final summary plot of all metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(131)
    plt.plot(metrics["rewards"], 'b-')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    
    # Plot steps
    plt.subplot(132)
    plt.plot(metrics["steps"], 'g-')
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    
    # Plot success rate
    plt.subplot(133)
    window = min(10, len(metrics["successes"]))
    rate = np.convolve(metrics["successes"], np.ones(window)/window, mode='valid')
    plt.plot(rate, 'r-')
    plt.title("Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (10-ep avg)")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    
    plt.tight_layout()

def animate_gridworld_episode(env, agent, online_model, metrics, num_episodes):
    """Animate the gridworld environment and update metrics in real-time."""
    plt.ion()  # Enable interactive mode
    
    # Create main environment figure with colorbar
    env_fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(1, 2, width_ratios=[20, 1], figure=env_fig)
    env_ax = env_fig.add_subplot(gs[0])
    cbar_ax = env_fig.add_subplot(gs[1])
    
    # Create metrics figure
    metrics_fig, (r_ax, s_ax, succ_ax) = plt.subplots(1, 3, figsize=(15, 4))
    
    try:
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            step_count = 0
            trajectory = [env.agent_pos]

            while not done:
                # Get action based on agent type
                if agent:
                    action = agent.act(obs)
                else:  # DQN
                    flat_obs = obs if not isinstance(obs, tuple) else obs[0]
                    action = dqn_act(flat_obs, online_model, 0.01)
                
                # Take step in environment
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                step_count += 1
                trajectory.append(env.agent_pos)

                # Update visualizations
                env_ax.clear()
                plot_gridworld(
                    env, agent, env_fig, env_ax,
                    trajectory=trajectory,
                    episode=ep,
                    step=step_count,
                    cbar_ax=cbar_ax
                )
                update_metric_plots(metrics, r_ax, s_ax, succ_ax)
                plt.pause(0.05)  # Control animation speed
                
            plt.pause(0.5)  # Pause between episodes
    finally:
        plt.ioff()  # Disable interactive mode

import matplotlib.pyplot as plt
import numpy as np

class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, (self.r_ax, self.s_ax, self.succ_ax) = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.suptitle("Training Metrics (Live)")
        self.r_ax.set_title("Rewards")
        self.s_ax.set_title("Steps")
        self.succ_ax.set_title("Success Rate")
        self.r_ax.set_xlabel("Episode")
        self.s_ax.set_xlabel("Episode")
        self.succ_ax.set_xlabel("Episode")
        self.r_ax.set_ylabel("Reward")
        self.s_ax.set_ylabel("Steps")
        self.succ_ax.set_ylabel("Success Rate")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update(self, metrics):
        self.r_ax.clear()
        self.s_ax.clear()
        self.succ_ax.clear()
        self.r_ax.plot(metrics["rewards"], 'b-')
        self.s_ax.plot(metrics["steps"], 'g-')
        if metrics["successes"]:
            window = min(10, len(metrics["successes"]))
            rate = np.convolve(metrics["successes"], np.ones(window)/window, mode='valid')
            self.succ_ax.plot(rate, 'r-')
            self.succ_ax.set_ylim(-0.1, 1.1)
        self.r_ax.set_title("Rewards")
        self.s_ax.set_title("Steps")
        self.succ_ax.set_title("Success Rate")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
