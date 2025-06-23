import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from envs.gridworld import GridWorldEnv
from agents.q_learning import QLearningAgent
from config import DEFAULT_ENV_CONFIG, QL_AGENT_CONFIG

# Initialize environment and agent
env = GridWorldEnv(config=DEFAULT_ENV_CONFIG)
agent = QLearningAgent(env, QL_AGENT_CONFIG)

num_episodes = 50
train_episodes = 1000

# Train the agent
agent.train(num_episodes=train_episodes)

metrics = {
    "rewards": [],
    "steps": [],
    "successes": []
}

def plot_gridworld(env, agent=None, fig=None, ax=None, terrain_seed=42):
    # Deterministic terrain for consistent visualization
    rng = np.random.RandomState(terrain_seed)
    grid_height, grid_width = env.config["grid_size"]
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
    # Draw terrain (ice and swamp)
    for i in range(grid_height):
        for j in range(grid_width):
            if rng.rand() < env.config["stochastic_terrain"]["ice"]:
                ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, color='lightblue', alpha=0.3))
            if rng.rand() < env.config["stochastic_terrain"]["swamp"]:
                ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, color='darkgreen', alpha=0.3))
    # Draw wind strength indicators
    for col, strength in enumerate(env.config["wind"]):
        if strength > 0:
            ax.text(col, -0.7, f'↑{strength}', ha='center', fontsize=10)
    # Draw agent
    agent_row, agent_col = env.agent_pos
    ax.add_patch(patches.Circle((agent_col, agent_row), 0.3, color='orange', label='Agent'))
    # Draw goal
    goal_row, goal_col = env.config["goal_pos"]
    ax.add_patch(patches.Rectangle((goal_col-0.3, goal_row-0.3), 0.6, 0.6, color='green', label='Goal'))
    # Draw policy arrows if agent is not None and has Q
    if agent is not None and hasattr(agent, 'Q'):
        for i in range(grid_height):
            for j in range(grid_width):
                if (i, j) != tuple(env.config["goal_pos"]):
                    action = np.argmax(agent.Q[i, j])
                    dx, dy = 0, 0
                    if action == 0: dy = 0.2  # up
                    elif action == 1: dy = -0.2  # down
                    elif action == 2: dx = -0.2  # left
                    elif action == 3: dx = 0.2  # right
                    ax.arrow(j, i, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
    # Add legend with terrain types
    legend_elements = [
        patches.Patch(color='orange', label='Agent'),
        patches.Patch(color='green', label='Goal'),
        patches.Patch(color='lightblue', alpha=0.3, label='Ice'),
        patches.Patch(color='darkgreen', alpha=0.3, label='Swamp')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title("GridWorld Environment")
    plt.tight_layout()
    if close_fig:
        plt.pause(0.1)
        plt.close(fig)
    else:
        fig.canvas.draw()
        plt.pause(0.01)

# Create persistent figures for visualization
env_fig, env_ax = plt.subplots(figsize=(8, 8))
metrics_fig, (reward_ax, steps_ax, success_ax) = plt.subplots(1, 3, figsize=(15, 4))

plt.ion()
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    while not done:
        if episode % 5 == 0:
            plot_gridworld(env, agent, env_fig, env_ax)
            reward_ax.clear()
            steps_ax.clear()
            success_ax.clear()
            reward_ax.plot(metrics["rewards"], 'b-')
            reward_ax.set_title("Rewards per Episode")
            reward_ax.set_xlabel("Episode")
            reward_ax.set_ylabel("Total Reward")
            steps_ax.plot(metrics["steps"], 'r-')
            steps_ax.set_title("Steps per Episode")
            steps_ax.set_xlabel("Episode")
            steps_ax.set_ylabel("Steps")
            if len(metrics["successes"]) > 0:
                window = min(10, len(metrics["successes"]))
                success_rate = np.convolve(metrics["successes"], np.ones(window)/window, mode='valid')
                success_ax.plot(success_rate, 'g-')
                success_ax.set_title("Success Rate (Moving Average)")
                success_ax.set_xlabel("Episode")
                success_ax.set_ylabel("Success Rate")
                success_ax.set_ylim(-0.1, 1.1)
            plt.pause(0.01)
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        step_count += 1
        total_reward += reward
    metrics["rewards"].append(total_reward)
    metrics["steps"].append(step_count)
    success = 1 if tuple(env.agent_pos) == tuple(env.config["goal_pos"]) else 0
    metrics["successes"].append(success)
    if episode % 10 == 0:
        print(f"Episode {episode}/{num_episodes}")
        print(f"Reward: {total_reward:.2f}, Steps: {step_count}, Success: {success}")

avg_reward = np.mean(metrics["rewards"])
avg_steps = np.mean(metrics["steps"])
success_rate = np.mean(metrics["successes"])

print("\nTraining Summary:")
print(f"Average reward across {num_episodes} episodes: {avg_reward:.2f}")
print(f"Average steps across {num_episodes} episodes: {avg_steps:.2f}")
print(f"Overall success rate: {success_rate:.2%}")

plt.ioff()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(metrics["rewards"], 'b-', label='Episode Reward')
plt.plot(np.convolve(metrics["rewards"], np.ones(10)/10, mode='valid'), 'r-', label='Moving Average')
plt.title("Rewards over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.subplot(132)
plt.plot(metrics["steps"], 'g-', label='Episode Steps')
plt.plot(np.convolve(metrics["steps"], np.ones(10)/10, mode='valid'), 'r-', label='Moving Average')
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.subplot(133)
window = min(10, len(metrics["successes"]))
success_rate = np.convolve(metrics["successes"], np.ones(window)/window, mode='valid')
plt.plot(success_rate, 'r-', label='Success Rate')
plt.title("Success Rate")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 8))
plot_gridworld(env, agent)
plt.show()