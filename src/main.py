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

def plot_gridworld(env):
    plt.clf()
    grid_height, grid_width = env.config["grid_size"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(-0.5, grid_height - 0.5)
    ax.set_xticks(range(grid_width))
    ax.set_yticks(range(grid_height))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    # Draw agent
    agent_row, agent_col = env.agent_pos
    ax.add_patch(patches.Rectangle((agent_col - 0.5, agent_row - 0.5), 1, 1, color='orange', label='Agent'))
    # Draw goal
    goal_row, goal_col = env.config["goal_pos"]
    ax.add_patch(patches.Rectangle((goal_col - 0.5, goal_row - 0.5), 1, 1, color='green', label='Goal'))
    ax.legend(handles=[patches.Patch(color='orange', label='Agent'), patches.Patch(color='green', label='Goal')], loc='upper right')
    plt.title("GridWorld Visualization")
    plt.pause(0.3)
    plt.close(fig)

plt.ion()

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    while not done:
        if episode % 10 == 0:
            plot_gridworld(env)
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        step_count += 1
        total_reward += reward

        
        

    metrics["rewards"].append(total_reward)
    metrics["steps"].append(step_count)


    
    success = 1 if tuple(obs) == tuple(env.config["goal_pos"]) else 0
    
    metrics["successes"].append(success)
    


avg_reward = np.mean(metrics["rewards"])
avg_steps = np.mean(metrics["steps"])
success_rate = np.mean(metrics["successes"])


  


print(f"Average reward across {num_episodes} episodes: {avg_reward:.2f}")
print(f"Average steps across {num_episodes} episodes: {avg_steps:.2f}")




plt.ioff()
plot_gridworld(env)
plt.show()
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(metrics["rewards"], marker='o')
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.subplot(1, 2, 2)
plt.plot(metrics["steps"], marker='x', color='orange')
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.figure(figsize=(5, 4))
plt.plot(metrics["successes"], marker='o', color='green')
plt.title("Success per Episode (1 = success, 0 = fail)")
plt.xlabel("Episode")
plt.ylabel("Success")
plt.tight_layout()
plt.show()