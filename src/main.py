import matplotlib.pyplot as plt
import matplotlib.patches as patches
from envs.gridworld import GridWorldEnv
from agents.value_iteration import ValueIterationAgent
from agents.policy_iteration import PolicyIterationAgent
from config import ENV_CONFIG, AGENT_CONFIG


env = GridWorldEnv(config=ENV_CONFIG)
agent = PolicyIterationAgent(env, AGENT_CONFIG)
agent.run_policy_iteration()

episode_rewards = []
episode_steps = []



obs = env.reset()


done = False


step_count = 0
total_reward = 0

num_episodes = 50

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
        plot_gridworld(env)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step_count += 1
        total_reward += reward

plt.ioff()
plot_gridworld(env)
plt.show()

episode_rewards.append(total_reward)
episode_steps.append(step_count)



print(f"Episode finished in {step_count} steps with total reward {total_reward}")


plt.figure(figsize=(10, 4))


plt.subplot(1, 2, 1)
plt.plot(episode_rewards, marker='o')
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")


plt.subplot(1, 2, 2)
plt.plot(episode_steps, marker='x', color='orange')
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")

plt.tight_layout()
plt.show()