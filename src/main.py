import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.gridspec import GridSpec
from envs.gridworld import GridWorldEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQN, act as dqn_act
from config import DEFAULT_ENV_CONFIG, QL_AGENT_CONFIG
from utils.plotting import plot_gridworld  # <- Move your plot_gridworld() to a separate file

# ========================== Config ==========================
NUM_EPISODES = 50
TRAIN_EPISODES = 1000
MAX_STEPS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== Metric Utilities =====================
def initialize_metrics():
    return {"rewards": [], "steps": [], "successes": []}

def record_metrics(metrics, reward, steps, goal_reached):
    metrics["rewards"].append(reward)
    metrics["steps"].append(steps)
    metrics["successes"].append(1 if goal_reached else 0)

# ======================== DQN Logic ==========================
def train_dqn(env, metrics):
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state
    state_size = np.array(state).size
    action_size = env.action_space.n if hasattr(env, 'action_space') else 4

    online_model = DQN(state_size, action_size).to(DEVICE)
    target_model = DQN(state_size, action_size).to(DEVICE)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(online_model.parameters(), lr=1e-3)
    replay_buffer = deque(maxlen=10000)

    epsilon, epsilon_min, decay = 1.0, 0.01, 0.995
    batch_size, sync_freq, gamma = 64, 10, 0.99

    for ep in range(NUM_EPISODES):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        total_reward, done = 0, False

        for t in range(MAX_STEPS):
            action = dqn_act(state, online_model, epsilon)
            next_state, reward, done, *_ = env.step(action)
            next_state = next_state[0] if isinstance(next_state, tuple) else next_state
            replay_buffer.append((state, action, reward, next_state, float(done)))

            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                update_dqn(minibatch, online_model, target_model, optimizer, gamma)

            if done:
                break

        if ep % sync_freq == 0:
            target_model.load_state_dict(online_model.state_dict())

        epsilon = max(epsilon_min, epsilon * decay)
        record_metrics(metrics, total_reward, t + 1, tuple(env.agent_pos) == tuple(env.config["goal_pos"]))
        print(f"[DQN] Episode {ep} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    return None, online_model  # agent=None for DQN

def update_dqn(batch, online_model, target_model, optimizer, gamma):
    states = torch.FloatTensor([s[0] for s in batch]).to(DEVICE)
    actions = torch.LongTensor([s[1] for s in batch]).to(DEVICE)
    rewards = torch.FloatTensor([s[2] for s in batch]).to(DEVICE)
    next_states = torch.FloatTensor([s[3] for s in batch]).to(DEVICE)
    dones = torch.FloatTensor([s[4] for s in batch]).to(DEVICE)

    with torch.no_grad():
        next_q = target_model(next_states).max(1)[0]
        target = rewards + (1 - dones) * gamma * next_q

    q_values = online_model(states)
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = torch.nn.MSELoss()(q_selected, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ======================= Q-Learning ==========================
def train_q_learning(env, metrics):
    agent = QLearningAgent(env, QL_AGENT_CONFIG)
    agent.train(num_episodes=TRAIN_EPISODES)

    for ep in range(NUM_EPISODES):
        obs = env.reset()
        total_reward, done, steps = 0, False, 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == tuple(env.config["goal_pos"]))
        if ep % 10 == 0:
            print(f"[QL] Episode {ep}/{NUM_EPISODES} | Reward: {total_reward:.2f}, Steps: {steps}")
    return agent, None  # no model used

# ====================== Visualization ========================
def visualize(env, agent, online_model, metrics):
    plt.ion()
    env_fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(1, 2, width_ratios=[20, 1], figure=env_fig)
    env_ax = env_fig.add_subplot(gs[0])
    cbar_ax = env_fig.add_subplot(gs[1])
    metrics_fig, (r_ax, s_ax, succ_ax) = plt.subplots(1, 3, figsize=(15, 4))

    for ep in range(NUM_EPISODES):
        obs = env.reset()
        done, total_reward, step_count = False, 0, 0
        trajectory = []

        while not done:
            action = (
                agent.act(obs) if agent
                else dqn_act(obs if not isinstance(obs, tuple) else obs[0], online_model, 0.01)
            )
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            step_count += 1
            trajectory.append(env.agent_pos)

            if ep % 5 == 0:
                plot_gridworld(env, agent, env_fig, env_ax, trajectory=trajectory, episode=ep, step=step_count, cbar_ax=cbar_ax)
                update_metric_plots(metrics, r_ax, s_ax, succ_ax)

        # Optional: Update metrics if needed

    plt.ioff()
    final_plot(metrics)
    plot_gridworld(env, agent)
    plt.show()

def update_metric_plots(metrics, r_ax, s_ax, succ_ax):
    r_ax.clear()
    s_ax.clear()
    succ_ax.clear()
    r_ax.plot(metrics["rewards"], 'b-')
    r_ax.set_title("Rewards per Episode")
    s_ax.plot(metrics["steps"], 'r-')
    s_ax.set_title("Steps per Episode")
    if metrics["successes"]:
        window = min(10, len(metrics["successes"]))
        rate = np.convolve(metrics["successes"], np.ones(window)/window, mode='valid')
        succ_ax.plot(rate, 'g-')
        succ_ax.set_ylim(-0.1, 1.1)
        succ_ax.set_title("Success Rate")

def final_plot(metrics):
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(metrics["rewards"], 'b-')
    plt.title("Episode Rewards")
    plt.subplot(132)
    plt.plot(metrics["steps"], 'g-')
    plt.title("Steps per Episode")
    plt.subplot(133)
    window = min(10, len(metrics["successes"]))
    rate = np.convolve(metrics["successes"], np.ones(window)/window, mode='valid')
    plt.plot(rate, 'r-')
    plt.title("Success Rate")
    plt.tight_layout()
    plt.show()

# ========================== Main =============================
def main():
    try:
        agent_type = input("Select agent (q_learning/dqn): ").strip().lower()
        env = GridWorldEnv(config=DEFAULT_ENV_CONFIG)
        metrics = initialize_metrics()

        if agent_type == "dqn":
            agent, model = train_dqn(env, metrics)
        else:
            agent, model = train_q_learning(env, metrics)

        visualize(env, agent, model, metrics)

        print("\nTraining Summary:")
        print(f"Avg Reward: {np.mean(metrics['rewards']):.2f}")
        print(f"Avg Steps: {np.mean(metrics['steps']):.2f}")
        print(f"Success Rate: {np.mean(metrics['successes']):.2%}")

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted.")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
    finally:
        plt.close('all')
        print("[INFO] Exiting program.")

if __name__ == "__main__":
    main()
