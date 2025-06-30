import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from envs.gridworld import GridWorldEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQN, act as dqn_act
from config import DEFAULT_ENV_CONFIG, QL_AGENT_CONFIG, DQN_AGENT_CONFIG
from utils.plotting import (
    plot_gridworld,
    plot_final_metrics,
    LivePlotter
)
import traceback

# ========================== Config ==========================
NUM_EPISODES = 50
TRAIN_EPISODES = 1000
MAX_STEPS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REWARD_STEP_PENALTY = -1.0  # Encourage shorter paths

# ====================== Metric Utilities =====================
def initialize_metrics():
    """Initialize empty metrics dictionary."""
    return {"rewards": [], "steps": [], "successes": []}

def record_metrics(metrics, reward, steps, goal_reached):
    """Record episode metrics."""
    metrics["rewards"].append(reward)
    metrics["steps"].append(steps)
    metrics["successes"].append(1 if goal_reached else 0)

# ======================== DQN Logic ==========================
def update_dqn(batch, online_model, target_model, optimizer, gamma):
    """Update DQN models using a batch of experiences (Double DQN)."""
    states = torch.FloatTensor([s[0] for s in batch]).to(DEVICE)
    actions = torch.LongTensor([s[1] for s in batch]).to(DEVICE)
    rewards = torch.FloatTensor([s[2] for s in batch]).to(DEVICE)
    next_states = torch.FloatTensor([s[3] for s in batch]).to(DEVICE)
    dones = torch.FloatTensor([s[4] for s in batch]).to(DEVICE)
    with torch.no_grad():
        next_actions = online_model(next_states).argmax(1)
        next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + (1 - dones) * gamma * next_q
    q_values = online_model(states)
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = torch.nn.MSELoss()(q_selected, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_dqn(env, metrics):
    num_episodes = DQN_AGENT_CONFIG.get("num_episodes", NUM_EPISODES)
    max_steps = DQN_AGENT_CONFIG.get("max_steps", MAX_STEPS)
    plotter = LivePlotter()
    state = env.reset()
    state = unwrap_state(state)
    state_size = np.array(state).size
    action_size = env.action_space.n if hasattr(env, 'action_space') else 4
    goal = tuple(env.config["goal_pos"])

    # Initialize models from dqn.py
    online_model = DQN(state_size, action_size, hidden_dim=DQN_AGENT_CONFIG["hidden_dim"]).to(DEVICE)
    target_model = DQN(state_size, action_size, hidden_dim=DQN_AGENT_CONFIG["hidden_dim"]).to(DEVICE)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(online_model.parameters(), lr=DQN_AGENT_CONFIG["learning_rate"])
    replay_buffer = deque(maxlen=DQN_AGENT_CONFIG["buffer_size"])

    epsilon = DQN_AGENT_CONFIG["epsilon_start"]
    epsilon_min = DQN_AGENT_CONFIG["epsilon_min"]
    decay = DQN_AGENT_CONFIG["epsilon_decay"]
    batch_size = DQN_AGENT_CONFIG["batch_size"]
    sync_freq = DQN_AGENT_CONFIG["sync_frequency"]
    gamma = DQN_AGENT_CONFIG["gamma"]
    reward_step_penalty = DQN_AGENT_CONFIG["reward_step_penalty"]

    for ep in range(num_episodes):
        state = env.reset()
        state = unwrap_state(state)
        total_reward, done = 0, False
        for t in range(max_steps):
            action = dqn_act(state, online_model, epsilon)
            next_state, reward, done, *_ = env.step(action)
            next_state = unwrap_state(next_state)
            shaped_reward = reward + reward_step_penalty
            replay_buffer.append((state, action, shaped_reward, next_state, float(done)))
            state = next_state
            total_reward += shaped_reward
            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                update_dqn(minibatch, online_model, target_model, optimizer, gamma)
            if done:
                break
        if ep % sync_freq == 0:
            target_model.load_state_dict(online_model.state_dict())
        epsilon = max(epsilon_min, epsilon * decay)
        record_metrics(metrics, total_reward, t + 1, tuple(env.agent_pos) == goal)
        plotter.update(metrics)
        print(f"[DQN] Episode {ep} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
    return None, online_model  # agent=None for DQN

# ======================= Q-Learning ==========================
def train_q_learning(env, metrics):
    """Train a Q-Learning agent."""
    plotter = LivePlotter()
    agent = QLearningAgent(env, QL_AGENT_CONFIG)
    agent.train(num_episodes=TRAIN_EPISODES)
    goal = tuple(env.config["goal_pos"])


    for ep in range(NUM_EPISODES):
        obs = env.reset()
        total_reward, done, steps = 0, False, 0
        
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == goal)
        plotter.update(metrics)
        if ep % 10 == 0:
            print(f"[QL] Episode {ep}/{NUM_EPISODES} | Reward: {total_reward:.2f}, Steps: {steps}")
            
    return agent, None  # no model used




def unwrap_state(state):
    """Unwrap a state from a tuple of states (e.g., from a MultiAgentWrapper)."""
    while isinstance(state, tuple):
        state = state[0]
    return state





# ====================== Visualization ========================
def visualize(env, agent, online_model, metrics):
    """Visualize the final results only (no duplicate/slow animation)."""
    plot_final_metrics(metrics)
    plot_gridworld(env, agent)

# ========================== Main =============================
def main():
    """Main execution function with error handling."""
    try:
        # Get agent type from user
        agent_type = input("Select agent (q_learning/dqn): ").strip().lower()
        if agent_type not in ["q_learning", "dqn"]:
            print("[ERROR] Invalid agent type. Must be 'q_learning' or 'dqn'.")
            return

        # Initialize environment and metrics
        env = GridWorldEnv(config=DEFAULT_ENV_CONFIG)
        metrics = initialize_metrics()
        agent, model = None, None

        # Train agent
        try:
            agent, model = (
                train_dqn(env, metrics) if agent_type == "dqn"
                else train_q_learning(env, metrics)
            )
        except Exception as train_err:
            print(f"[FATAL] Training failed: {train_err}", file=sys.stderr)
            traceback.print_exc()
            return

        # Visualize results
        try:
            visualize(env, agent, model, metrics)
        except Exception as viz_err:
            print(f"[FATAL] Visualization failed: {viz_err}", file=sys.stderr)
            traceback.print_exc()

        # Print training summary
        print("\nTraining Summary:")
        try:
            print(f"Avg Reward: {np.mean(metrics['rewards']):.2f}")
            print(f"Avg Steps: {np.mean(metrics['steps']):.2f}")
            print(f"Success Rate: {np.mean(metrics['successes']):.2%}")
        except Exception as summary_err:
            print(f"[WARN] Could not compute summary stats: {summary_err}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"[FATAL ERROR] Unhandled exception: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        try:
            plt.close('all')
        except Exception as close_err:
            print(f"[WARN] Could not close all figures: {close_err}", file=sys.stderr)
        print("[INFO] Exiting program.")

if __name__ == "__main__":
    main()
