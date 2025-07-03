import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from envs.gridworld import GridWorldEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.dqn import DQN, act as dqn_act
from config import (
    DEFAULT_ENV_CONFIG,
    QL_AGENT_CONFIG,
    DQN_AGENT_CONFIG,
    SARSA_AGENT_CONFIG
)
from utils.plotting import (
    GridWorldVisualizer as MatplotlibGridWorldVisualizer,
    MetricsVisualizer,
    plot_comparison,
    animate_gridworld_episode
)
from utils.game_visual import GridWorldVisualizer as PyGameVisualizer
import traceback
import time
import argparse


# ========================== Config ==========================
USE_PYGAME = True  # Enable PyGame visualization by default
REWARD_STEP_PENALTY = -1.0  # Encourage shorter paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for DQN
# ===========================================================


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


def train_dqn(env, metrics, use_pygame=USE_PYGAME):
    """Train a DQN agent."""
    num_episodes = DQN_AGENT_CONFIG.get("num_episodes", 50)
    max_steps = DQN_AGENT_CONFIG.get("max_steps", 200)

    # Initialize visualizers
    metrics_vis = MetricsVisualizer("DQN")
    pygame_vis = PyGameVisualizer() if use_pygame else None

    state = env.reset()
    state = unwrap_state(state)
    state_size = np.array(state).size
    action_size = env.action_space.n if hasattr(env, 'action_space') else 4
    goal = tuple(env.config["goal_pos"])

    # Initialize models
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

    for ep in range(num_episodes):
        state = env.reset()
        state = unwrap_state(state)
        total_reward, done = 0, False
        steps = 0

        for t in range(max_steps):
            action = dqn_act(state, online_model, epsilon)
            next_state, reward, done, *_ = env.step(action)
            next_state = unwrap_state(next_state)
            shaped_reward = reward + REWARD_STEP_PENALTY
            replay_buffer.append((state, action, shaped_reward, next_state, float(done)))

            # Update PyGame visualization
            if pygame_vis:
                time.sleep(0.05)
                pygame_vis.render(env, online_model, episode=ep, step=t, reward=total_reward)

            state = next_state
            total_reward += shaped_reward
            steps = t + 1

            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                update_dqn(minibatch, online_model, target_model, optimizer, gamma)

            if done:
                break

        if ep % sync_freq == 0:
            target_model.load_state_dict(online_model.state_dict())

        epsilon = max(epsilon_min, epsilon * decay)
        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == goal)
        metrics_vis.update(metrics)

        if ep % 10 == 0:
            print(f"[DQN] Episode {ep} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    return None, online_model


def train_q_learning(env, metrics, use_pygame=USE_PYGAME):
    """Train a Q-Learning agent."""
    metrics_vis = MetricsVisualizer("Q-Learning")
    pygame_vis = PyGameVisualizer() if use_pygame else None
    agent = QLearningAgent(env, QL_AGENT_CONFIG)
    goal = tuple(env.config["goal_pos"])

    num_episodes = QL_AGENT_CONFIG.get("num_episodes", 50)

    for ep in range(num_episodes):
        obs = env.reset()
        total_reward, done, steps = 0, False, 0

        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

            # Update PyGame visualization
            if pygame_vis:
                pygame_vis.render(env, agent, episode=ep, step=steps, reward=total_reward)

        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == goal)
        metrics_vis.update(metrics)

        if ep % 10 == 0:
            print(f"[QL] Episode {ep}/{num_episodes} | Reward: {total_reward:.2f}, Steps: {steps}")

    return agent, None


def train_sarsa(env, metrics, use_pygame=USE_PYGAME):
    """Train a SARSA agent."""
    agent = SarsaAgent(env, SARSA_AGENT_CONFIG)
    num_episodes = SARSA_AGENT_CONFIG.get("num_episodes", 50)

    metrics_vis = MetricsVisualizer("SARSA")
    pygame_vis = PyGameVisualizer() if use_pygame else None
    goal = tuple(env.config["goal_pos"])

    for ep in range(num_episodes):
        obs = env.reset()
        total_reward, done, steps = 0, False, 0

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            next_action = agent.act(next_obs) if not done else None

            # Update PyGame visualization
            if pygame_vis:
                pygame_vis.render(env, agent, episode=ep, step=steps, reward=total_reward)

            agent.update(obs, action, reward, next_obs, next_action)
            total_reward += reward
            steps += 1
            obs = next_obs
            action = next_action

        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == goal)
        metrics_vis.update(metrics)

        if ep % 10 == 0:
            print(f"[SARSA] Episode {ep}/{num_episodes} | Reward: {total_reward:.2f}, Steps: {steps}")

    return agent


def unwrap_state(state):
    """Unwrap a state from a tuple of states."""
    while isinstance(state, tuple):
        state = state[0]
    return state


def visualize(env, agent, model, metrics, agent_type):
    """Visualize the final results."""
    grid_vis = MatplotlibGridWorldVisualizer()
    metrics_vis = MetricsVisualizer(agent_type)

    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(1, 2, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])

    grid_vis.plot_gridworld(env, agent, fig, ax, cbar_ax=cbar_ax)
    metrics_vis.update(metrics)
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RL GridWorld with Visualization")
    parser.add_argument(
        "--agent", "-a",
        choices=["q_learning", "dqn", "sarsa"],
        default=None,
        help="Type of agent to train"
    )
    parser.add_argument(
        "--no-pygame",
        action="store_true",
        help="Disable PyGame visualization"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=None,
        help="Number of episodes to train"
    )
    return parser.parse_args()


def main():
    """Main execution function with error handling."""
    try:
        args = parse_args()
        use_pygame = not args.no_pygame

        # Get agent type via CLI or prompt
        agent_type = args.agent
        if not agent_type:
            print("\n===== GridWorld Reinforcement Learning =====")
            print("Available agents:")
            print("1. Q-Learning - Classic tabular Q-learning")
            print("2. DQN - Deep Q-Network with neural network")
            print("3. SARSA - State-Action-Reward-State-Action learning")

            while not agent_type:
                choice = input("\nSelect agent (1-3 or q_learning/dqn/sarsa): ").strip().lower()
                if choice in ["1", "q_learning", "q", "ql"]:
                    agent_type = "q_learning"
                elif choice in ["2", "dqn", "d"]:
                    agent_type = "dqn"
                elif choice in ["3", "sarsa", "s"]:
                    agent_type = "sarsa"
                else:
                    print("[ERROR] Invalid selection. Please choose a valid agent.")

        # Optionally override number of episodes
        num_episodes = args.episodes
        if not num_episodes:
            try:
                ep_input = input("\nNumber of training episodes (default: 50): ").strip()
                if ep_input:
                    num_episodes = int(ep_input)
            except ValueError:
                print("[WARN] Invalid episode count. Using default value.")
                num_episodes = 50

        print(f"\nStarting training with {agent_type.upper()} agent for {num_episodes} episodes.")
        print("Press Ctrl+C at any time to stop the training.")
        input("Press Enter to continue...")

        # Only now initialize environment and metrics
        env = GridWorldEnv(config=DEFAULT_ENV_CONFIG)
        metrics = initialize_metrics()

        agent, model = None, None

        # Train selected agent
        try:
            if agent_type == "dqn":
                DQN_AGENT_CONFIG["num_episodes"] = num_episodes
                agent, model = train_dqn(env, metrics, use_pygame)
            elif agent_type == "q_learning":
                QL_AGENT_CONFIG["num_episodes"] = num_episodes
                agent, model = train_q_learning(env, metrics, use_pygame), None
            elif agent_type == "sarsa":
                SARSA_AGENT_CONFIG["num_episodes"] = num_episodes
                agent, model = train_sarsa(env, metrics, use_pygame), None
        except Exception as train_err:
            print(f"[FATAL] Training failed: {train_err}", file=sys.stderr)
            traceback.print_exc()
            return

        # Visualization
        if not use_pygame:
            try:
                visualize(env, agent, model, metrics, agent_type.upper())
            except Exception as viz_err:
                print(f"[FATAL] Visualization failed: {viz_err}", file=sys.stderr)
                traceback.print_exc()

        # Print summary
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
        # Cleanup resources
        try:
            plt.close('all')
        except:
            pass
        try:
            import pygame
            pygame.quit()
        except:
            pass
        print("[INFO] Exiting program.")


if __name__ == "__main__":
    main()