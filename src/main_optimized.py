"""
Optimized main training script with enhanced visualization.

This version uses the async_visualization module to ensure smooth
rendering of both Pygame and Matplotlib visualizations.
"""

import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse
import traceback

# Set matplotlib backend for thread safety
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better thread safety

# Import project modules
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
    plot_comparison,
    animate_gridworld_episode
)
from utils.game_visual import GridWorldVisualizer as PyGameVisualizer
from utils.async_visualization import AsyncMetricsVisualizer, VisualizationCoordinator

# ========================== Config ==========================
USE_PYGAME = True  # Enable PyGame visualization by default
REWARD_STEP_PENALTY = -1.0  # Encourage shorter paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for DQN

# Visualization settings for performance optimization
PYGAME_MAX_FPS = 30.0         # Maximum FPS for PyGame updates
METRICS_UPDATE_INTERVAL = 0.5  # Seconds between metrics plot updates
# ===========================================================

# Create visualization coordinator for optimized rendering
viz_coordinator = VisualizationCoordinator(
    pygame_fps=PYGAME_MAX_FPS,
    metrics_update_interval=METRICS_UPDATE_INTERVAL
)


def initialize_metrics():
    """Initialize empty metrics dictionary."""
    return {"rewards": [], "steps": [], "successes": []}


def record_metrics(metrics, reward, steps, goal_reached):
    """Record episode metrics."""
    metrics["rewards"].append(reward)
    metrics["steps"].append(steps)
    metrics["successes"].append(1 if goal_reached else 0)


# ======================== DQN Logic ==========================
def unwrap_state(state):
    """Unwrap a state from a tuple of states."""
    while isinstance(state, tuple):
        state = state[0]
    return state


def update_dqn(batch, online_model, target_model, optimizer, gamma):
    """Update DQN models using a batch of experiences (Double DQN)."""
    # Process batch into tensors
    # Convert list of experiences to numpy arrays first for better performance
    np_states = np.array([unwrap_state(s[0]) for s in batch])
    np_actions = np.array([s[1] for s in batch])
    np_rewards = np.array([s[2] for s in batch])
    np_next_states = np.array([unwrap_state(s[3]) for s in batch])
    np_dones = np.array([s[4] for s in batch], dtype=np.bool_)
    
    # Convert numpy arrays to tensors
    states = torch.FloatTensor(np_states).to(DEVICE)
    actions = torch.LongTensor(np_actions).to(DEVICE)
    rewards = torch.FloatTensor(np_rewards).to(DEVICE)
    next_states = torch.FloatTensor(np_next_states).to(DEVICE)
    dones = torch.FloatTensor(np_dones).to(DEVICE)

    # Compute target Q values using Double DQN approach
    with torch.no_grad():
        # Get best actions according to online model
        best_actions = online_model(next_states).max(1)[1].unsqueeze(1)
        # Evaluate those actions using target network
        next_q_values = target_model(next_states).gather(1, best_actions).squeeze(1)
        # Calculate target using Bellman equation
        target = rewards + gamma * next_q_values * (1 - dones)

    # Compute current Q values and loss
    q_values = online_model(states)
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = torch.nn.MSELoss()(q_selected, target)

    # Update online model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_dqn(env, metrics, use_pygame=USE_PYGAME):
    """Train a DQN agent with optimized visualization."""
    num_episodes = DQN_AGENT_CONFIG.get("num_episodes", 50)
    max_steps = DQN_AGENT_CONFIG.get("max_steps", 200)
    batch_size = DQN_AGENT_CONFIG.get("batch_size", 64)
    gamma = DQN_AGENT_CONFIG.get("gamma", 0.99)
    epsilon_start = DQN_AGENT_CONFIG.get("epsilon_start", 1.0)
    epsilon_end = DQN_AGENT_CONFIG.get("epsilon_end", 0.1)
    epsilon_decay = DQN_AGENT_CONFIG.get("epsilon_decay", 0.995)
    target_update = DQN_AGENT_CONFIG.get("target_update", 10)

    # Initialize visualizers
    metrics_vis = viz_coordinator.create_metrics_visualizer("DQN")
    pygame_vis = PyGameVisualizer() if use_pygame else None

    # Get environment dimensions
    state = env.reset()
    state = unwrap_state(state)
    input_dim = len(state)
    output_dim = env.action_space.n

    # Initialize DQN models
    online_model = DQN(input_dim, output_dim).to(DEVICE)
    target_model = DQN(input_dim, output_dim).to(DEVICE)
    target_model.load_state_dict(online_model.state_dict())
    optimizer = torch.optim.Adam(online_model.parameters(), lr=0.001)

    # Initialize replay buffer
    buffer_size = DQN_AGENT_CONFIG.get("buffer_size", 10000)
    replay_buffer = deque(maxlen=buffer_size)
    goal = tuple(env.config["goal_pos"])
    
    # Initialize metrics
    avg_losses = []
    epsilon = epsilon_start
    
    # Create a deque to track frame rendering times for adaptive performance
    render_times = deque(maxlen=30)

    # Main training loop
    for ep in range(num_episodes):
        state = env.reset()
        state = unwrap_state(state)
        total_reward, done, t = 0, False, 0
        episode_losses = []

        while not done and t < max_steps:
            # Select action with epsilon-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action = dqn_act(online_model, state_tensor)

            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            next_state = unwrap_state(next_state)
            total_reward += reward
            t += 1

            # Store transition in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            # Update PyGame visualization with rate limiting
            if pygame_vis and viz_coordinator.should_update_pygame():
                render_start = time.time()
                pygame_vis.render(env, online_model, episode=ep, step=t, reward=total_reward)
                render_time = time.time() - render_start
                render_times.append(render_time)
            
            # Update model if enough samples in buffer
            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                loss = update_dqn(minibatch, online_model, target_model, optimizer, gamma)
                episode_losses.append(loss)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Update target network periodically
        if ep % target_update == 0:
            target_model.load_state_dict(online_model.state_dict())

        # Record metrics
        is_success = tuple(env.agent_pos) == goal
        record_metrics(metrics, total_reward, t, is_success)
        
        # Log average loss if available
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            avg_losses.append(avg_loss)
        
        # Update metrics visualization (non-blocking)
        metrics_vis.update(metrics)
        
        # Calculate average render time for performance monitoring
        if render_times:
            avg_render_time = sum(render_times) / len(render_times)
            fps = 1.0 / max(0.001, avg_render_time)
        else:
            fps = 0
        
        # Log progress
        if ep % 5 == 0 or ep == num_episodes - 1:
            print(f"[DQN] Episode {ep}/{num_episodes} | Reward: {total_reward:.2f} | "
                 f"Steps: {t} | Epsilon: {epsilon:.3f} | FPS: {fps:.1f}")

    return online_model, avg_losses


def train_q_learning(env, metrics, use_pygame=USE_PYGAME):
    """Train a Q-Learning agent with optimized visualization."""
    metrics_vis = viz_coordinator.create_metrics_visualizer("Q-Learning")
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
            agent.update(obs, action, reward, done)
            total_reward += reward
            steps += 1

            # Update PyGame visualization with rate limiting
            if pygame_vis and viz_coordinator.should_update_pygame():
                pygame_vis.render(env, agent, episode=ep, step=steps, reward=total_reward)

        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == goal)
        metrics_vis.update(metrics)

        if ep % 10 == 0 or ep == num_episodes - 1:
            print(f"[QL] Episode {ep}/{num_episodes} | Reward: {total_reward:.2f}, Steps: {steps}")

    return agent, None


def train_sarsa(env, metrics, use_pygame=USE_PYGAME):
    """Train a SARSA agent with optimized visualization."""
    metrics_vis = viz_coordinator.create_metrics_visualizer("SARSA")
    pygame_vis = PyGameVisualizer() if use_pygame else None
    agent = SarsaAgent(env, SARSA_AGENT_CONFIG)
    goal = tuple(env.config["goal_pos"])

    num_episodes = SARSA_AGENT_CONFIG.get("num_episodes", 50)

    for ep in range(num_episodes):
        obs = env.reset()
        action = agent.act(obs)
        total_reward, done, steps = 0, False, 0

        while not done:
            next_obs, reward, done, _ = env.step(action)
            next_action = agent.act(next_obs) if not done else None

            # Update PyGame visualization with rate limiting
            if pygame_vis and viz_coordinator.should_update_pygame():
                pygame_vis.render(env, agent, episode=ep, step=steps, reward=total_reward)

            agent.update(obs, action, reward, next_obs, next_action)
            total_reward += reward
            steps += 1
            obs = next_obs
            action = next_action

        record_metrics(metrics, total_reward, steps, tuple(env.agent_pos) == goal)
        metrics_vis.update(metrics)

        if ep % 10 == 0 or ep == num_episodes - 1:
            print(f"[SARSA] Episode {ep}/{num_episodes} | Reward: {total_reward:.2f}, Steps: {steps}")

    return agent, None


def visualize(env, agent, model, metrics, agent_type):
    """Visualize the final results."""
    grid_vis = MatplotlibGridWorldVisualizer()
    metrics_vis = viz_coordinator.create_metrics_visualizer(agent_type)

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
    parser.add_argument(
        "--no-threading", 
        action="store_true",
        help="Disable threading for metrics visualization (may be more stable on some systems)"
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
            print("3. SARSA - On-policy tabular learning")
            choice = input("Select agent type (1-3): ")
            agent_types = ["q_learning", "dqn", "sarsa"]
            try:
                agent_type = agent_types[int(choice) - 1]
            except:
                print("Invalid choice. Using Q-Learning as default.")
                agent_type = "q_learning"
                
        # Update episode counts if specified
        if args.episodes is not None:
            QL_AGENT_CONFIG["num_episodes"] = args.episodes
            DQN_AGENT_CONFIG["num_episodes"] = args.episodes
            SARSA_AGENT_CONFIG["num_episodes"] = args.episodes
        
        # Setup environment
        print(f"Initializing environment...")
        env_config = DEFAULT_ENV_CONFIG.copy()
        env_config["reward_step"] = REWARD_STEP_PENALTY
        env = GridWorldEnv(env_config)
        metrics = initialize_metrics()
        
        print(f"Starting training with {agent_type.upper()} agent...")
        
        # Train the selected agent
        agent, model = None, None
        if agent_type == "q_learning":
            agent, _ = train_q_learning(env, metrics, use_pygame)
        elif agent_type == "dqn":
            model, _ = train_dqn(env, metrics, use_pygame)
        elif agent_type == "sarsa":
            agent, _ = train_sarsa(env, metrics, use_pygame)
        else:
            print(f"Unknown agent type: {agent_type}")
            return

        # Show final visualization (non-PyGame)
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
            # Clean up visualization threads and resources
            viz_coordinator.cleanup()
            plt.close('all')
        except:
            pass


if __name__ == "__main__":
    main()
