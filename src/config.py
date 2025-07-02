from typing import Dict, Tuple, Any, Literal

# ====== Environment Configuration Template ======
DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    "grid_size": (5, 5),                # (rows, cols)
    "start_pos": (0, 0),                # Starting position (row, col)
    "goal_pos": (4, 4),                 # Goal position (row, col)
    "wind": [0, 0, 1, 0, 0],            # Wind strength per column (0 = no wind)
    "stochastic_terrain": {            # Terrain types and probabilities
        "ice": 0.2,                    # 20% slip chance
        "swamp": 0.5                   # 50% slow-down chance
    },
    "max_steps": 200,                  # Optional: max steps per episode
    "reward_structure": {
        "step": -1,                    # Penalty per step
        "goal": 100,                   # Reward for reaching goal
        "swamp_penalty": -5,          # Optional: penalty for swamp
        "ice_slip_penalty": -2        # Optional: penalty for slipping
    }
}

# ====== Multi-Environment Configurations ======
ENVIRONMENTS: Dict[str, Dict[str, Any]] = {
    "5x5_basic": DEFAULT_ENV_CONFIG,
    "10x10_windy": {
        **DEFAULT_ENV_CONFIG,
        "grid_size": (10, 10),
        "goal_pos": (9, 9),
        "wind": [0, 1, 1, 2, 2, 1, 0, 0, 1, 0],
        "stochastic_terrain": {"ice": 0.1, "swamp": 0.3}
    },
    "7x7_dense": {
        **DEFAULT_ENV_CONFIG,
        "grid_size": (7, 7),
        "goal_pos": (6, 6),
        "wind": [0, 1, 1, 1, 1, 1, 0],
        "stochastic_terrain": {"ice": 0.25, "swamp": 0.6}
    }
}

# ====== Agent (Q-learning) Configuration ======
QL_AGENT_CONFIG: Dict[str, Any] = {
    "algorithm": "q_learning",      # Could be "q_learning", "sarsa", "dqn", etc.
    "alpha": 0.1,                   # Learning rate
    "gamma": 0.99,                  # Discount factor
    "epsilon": 0.1,                 # Exploration rate
    "min_epsilon": 0.01,           # Floor value for epsilon
    "decay_rate": 0.995,           # Epsilon decay per episode
    "init_q": 0.0,                 # Optional: initial Q-value
    "adaptive_lr": False,          # Optional: enable learning rate annealing
    "update_strategy": "max",      # Can switch to "expected" or "softmax"
}

# ====== Path Configuration ======
PATHS: Dict[str, str] = {
    "data": "data/processed/gridworld_v1.pkl",
    "models": "models/trained/q_learning_agent.pkl",
    "logs": "logs/training_log.csv",
    "plots": "results/plots/",
}

# ====== Example for curriculum training loop ======
# for level_name, config in ENVIRONMENTS.items():
#     env = GridWorldEnv(config=config)
#     agent = QLearningAgent(env, QL_AGENT_CONFIG)
#     agent.train(num_episodes=500)
