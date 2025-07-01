from typing import Dict, Tuple, Any, Literal

# ====== Type Aliases ======
AlgorithmOption = Literal["q_learning", "sarsa", "dqn"]
UpdateStrategy = Literal["max", "expected", "softmax"]

# ====== Stochastic Terrain Presets ======
STOCHASTIC_TERRAIN_WINDY: Dict[str, float] = {
    "ice": 0.1,   # 10% chance of slipping
    "swamp": 0.3   # 30% chance of slow-down
}

# ====== Environment Configuration Template ======
DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    "grid_size": (5, 5),                # (rows, cols)
    "start_pos": (0, 0),                # Starting position (row, col)
    "goal_pos": (4, 4),                 # Goal position (row, col)
    "wind": [0, 0, 1, 0, 0],            # Wind strength per column
    "stochastic_terrain": {},           # Dict[str, float], e.g., {"ice": 0.1, "swamp": 0.3}
    "max_steps": 200,                   # Max steps per episode
    "reward_structure": {
        "step": -1,                     # Penalty per step
        "goal": 100,                    # Reward for reaching goal
        "swamp_penalty": -5,            # Penalty for swamp
        "ice_slip_penalty": -2          # Penalty for slipping
    }
}

# ====== Multi-Environment Configurations ======
ENVIRONMENTS: Dict[str, Dict[str, Any]] = {
    "5x5_basic": {**DEFAULT_ENV_CONFIG},
    "10x10_windy": {
        **DEFAULT_ENV_CONFIG,
        "grid_size": (10, 10),
        "goal_pos": (9, 9),
        "wind": [0, 1, 1, 2, 2, 1, 0, 0, 1, 0],
        "stochastic_terrain": STOCHASTIC_TERRAIN_WINDY,
    },
    "7x7_dense": {
        **DEFAULT_ENV_CONFIG,
        "grid_size": (7, 7),
        "goal_pos": (6, 6),
        "wind": [0, 1, 1, 1, 1, 1, 0],
    }
}

# ====== Agent (Q-learning) Configuration ======
QL_AGENT_CONFIG: Dict[str, Any] = {
    "algorithm": "q_learning",      # Could be "q_learning", "sarsa", "dqn", etc.
    "alpha": 0.1,                   # Learning rate
    "gamma": 0.99,                  # Discount factor
    "epsilon": 0.1,                 # Exploration rate
    "min_epsilon": 0.01,            # Floor value for epsilon
    "decay_rate": 0.995,            # Epsilon decay per episode
    "init_q": 0.0,                  # Initial Q-value
    "adaptive_lr": False,           # Enable learning rate annealing
    "update_strategy": "max",      # Can switch to "expected" or "softmax"
}

# ====== Agent (DQN) Configuration ======
DQN_AGENT_CONFIG: Dict[str, Any] = {
    "algorithm": "dqn",
    "hidden_dim": 128,
    "buffer_size": 10000,
    "batch_size": 64,
    "sync_frequency": 5,
    "gamma": 0.99,
    "learning_rate": 1e-3,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "reward_step_penalty": -1.0,
    "max_grad_norm": 1.0,
    "num_episodes": 500,
    "max_steps": 200,
    # Add more DQN-specific options as needed
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
