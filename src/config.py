from typing import Dict, Tuple, Any

# ====== Environment Configuration ======
ENV_CONFIG: Dict[str, Any] = {
    "grid_size": (5, 5),                # (rows, cols)
    "start_pos": (0, 0),                # Starting position (row, col)
    "goal_pos": (4, 4),                 # Goal position (row, col)
    "wind": [0, 0, 1, 0, 0],            # Wind strength per column
    "stochastic_terrain": {             # Terrain types and their probabilities
        "ice": 0.2,                     # 20% chance to slip on ice
        "swamp": 0.5                    # 50% chance to slow movement on swamp
    }
}

# ====== Agent/Algorithm Configuration ======
AGENT_CONFIG: Dict[str, Any] = {
    "gamma": 0.95,          # Discount factor
    "theta": 1e-6,          # Convergence threshold for Value Iteration
    "max_iterations": 1000, # Max iterations for planning algorithms
}

# ====== Paths for Data/Models ======
PATHS: Dict[str, str] = {
    "data": "data/processed/gridworld_v1.pkl",
    "models": "models/trained/value_iteration_optimal.pkl",
}

# ====== Multi-Environment Support ======
ENVIRONMENTS: Dict[str, Dict[str, Any]] = {
    "small_grid": {
        "grid_size": (5, 5),
        "start_pos": (0, 0),
        "goal_pos": (4, 4),
        "wind": [0, 0, 1, 0, 0],
        "stochastic_terrain": {"ice": 0.2}
    },
    "large_grid": {
        "grid_size": (10, 10),
        "start_pos": (0, 0),
        "goal_pos": (9, 9),
        "wind": [0, 1, 1, 2, 2, 1, 0, 0, 1, 0],
        "stochastic_terrain": {"ice": 0.1, "swamp": 0.3}
    }
}

# ====== Example Usage ======
# wind_strength = ENV_CONFIG["wind"][column]
# env_cfg = ENVIRONMENTS["small_grid"]