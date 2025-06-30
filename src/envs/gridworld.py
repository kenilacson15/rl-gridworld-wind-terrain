from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

class GridWorldEnv(Env):
    """
    Custom GridWorld environment compatible with OpenAI Gym.
    Supports wind, stochastic terrain, and modern RL API.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        # Default configuration
        self.config = config or {
            "grid_size": (5, 5),
            "wind": [0, 0, 1, 0, 0],
            "stochastic_terrain": {},
            "goal_pos": (4, 4),
            "start_pos": (0, 0),
        }
        self.grid_height, self.grid_width = self.config["grid_size"]
        self.action_space = Discrete(4)  # 0-Up, 1-Right, 2-Down, 3-Left
        self.observation_space = Box(
            low=0,
            high=max(self.config["grid_size"]) - 1,
            shape=(2,),
            dtype=np.int32
        )
        self.agent_pos = np.array(self.config["start_pos"], dtype=np.int32)
        self.goal_pos = tuple(self.config["goal_pos"])
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agent_pos = np.array(self.config["start_pos"], dtype=np.int32)
        return self.agent_pos.copy()

    def step(self, action):
        direction = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }
        dx, dy = direction[action]
        new_pos = self.agent_pos + [dx, dy]
        # Stay in bounds
        new_pos[0] = np.clip(new_pos[0], 0, self.grid_height - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.grid_width - 1)
        # Apply wind
        col = new_pos[1]
        wind_strength = self.config["wind"][col] if col < len(self.config["wind"]) else 0
        if wind_strength > 0:
            new_pos[0] = max(0, new_pos[0] - wind_strength)
        # Stochastic terrain (ice slip)
        ice_prob = self.config["stochastic_terrain"].get("ice", 0)
        if ice_prob > 0 and self._rng.random() < ice_prob:
            slip_action = self._rng.integers(0, 4)
            dx_slip, dy_slip = direction[slip_action]
            slip_pos = new_pos + [dx_slip, dy_slip]
            # Stay in bounds
            slip_pos[0] = np.clip(slip_pos[0], 0, self.grid_height - 1)
            slip_pos[1] = np.clip(slip_pos[1], 0, self.grid_width - 1)
            new_pos = slip_pos
        self.agent_pos = new_pos
        done = tuple(self.agent_pos) == self.goal_pos
        reward = 0 if done else -1
        info = {"position": self.agent_pos.copy()}
        return self.agent_pos.copy(), reward, done, info

    def render(self, mode="human"):
        grid = np.full(self.config["grid_size"], ".", dtype=str)
        grid[tuple(self.agent_pos)] = "A"
        grid[self.goal_pos] = "G"
        for row in grid:
            print(" ".join(row))

    @property
    def state(self):
        return self.agent_pos.copy()
