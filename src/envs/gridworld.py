from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

class GridworldEnv(Env):
    """
    In this implementation, I define a custom Gridworld environment compatible with OpenAI Gym.
    This environment simulates a 2D grid where an agent must navigate from a start position to a goal,
    potentially affected by wind and slippery (stochastic) terrain.

    The environment includes:
    - Discrete action space: Up, Right, Down, Left
    - Continuous state space: Agent's (x, y) coordinates
    - Optional wind effects that push the agent upward
    - Optional stochastic terrain that can cause the agent to slip
    """

    def __init__(self, config=None):
        """
        Initializes the Gridworld environment. Default parameters are used if no configuration is provided.

        Parameters:
        - config (dict, optional): Dictionary with environment parameters:
            - "grid_size": Tuple defining grid dimensions (height, width)
            - "wind": List defining upward wind strength in each column
            - "stochastic_terrain": Dict defining slip probabilities (e.g., {"ice": 0.2})
            - "goal_pos": Tuple with goal coordinates
            - "start_pos": Tuple with starting coordinates
        """
        super(GridworldEnv, self).__init__()

        # Default configuration
        self.config = config or {
            "grid_size": (5, 5),
            "wind": [0, 0, 1, 0, 0],
            "stochastic_terrain": {"ice": 0.2},
            "goal_pos": (4, 4),
            "start_pos": (0, 0),
        }

        self.grid_height, self.grid_width = self.config["grid_size"]

        # The agent can take 4 discrete actions: 0-Up, 1-Right, 2-Down, 3-Left
        self.action_space = Discrete(4)

        # The observation is a 2D position in the grid: (row, column)
        self.observation_space = Box(
            low=0,
            high=max(self.config["grid_size"]) - 1,
            shape=(2,),
            dtype=np.int32
        )

        # Initialize agent's position
        self.agent_pos = np.array(self.config["start_pos"])

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
        - observation (np.array): The starting position of the agent
        """
        self.agent_pos = np.array(self.config["start_pos"])
        return self.agent_pos

    def step(self, action):
        """
        Executes a single step in the environment given an action.

        Parameters:
        - action (int): One of the four discrete actions (0: Up, 1: Right, 2: Down, 3: Left)

        Returns:
        - next_state (np.array): New position of the agent
        - reward (int): Reward signal (+0 for reaching goal, -1 otherwise)
        - done (bool): Whether the episode has terminated
        - info (dict): Additional diagnostic information (e.g., current position)
        """

        # Define how each action modifies position (dy, dx)
        direction = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }

        dx, dy = direction[action]
        new_pos = self.agent_pos + [dx, dy]

        # Check if new position is within grid bounds
        if (
            new_pos[0] < 0 or new_pos[0] >= self.grid_height or
            new_pos[1] < 0 or new_pos[1] >= self.grid_width
        ):
            new_pos = self.agent_pos  # Agent stays in place if move is invalid

        # Apply wind effect based on column
        col = new_pos[1]
        wind_strength = self.config["wind"][col]
        if wind_strength > 0:
            new_pos[0] -= wind_strength  # Push agent upward
            new_pos[0] = max(0, new_pos[0])  # Prevent going above top boundary

        # Simulate stochastic slip due to terrain (e.g., ice)
        ice_prob = self.config["stochastic_terrain"].get("ice", 0)
        if np.random.rand() < ice_prob:
            slip_action = np.random.choice([0, 1, 2, 3])
            dx_slip, dy_slip = direction[slip_action]
            slip_pos = new_pos + [dx_slip, dy_slip]
            # Ensure the slipped position is still valid
            if (
                0 <= slip_pos[0] < self.grid_height and
                0 <= slip_pos[1] < self.grid_width
            ):
                new_pos = slip_pos

        self.agent_pos = new_pos

        # Determine if goal has been reached
        if np.array_equal(self.agent_pos, self.config["goal_pos"]):
            reward = 0  # No penalty for reaching the goal
            done = True
        else:
            reward = -1  # Penalty to encourage efficiency
            done = False

        info = {"position": self.agent_pos.tolist()}
        return self.agent_pos, reward, done, info

    def render(self):
        """
        Renders the current grid state to the console.
        'A' denotes the agent, and 'G' denotes the goal.
        """
        grid = np.full(self.config["grid_size"], ".", dtype=str)
        grid[tuple(self.agent_pos)] = "A"
        grid[tuple(self.config["goal_pos"])] = "G"

        for row in grid:
            print(" ".join(row))
