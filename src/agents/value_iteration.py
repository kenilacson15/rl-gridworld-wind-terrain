import numpy as np
from numpy import ndindex
from envs.gridworld import GridWorldEnv
from config import QL_AGENT_CONFIG
from typing import Any, Dict, Tuple
from tqdm import trange


class ValueIterationAgent:
    def __init__(self, env: GridWorldEnv, config: Dict[str, Any]) -> None:
        self.env = env
        self.config = config
        self.V = np.zeros((env.grid_height, env.grid_width))
        self.policy: np.ndarray = None


    def run_value_iteration(self):
        for _ in trange(self.config["max_iterations"], desc="Value Iteration"):
            delta = 0
            for state in ndindex(self.V.shape):
                if self.env.is_terminal(state):
                    continue
                v = self.V[state]
                max_value = -float('inf')
                for action in range(self.env.action_space.n):
                    next_state, reward, _, _ = self.env.step(state, action)
                    max_value = max(max_value, reward + self.config["gamma"] * self.V[next_state])
                self.V[state] = max_value
                delta = max(delta, abs(v - self.V[state]))
            
            if delta < self.config["theta"]:
                break
                 

    def extract_policy(self):
        self.policy = np.zeros_like(self.V, dtype=int)
        for state in np.ndindex(self.V.shape):
            if self.env.is_terminal(state):
                continue
            best_action = None
            best_value = -float('inf')
            for action in range(self.env.action_space.n):
                next_state, reward, _, _ = self.env.step(state, action)
                value = reward + self.config["gamma"] * self.V[next_state]
                if value > best_value:
                    best_value = value
                    best_action = action
                
            self.policy[state] = best_action


    def act(self, state: Tuple[int, int]) -> int:
        if self.policy is None:
            raise ValueError("Policy not initialized. Run value iteration first.")
        return self.policy[state]

    def save(self, filepath):

        pass


    def load(self, filepath):

        pass