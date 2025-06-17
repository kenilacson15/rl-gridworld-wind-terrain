import numpy as np
from envs.gridworld import GridWorldEnv
from config import AGENT_CONFIG
from typing import Any, Dict, Tuple
from tqdm import trange


class ValueIterationAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.V = np.zeros((env.grid_height, env.grid_width))


    def run_value_iteration(self):
        for i in range(self.config["max_iterations"]):
            delta = 0
            for state in ndindex(self.V.shape):
                v = self.V[state]


            if delta < self.config["theta"]:
                break



    def extract_policy(self):
        self.policy = np.zeros_like(self.V, dtype=int)
        for state in np.ndindex(self.V.shape):


            self.policy[state] = best_action


    def act(self, state):
        return self.policy[state]



    def save(self, filepath):

        pass


    def load(self, filepath):

        pass