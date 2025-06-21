import numpy as np
from numpy import ndindex
from envs.gridworld import GridWorldEnv
from config import AGENT_CONFIG
from typing import Any, Dict, Tuple
from tqdm import trange


for iteration in trange(self.config["max_iterations"]):


class PolicyIterationAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.V = np.zeros((height, width))
        self.policy = np. random.randint(0, 4, size=(height, width))
        height, width = env.grid_height, env.grid_width
        self.transitions = np.zeros((height, width, 4, 2), dtype=int)
        self.rewards = np.zeros((height, width, 4))

        for i in range(height):
            for j in range(width):
                for a in range(4):
                    next_state, reward = self.env.get_transition_reward((i, j), a)
                    self.transitions[i, j, a] = next_state
                    self.rewards[i, j, a] = reward

    def policy_iteration(self):
        gamma = self.config["gamma"]
        theta = self.config["theta"]
        white True:
        old_V = self.V.copy()

        best_actions = self.policy
        next_i = self.transitions[np.arange(self.V.shape[0])[:, None], np.arange(self.V.shape[1]), best_actions, 0]
        next_j = self.transitions[np.arange(self.V.shape[0])[:, None], np.arange(self.V.shape[1]), best_actions, 1]
        self.V = self.rewards[np.arange(self.V.shape[0])[:, None], np.arange(self.V.shape[1]), best_actions] + gamma * old_V[next_i, next_j]
        delta = np.max(np.abs(self.V - old_V))
        if delta < theta:

            break


def policy_iteration(self):
    for _ in range(self.config["max_iterations"]):
        delta = 0
        for state in ndindex(self.V.shape):

        if delta < self.config["theta"]:
            break


def policy_evaluation(self):
    gamma = self.config["gamma"]
    theta = self.config["theta"]
    
    while True:
        delta = 0
        for i, j in ndindex(self.V.shape):
            v = self.V[i, j]
            action = self.policy[i, j]
            next_state = self.transitions[i, j, action]
            reward = self.rewards[i, j, action]
            self.V[i, j] = reward + gamma * self.V[next_state[0], next_state[1]]
            delta = max(delta, abs(v - self.V[i, j]))

        if delta < theta:
            break


for state in ndindex(self.V.shape):
    pass


def policy_improvement(self):
    policy_stable = True
    for i, j in np.ndindex(self.V.shape):
        old_action = self.policy[i, j]


        q_values = []
        for a in range(4):
            next_state = self.transitions[i, j, a]
            reward = self.rewards[i, j, a]
            q_value = reward + self.config["gamma"] * self.V[next_state[0], next_state[1]]
            q_values.append(q_value)

        
        best_action = np.argmax(q_values)
        self.policy[i, j] = best_action

        if best_action != old_action:
            policy_stable = False

        return policy_stable
def run_policy_iteration(self):
    for _ in trange(self.config["max_iterations"]):
        self.policy_evaluation()
        policy_stable = self.policy_improvement()
        if policy_stable:
            break

def act(self, state):
    return self.policy[state]