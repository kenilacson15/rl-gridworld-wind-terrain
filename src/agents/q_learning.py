import numpy as np
from envs.gridworld import GridWorldEnv
from config import QL_AGENT_CONFIG as AGENT_CONFIG

class QLearningAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.n_actions = env.action_space.n
        self.Q = np.zeros((env.grid_height, env.grid_width, env.action_space.n))
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]



    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)

                self._update_q_value(state, action, reward, next_state, done)

                state = next_state




    def act(self, state):
        row, col = state
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state[0], state[1]])
    

    def _update_q_value(self, state, action, reward, next_state, done):
        
        current_q = self.Q[state[0], state[1], action]
        target_q = reward


        if not done:
            max_q_next = np.max(self.Q[next_state[0], next_state[1]])
            target_q += self.gamma * max_q_next
        
        self.Q[state[0], state[1], action] = (
            (1 - self.alpha) * current_q + self.alpha * target_q
        )



    def _decay_epsilon(self):
        if self.epsilon > self.config["min_epsilon"]:
            self.epsilon *= self.config["decay_rate"]

