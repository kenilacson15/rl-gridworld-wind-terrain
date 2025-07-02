import numpy as np
import random
from envs.gridworld import GridWorldEnv
from config import SARSA_AGENT_CONFIG as AGENT_CONFIG

class SarsaAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.n_actions = env.action_space.n
        self.Q = np.zeros((env.grid_height, env.grid_width, env.action_space.n))
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]

    def train(self, num_episodes):
        episode_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.act(state)
            total_reward = 0
            done = False
            
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.act(next_state) if not done else None
                
                self._update_q_value(state, action, reward, next_state, next_action)
                total_reward += reward
                
                state = next_state
                action = next_action
                
                if done:
                    episode_rewards.append(total_reward)
                    
            self._decay_epsilon()
        return episode_rewards

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state[0], state[1]])

    def _update_q_value(self, state, action, reward, next_state, next_action):
        current_q = self.Q[state[0], state[1], action]
        
        if next_action is not None:
            next_q = self.Q[next_state[0], next_state[1], next_action]
            target_q = reward + self.gamma * next_q
        else:
            target_q = reward
            
        self.Q[state[0], state[1], action] = (
            (1 - self.alpha) * current_q + self.alpha * target_q
        )

    def _decay_epsilon(self):
        if self.epsilon > self.config["min_epsilon"]:
            self.epsilon *= self.config["decay_rate"]




agent = SarsaAgent(state_space_size=10, action_space_size=4)
state = env.reset()
action = agent.get_action(state)



for _ in range(1000):
    next_state, reward, done, _ = env.step(action)
    next_action = agent.get_action(next_state)
    agent.update(state, action, reward, next_state, next_action)
    state, action = next_state, next_action

    if done:
        state = env.reset()
        action = agent.get_action(state)