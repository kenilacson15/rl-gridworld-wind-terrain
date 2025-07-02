import numpy as np
import random

class SarsaAgent:
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.states_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    

        self.q_table = np.zeros((state_space_size, action_space_size))


    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state])



    def update(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state, action]
        next_q = self.q_table(next_state, next_action)
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[state, action] += self.alpha * td_error


    def decay_epsilon(self, decay_rate=0.99, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)




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