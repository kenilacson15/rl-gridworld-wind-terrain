import numpy as np
from envs.gridworld import GridWorldEnv
from config import QL_AGENT_CONFIG as AGENT_CONFIG

class QLearningAgent:

    """
    Q-Learning agent for reinforcement learning in discrete state-action spaces.
    
    Attributes:
        env: The environment the agent interacts with
        config: Configuration dictionary containing hyperparameters
        n_actions: Number of possible actions
        Q: Q-value table of shape (height, width, actions)
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.n_actions = env.action_space.n
        self.state_shape = (env.grid_height, env.grid_width)
        self.Q = np.zeros(self.state_shape + (self.n_actions,))
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

            
            self.decay_epsilon()




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



    def decay_epsilon(self):
        if self.epsilon > self.config["min_epsilon"]:
            self.epsilon *= self.config["decay_rate"]

    
    def update(self, state, action, reward, next_state):

        self._update_q_value(state, action, reward, next_state, done=False)

    
    def get_policy(self):

        policy = {}
        for row in range(self.state_shape[0]):
            for col in range(self.state_shape[1]):
                state = (row, col)
                policy[state] = np.argmax(self.Q[row, col])
        
        return policy

