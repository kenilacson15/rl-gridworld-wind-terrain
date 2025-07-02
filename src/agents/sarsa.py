import numpy as np
from envs.gridworld import GridWorldEnv
from config import SARSA_AGENT_CONFIG as AGENT_CONFIG

class SarsaAgent:
    def __init__(self, env, config):
        """Initialize SARSA agent.
        
        Args:
            env: GridWorld environment instance
            config: Configuration dictionary with learning parameters
        """
        self.env = env
        self.config = config
        
        # Initialize Q-table
        self.Q = np.zeros((env.grid_height, env.grid_width, env.action_space.n))
        
        # Get parameters from config
        self.alpha = config.get("alpha", 0.1)        # Learning rate
        self.gamma = config.get("gamma", 0.99)       # Discount factor
        self.epsilon = config.get("epsilon", 1.0)    # Starting epsilon
        self.min_epsilon = config.get("min_epsilon", 0.01)
        self.decay_rate = config.get("decay_rate", 0.995)
        
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.Q[state[0], state[1]])
    
    def train(self, num_episodes):
        """Train the agent for given number of episodes."""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.act(state)
            total_reward = 0
            done = False
            
            while not done:
                # Take action and observe next state
                next_state, reward, done, _ = self.env.step(action)
                
                # Choose next action using epsilon-greedy
                next_action = self.act(next_state) if not done else None
                
                # SARSA update
                self._update_q_value(state, action, reward, next_state, next_action)
                
                # Move to next state-action pair
                state = next_state
                action = next_action
                total_reward += reward
                
                if done:
                    episode_rewards.append(total_reward)
                    break
            
            # Decay epsilon
            self._decay_epsilon()
            
            if episode % 10 == 0:
                print(f"[SARSA] Episode {episode}/{num_episodes} | "
                      f"Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards
    
    def _update_q_value(self, state, action, reward, next_state, next_action):
        """Update Q-value using SARSA update rule."""
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
        """Decay epsilon value."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)