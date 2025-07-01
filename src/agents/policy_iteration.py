import numpy as np
from numpy import ndindex
from envs.gridworld import GridWorldEnv
from config import QL_AGENT_CONFIG
from tqdm import trange


class PolicyIterationAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        height, width = env.grid_height, env.grid_width
        self.V = np.zeros((height, width))
        self.policy = np.random.randint(0, 4, size=(height, width))
        self.transitions = np.zeros((height, width, 4, 2), dtype=int)
        self.rewards = np.zeros((height, width, 4))

        # Fix: Remove call to non-existent get_transition_reward, use a placeholder or implement logic here
        for i in range(height):
            for j in range(width):
                for a in range(4):
                    # Placeholder: assume next_state is (i, j) and reward is 0
                    # Replace with actual transition logic if available
                    next_state = (i, j)
                    reward = 0
                    self.transitions[i, j, a] = next_state
                    self.rewards[i, j, a] = reward

    def policy_evaluation(self):
        """Evaluate the current policy until convergence."""
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

    def policy_improvement(self):
        """Update the policy greedily based on the value function."""
        policy_stable = True
        for i, j in ndindex(self.V.shape):
            old_action = self.policy[i, j]
            q_values = []
            for a in range(4):
                next_state = self.transitions[i, j, a]
                reward = self.rewards[i, j, a]
                q_values.append(reward + self.config["gamma"] * self.V[next_state[0], next_state[1]])
            best_action = np.argmax(q_values)
            self.policy[i, j] = best_action
            if best_action != old_action:
                policy_stable = False
        return policy_stable

    def run_policy_iteration(self):
        """Perform the full policy iteration."""
        for _ in trange(self.config["max_iterations"]):
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def act(self, state):
        """Return the best action for a given state."""
        return self.policy[state]


#  STEP 5: Main Execution
if __name__ == "__main__":
    # Instantiate the environment
    from envs.gridworld import GridWorldEnv
    from config import QL_AGENT_CONFIG

    env = GridWorldEnv()
    agent = PolicyIterationAgent(env, QL_AGENT_CONFIG)

    agent.run_policy_iteration()

    # Inspect final results
    print("V shape:", agent.V.shape)
    print("Policy shape:", agent.policy.shape)
    print("Transitions shape:", agent.transitions.shape)
    print("Reward shape:", agent.rewards.shape)
