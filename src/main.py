from envs.gridworld import GridWorldEnv
from agents.value_iteration import ValueIterationAgent
from config import ENV_CONFIG, AGENT_CONFIG


env = GridWorldEnv(config=ENV_CONFIG)

obs = env.reset()


done = False


step_count = 0
total_reward = 0

while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    step_count += 1
    total_reward += reward


print(f"Episode finished in {step_count} steps with total reward {total_reward}")