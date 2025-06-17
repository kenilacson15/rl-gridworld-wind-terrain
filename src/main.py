from envs.gridworld import GridworldEnv
from src.agents.value_iteration import ValueIterationAgent
from src.config import ENV_CONFIG, AGENT_CONFIG


env = GridworldEnv(config=ENV_CONFIG)

obs = env.reset()


done = False

while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)


print(f"Episode finished in {step_count} steps with total reward {total_reward}")