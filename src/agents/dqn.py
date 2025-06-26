import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from envs.gridworld import GridWorldEnv

env = GridWorldEnv()
state = env.reset()
if isinstance(state, tuple):
    state = state[0]  # Unpack if env returns (obs, info)
state_size = np.array(state).size

buffer_size = 10000
batch_size = 64
action_size = env.action_space.n if hasattr(env, 'action_space') and hasattr(env.action_space, 'n') else 4
sync_frequency = 10

gamma = 0.99
learning_rate = 1e-3

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

replay_buffer = deque(maxlen=buffer_size)

online_model = DQN(input_dim=state_size, output_dim=action_size).to(device)
target_model = DQN(input_dim=state_size, output_dim=action_size).to(device)
target_model.load_state_dict(online_model.state_dict())
target_model.eval()
optimizer = optim.Adam(online_model.parameters(), lr=learning_rate)


def act(state, model, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            return q_values.argmax().item()

def sample_minibatch():
    minibatch = random.sample(replay_buffer, batch_size)
    states = torch.FloatTensor([transition[0] for transition in minibatch]).to(device)
    actions = torch.LongTensor([transition[1] for transition in minibatch]).to(device)
    rewards = torch.FloatTensor([transition[2] for transition in minibatch]).to(device)
    next_states = torch.FloatTensor([transition[3] for transition in minibatch]).to(device)
    dones = torch.FloatTensor([transition[4] for transition in minibatch]).to(device)
    return states, actions, rewards, next_states, dones

num_episodes = 500
max_steps = 200

for episode in range(num_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    total_reward = 0
    for t in range(max_steps):
        action = act(state, online_model, epsilon)
        next_state, reward, done, *info = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        replay_buffer.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = sample_minibatch()
            with torch.no_grad():
                next_q_values = target_model(next_states).max(dim=1).values
                target = rewards + (1 - dones) * gamma * next_q_values
            q_values = online_model(states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = nn.MSELoss()(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if done:
            break
    if episode % sync_frequency == 0:
        target_model.load_state_dict(online_model.state_dict())
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

print("Training complete.")