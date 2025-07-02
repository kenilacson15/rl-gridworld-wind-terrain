import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from envs.gridworld import GridWorldEnv



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

env = GridWorldEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
buffer_size = 10000
batch_size = 64
sync_frequency = 5
gamma = 0.99
learning_rate = 1e-3
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
reward_step_penalty = -1.0  # Encourage shorter paths

action_size = env.action_space.n if hasattr(env, 'action_space') and hasattr(env.action_space, 'n') else 4

# Dueling DQN architecture
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

def act(state, model, epsilon):
    state = unwrap_state(state)
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    else:
        with torch.no_grad():

            if isinstance(state, tuple):
                state = state[0]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            q_values = model(state_tensor)
            return q_values.argmax().item()
        

def unwrap_state(state):
    while isinstance(staple, tuple):
        state = state[0]
    return state


def sample_minibatch(replay_buffer):
    minibatch = random.sample(replay_buffer, batch_size)
    states = torch.FloatTensor([transition[0] for transition in minibatch]).to(device)
    actions = torch.LongTensor([transition[1] for transition in minibatch]).to(device)
    rewards = torch.FloatTensor([transition[2] for transition in minibatch]).to(device)
    next_states = torch.FloatTensor([transition[3] for transition in minibatch]).to(device)
    dones = torch.FloatTensor([transition[4] for transition in minibatch]).to(device)
    return states, actions, rewards, next_states, dones

# Initialize
state = env.reset()
state = unwrap_state(state)
state_size = np.array(state).size





def build_models(input_dim, output_dim):
    online = DuelingDQN(input_dim, output_dim).to(device)
    target = DuelingDQN(input_dim, output_dim).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    return online, target





online_model = DuelingDQN(input_dim=state_size, output_dim=action_size).to(device)
target_model = DuelingDQN(input_dim=state_size, output_dim=action_size).to(device)
target_model.load_state_dict(online_model.state_dict())
target_model.eval()
optimizer = optim.Adam(online_model.parameters(), lr=learning_rate)
replay_buffer = deque(maxlen=buffer_size)

num_episodes = 500
max_steps = 200

for episode in range(num_episodes):
    state = env.reset()
    state = unwrap_state(state)

    total_reward = 0
    for t in range(max_steps):
        action = act(state, online_model, epsilon)
        result = env.step(action)
        next_state, reward, done, = result[:3]
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        # Reward shaping: add step penalty
        shaped_reward = reward + reward_step_penalty
        replay_buffer.append((state, action, shaped_reward, next_state, float(done)))
        state = next_state
        total_reward += shaped_reward
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = sample_minibatch(replay_buffer)
            # Double DQN target
            with torch.no_grad():
                next_actions = online_model(next_states).argmax(1)
                next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = rewards + (1 - dones) * gamma * next_q
            q_values = online_model(states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = nn.SmoothL1Loss()(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            torch.nn.utils.clip_grad_norm_(online_model.parameters(), 1.0)





        if done:
            break
    if episode % sync_frequency == 0:
        target_model.load_state_dict(online_model.state_dict())
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if episode % 10 == 0:
        print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

print("Training complete.")

DQN = DuelingDQN  # For backward compatibility with main.py

__all__ = ["DQN", "act"]