import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

env = GridWorldEnv()
state_size = env.observation_space.shape[0]



buffer_size = 1000
batch_size = 32
action_size = 4
sync_frequency = 10
epsilon_min = 0.01
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)

        )


    def forward(self, x):
        return self.net(x)



replay_buffer = deque(maxlen=buffer_size)



def act(state, model, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    else:
        with torch.no_grad():
            q_values = model(torch.FloatTensor(state).unsqueeze(0))
            return q_values.argmax().item()
        
def sample_minibatch():
    minibatch = random.sample(replay_buffer, batch_size)
    states = torch.FloatTensor([transition[0] for transition in minibatch])
    actions = torch.LongTensor([transition[1] for transition in minibatch])
    rewards = torch.FloatTensor([transition[2] for transition in minibatch])
    next_states = torch.FloatTensor([transition[3] for transition in minibatch])
    dones = torch.FloatTensor([transition[4] for transition in minibatch])
    return states, actions, rewards, next_states, dones


with torch.no_grad():
    next_q_values = target_model(next_states).max(dim=1).values
target = rewards + (1 - dones) * gamma * next_q_values


online_model = DQN(input_dim=state_size, output_dim=action_size)
target_model = DQN(input_dim=state_size, output_dim=action_size)
target_model.load_state_dict(online_model.state_dict())
target_model.eval()





q_values = online_model(states)
q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
loss = nn.MSELoss()(q_value, target)


optimizer = optim.Adam(online_model.parameters(), lr=learning_rate)


if episode % sync_frequency == 0:
    target_model.load_state_dict(online_model.state_dict())


epsilon = max(epsilon_min, epsilon * epsilon_decay)

replay_buffer.append((state, action, reward, next_state, done))