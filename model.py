import torch
import torch.nn as nn
import torch.optim as optim

class RLModel(nn.Module):
    def __init__(self, config):
        super(RLModel, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(config.state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, config.action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = self.forward(state)
        return action.numpy()