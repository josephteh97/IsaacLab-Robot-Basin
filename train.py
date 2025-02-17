import torch
import torch.optim as optim
from environment import WashbasinCleaningEnv
from model import RLModel
from config import Config

def train_model():
    config = Config()
    env = WashbasinCleaningEnv(config)
    model = RLModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    for episode in range(config.episodes):
        state = env.reset()
        done = False
        while not done:
            action = model.act(state)
            next_state, reward, done, _ = env.step(action)
            target = reward + config.gamma * model.act(next_state)
            loss = criterion(model.act(state), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state

        print(f"Episode {episode} completed.")

    torch.save(model.state_dict(), config.model_path)

if __name__ == "__main__":
    train_model()