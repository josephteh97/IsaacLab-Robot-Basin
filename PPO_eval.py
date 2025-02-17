import torch
from environment import WashbasinCleaningEnv
from model import RLModel
from config import Config

def evaluate_model():
    config = Config()
    env = WashbasinCleaningEnv(config)
    model = RLModel(config)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    state = env.reset()
    done = False
    while not done:
        action = model.act(state)
        state, reward, done, _ = env.step(action)
        env.render()

if __name__ == "__main__":
    evaluate_model()