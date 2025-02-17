class Config:
    def __init__(self):
        self.state_dim = 24
        self.action_dim = 4
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.episodes = 1000
        self.model_path = "robot_arm_model.pth"
        self.dt = 0.02
        self.initial_state = [0, 0, 0, 0]  # Example initial state, modify as needed