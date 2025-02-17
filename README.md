# Robot Arm Washbasin Cleaning with Reinforcement Learning

This repository contains code to train a robot arm to clean a washbasin using Reinforcement Learning (RL) with NVIDIA's Isaac Gym as the simulation environment.

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) installed

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/josephteh97/robot-arm-washbasin-cleaning.git
   cd robot-arm-washbasin-cleaning
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Configure the training parameters in `config.py`.

2. Run the training script:
   ```bash
   python train.py
   ```

3. To evaluate the trained model, run:
   ```bash
   python main.py
   ```

## Files

- `main.py`: The main script to run the trained model.
- `environment.py`: Defines the custom environment for the robot arm in Isaac Gym.
- `model.py`: Defines the neural network model for the RL agent.
- `train.py`: The training script for the RL agent.
- `config.py`: Configuration file for training parameters.
- `requirements.txt`: List of required Python packages.

## License

This project is licensed under the MIT License.
