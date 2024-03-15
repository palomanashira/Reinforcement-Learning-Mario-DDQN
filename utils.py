import torch.nn as nn
from gym import Wrapper

# Define network to train mario
class CNN(nn.Module):
    def __init__(self, n_actions, n_frames = 5):
        super(CNN, self).__init__()

        # Convolutional layers
        self.network = nn.Sequential(
            nn.Conv2d(in_channels = n_frames, out_channels = 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features = 3136, out_features = 512),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = n_actions)
        )

    def forward(self, x):

        x = x.squeeze(4)
        x = self.network(x)

        return x # predicted Q values for each action
    
# Overload step function
class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return state, total_reward, done, trunc, info
