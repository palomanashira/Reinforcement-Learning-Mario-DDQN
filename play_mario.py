import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from utils import *
import time

# Make environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human', apply_api_compatibility=True)
# Restrict actions to right only
env = JoypadSpace(env, RIGHT_ONLY)
# Pre-process environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
env = FrameStack(env, num_stack=4)

# define variables for training
n_episodes = 100000
load_model = "/Users/nashirarodriguez/Library/Mobile Documents/com~apple~CloudDocs/Mario/model_100000_0.1.tar"
first_episode = 0

device = torch.device('cpu')
print(device)

# Instantiate model
online_network = CNN(n_actions = env.action_space.n, 
                         n_frames = env.observation_space.shape[0]).to(device)

online_network.load_state_dict(torch.load(load_model, map_location=torch.device('cpu'))['model_state_dict'])

# Training Loop
for i in range(first_episode, n_episodes):    
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
        a = online_network(state).argmax().item()
        next_state, reward, done, truncated, info  = env.step(a)
        time.sleep(0.05)
        total_reward += reward
        state = next_state

env.close()
