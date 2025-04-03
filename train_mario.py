import torch, os
import torch.nn as nn
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from torch.optim import Adam
from utils import *
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

# Make environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb', apply_api_compatibility=True)
# Restrict actions to right only
env = JoypadSpace(env, RIGHT_ONLY)
# Pre-process environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
env = FrameStack(env, num_stack=4)

# define variables for training
n_episodes = 100000
n_save_iter = 2
lr = 25e-5
batch_size = 64
eps = 1
eps_decay = 0.99999975
gamma = 0.9
eps_min = 0.1
replay_buffer_capacity = 100_000
sync_network_rate = 200
load_model = None #'trained_models/model_8_0.9994711398296666.pt'
first_episode = 0

# Choose and print device (mps for using GPU on MacOS, can use cuda if nvidia GPU is available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Instantiate model
online_network = CNN(n_actions = env.action_space.n, 
                         n_frames = env.observation_space.shape[0]).to(device)
target_network = CNN(n_actions = env.action_space.n, 
                         n_frames = env.observation_space.shape[0]).to(device)

if not isinstance(load_model, type(None)):
    online_network.load_state_dict(torch.load(load_model))
    target_network.load_state_dict(online_network.state_dict())
    base_name = os.path.basename(load_model)  # Extracts the file name from the path
    first_episode = int(base_name.split('_')[1]) - 1 # Extracts last episode from file name
    eps = float(base_name.split('_')[2].split('.')[0] + '.' + base_name.split('_')[2].split('.')[1]) # Extracts last epsilon from file name

# Optimiser and loss
optimizer = Adam(online_network.parameters(), lr= lr) # only pass online network parameters to optimizer
loss_fun = nn.MSELoss()

# Replay buffer
storage = LazyMemmapStorage(replay_buffer_capacity)
replay_buffer = TensorDictReplayBuffer(storage=storage)

# Training Loop
for i in range(first_episode, n_episodes):    
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:

        # Populate buffer through epsilon greedy approach
        # Exploration approach
        if np.random.random() < eps:
            a = np.random.randint(env.action_space.n)
        else:
            # Exploitation approach
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            # Grabbing the index of the action that's associated with the highest Q-value
            a = online_network(state).argmax().item()
            state = state.cpu()
        # Take step with chosen action to obtain next state, and reward
        next_state, reward, done, truncated, info  = env.step(a)
        total_reward += reward
        # Store tuple in replay buffer
        replay_buffer.add(TensorDict({"state": torch.tensor(np.array(state), dtype=torch.float32),
                                      "action": torch.tensor(a),"reward": torch.tensor(reward), 
                                      "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                      "done": torch.tensor(done)}, batch_size=[]))
        # Update state in currect episode
        state = next_state

        #Train network
        if len(replay_buffer) > batch_size:
            # Sample replay buffer n tuples, where n is batch_size
            samples = replay_buffer.sample(batch_size).to(device)
            states, actions, rewards, next_states, dones = [samples[key] for key in ("state", "action", "reward", "next_state", "done")]
            # Predict Qs for all state action pairs/ for all samples in batch
            all_pred_Qs = online_network(states)
            # Retrieve Qs for each state action pair
            state_action_Qs = all_pred_Qs[np.arange(batch_size), actions.squeeze()]
            max_next_Qs = target_network(next_states).max(dim=1)[0]
            target_Qs = rewards + gamma * max_next_Qs * (1 - dones.float()) # "Grount Truth" provided by the target network put through action value function
            # Calculate loss and update network weights
            loss = loss_fun(state_action_Qs, target_Qs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Decay epsilon
            eps = max(eps * eps_decay, eps_min)

    print("Episode:", i, 
          "Total reward:", total_reward, 
          "Epsilon:", eps, 
          "Size of replay buffer:", len(replay_buffer))

    # Sync networks
    if i+1 % sync_network_rate == 0:
        target_network.load_state_dict(online_network.state_dict())

    #save model
    if (i+1) % n_save_iter == 0:
        online_network.save_model(out_dir+'/Mario/trained_models/model_' + str(i+1) + '_' + str(eps))
        print('model_' + str(i+1) + '_' + str(eps), 'saved')

env.close()
