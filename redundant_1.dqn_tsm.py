#!/usr/bin/env python
# coding: utf-8

#todo MakeNote1 -> I stacked 3 frames and reshaped the PyTorch wrapper to have a shape (3,1,108,108)
#todo MakeNote2 -> The state and env.observation.shape don't have the same shape


# In[1]:


import math, random
import cv2
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F


# <h3>Use Cuda</h3>

# In[3]:


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


# <h2>Replay Buffer</h2>

# In[4]:


from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)


# <h2>Computing Temporal Difference Loss</h2>

# In[19]:


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = state.reshape(batch_size*T,1,state.shape[-2],state.shape[-1])
    next_state = next_state.reshape(batch_size*T, 1, state.shape[-2],state.shape[-1])
    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))
    q_values      = model(state)
    next_q_values = model(next_state)
    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss


# In[20]:


# In[22]:


from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch


# In[23]:


env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)


# In[24]:

T = 3
class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(input_shape[1], 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()

        
        self.fc = nn.Sequential(
            nn.Linear(19200, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))

        n,c,h,w = x.shape
        x = x.reshape(n//T, T, c, h, w)
        copy = torch.clone(x)
        x[:,:, :c//8, :, :] = torch.roll(x[:,:, :c//8, :, :], shifts = 1, dims = 1)
        x[:,0, :c//8, :, :] = copy[:,0, :c//8, :, :]
        x = x.reshape(n, c, h, w)

        x = self.relu2(self.conv2(x))
        n, c, h, w = x.shape
        x = x.reshape(n//T, T, c, h, w)
        copy = torch.clone(x)
        x[:, :, :c // 8, :, :] = torch.roll(x[:, :, :c // 8, :, :], shifts=1, dims=1)
        x[:, 0, :c // 8, :, :] = copy[:, 0, :c // 8, :, :]
        x = x.reshape(n,c,h,w)

        x = self.relu3(self.conv3(x))
        n, c, h, w = x.shape
        x = x.reshape(n//T, T, c, h, w)
        copy = torch.clone(x)
        x[:, :, :c // 8, :, :] = torch.roll(x[:, :, :c // 8, :, :], shifts=1, dims=1)
        x[:, 0, :c // 8, :, :] = copy[:, 0, :c // 8, :, :]

        x = x.view(x.size(0), -1)

        # todo MakeNote3 -> Changed the shape in x.view to get back 32 states
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action
model = CnnDQN(env.observation_space.shape, env.action_space.n)
print("Input Shape 0 = ", env.observation_space.shape)
if USE_CUDA:
    model = model.cuda()
    
optimizer = optim.Adam(model.parameters(), lr=0.00001)

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)


# In[26]:


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


# In[27]:


plt.plot([epsilon_by_frame(i) for i in range(1000000)])


# In[33]:


num_frames = 1400000
batch_size = 32
gamma      = 0.99

losses = []
reward_step = np.empty(shape = num_frames)

all_rewards = []
episode_reward = 0

state = np.expand_dims(env.reset(), axis = 1)
for frame_idx in range(1, num_frames + 1):
    print("Frame = ", frame_idx)
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    next_state = np.expand_dims(next_state, axis = 1)


    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    reward_step[frame_idx - 1] = reward
    
    if done:
        state = np.expand_dims(env.reset(), axis = 1)
        all_rewards.append(episode_reward)
        np.savetxt('rad.out', all_rewards, delimiter=',')
        episode_reward = 0
        
    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(batch_size)
        losses.append(loss.data)
        
    if frame_idx % 100000 == 0:
        print("Frame Index = ", frame_idx)
        np.savetxt('rad_step_reward.out', reward_step, delimiter=',')


# In[ ]:




