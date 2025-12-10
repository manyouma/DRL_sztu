import gymnasium as gym
import torch
import numpy as np
import time
from hopper import Actor  
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)

env = gym.make("Hopper-v4", render_mode="human")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

actor = Actor(obs_dim, act_dim, max_action)
actor.load_state_dict(torch.load("ddpg_hopper_actor_class.pth", map_location="cpu"))
actor.eval()

state, _ = env.reset()

while True:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = actor(state_tensor).detach().numpy().flatten()
    state, _, terminated, truncated, _ = env.step(action)
    time.sleep(0.01)
    
    if (terminated or truncated):
        state, _ = env.reset()

    
