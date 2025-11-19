import torch
from torchrl.envs import GymEnv, TransformedEnv, Compose
from torchrl.envs.transforms import ToTensorImage, GrayScale, Resize, CatFrames, DoubleToFloat, RewardClipping
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from torchrl.objectives import DQNLoss
from torchrl.modules import QValueModule 
from torchrl.envs.transforms import ObservationNorm


writer = SummaryWriter(log_dir=f"runs/pong_dqn_{time.strftime('%Y%m%d-%H%M%S')}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 10_000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10_000
EPSILON_DECAY_LAST_FRAME = 150_000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


base_env = GymEnv("ALE/Pong-v5", from_pixels=True, pixels_only=True, render_mode="rgb_array")
n_actions = base_env.action_space.n
input_shape = base_env.observation_space.shape
obs_shape = (4, 84, 84)
from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.transforms import (
    ToTensorImage,
    GrayScale,
    Resize,
    CenterCrop,
    CatFrames,
    RewardClipping,
    FrameSkipTransform,
    TimeMaxPool,
)

env = TransformedEnv(
    base_env,
    Compose(
        FrameSkipTransform(frame_skip=4),
        TimeMaxPool(in_keys=["pixels"], T=2),
        ToTensorImage(from_int=True, in_keys=["pixels"]), 
        GrayScale(in_keys=["pixels"]),
        Resize(84, 110, in_keys=["pixels"]),
        CenterCrop(84, 84, in_keys=["pixels"]),
        CatFrames(N=4, dim=-3, in_keys=["pixels"]),
        RewardClipping(-1, 1),
    ),
)

rb = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(max_size=REPLAY_SIZE),  
    sampler=RandomSampler(),                         
    batch_size=BATCH_SIZE,                         
)


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()

        self.n_actions = n_actions
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1conv = nn.Linear(3136, 512)
        self.fc2conv = nn.Linear(512, self.n_actions)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1conv(x))
        x = self.fc2conv(x)
        return x
    

q = DQN(n_actions).to(device)
q_target = DQN(n_actions).to(device)
q_target.load_state_dict(q.state_dict())
q_target.eval()
optimizer = optim.Adam(q.parameters(), lr=LEARNING_RATE)


frame_idx = 0
td = env.reset()
total_rewards = []
episode = 0
opt_count = 0
next_obs = td.get("pixels").float() 


while True:
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx/EPSILON_DECAY_LAST_FRAME)
    
    obs = next_obs
    if torch.rand(1).item() < epsilon:
        a = env.action_spec.rand()
    else: 
        with torch.no_grad():
            x = obs.unsqueeze(0).to(device)   
            qvals = q(x)                             
            out = torch.argmax(qvals, dim=1).to("cpu")  
            out = out.squeeze(0)      
            a = torch.zeros(6)
            a[out] = 1.0

    td = env.step(td.set("action", a))
    next_obs = td.get(("next", "pixels")).float()
    r = td.get(("next", "reward"))
    d = td.get(("next", "done"))

    transition = TensorDict(
        {
            "obs": obs,
            "action": a,
            "reward": r,
            "next_obs": next_obs,
            "done": d,
        },
        batch_size=[],
    )
    rb.add(transition)
    total_rewards.append(r)
   

    if d.item():
        td = env.reset()
        next_obs = td.get("pixels").float() 
        episode += 1
        m_reward = np.sum(total_rewards)
        print(f"{frame_idx}: done {episode} games, reward: {m_reward: .3f}, rb: {len(rb)}, eps: {epsilon}")
        writer.add_scalar("Reward/episode", m_reward, episode)      
        writer.add_scalar("Epsilon", epsilon, episode)
        total_rewards=[]

 
    if len(rb) >= REPLAY_START_SIZE:
        print("Starting training...")
        
        batch = rb.sample(BATCH_SIZE)
        obs_b      = batch["obs"].to(device)
        act_b      = batch["action"].long().to(device)   
        rew_b      = batch["reward"].to(device).squeeze(-1)  
        next_obs_b = batch["next_obs"].to(device)
        done_b     = batch["done"].to(device).float().squeeze(-1)
        
        with torch.no_grad():
            opt_act =  q(next_obs_b).argmax(1)
            q_next = q_target(next_obs_b).gather(1, opt_act.unsqueeze(-1)).squeeze(1)
            target = rew_b + gamma * (1.0 - done_b) * q_next
            
        act_b_ind = act_b.argmax(dim=-1)
        q_values = q(obs_b).gather(1, act_b_ind.unsqueeze(-1)).squeeze(1)
        
        loss = F.mse_loss(q_values, target) 
        writer.add_scalar("Loss/frame_idx", loss.item(), frame_idx)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            q_target.load_state_dict(q.state_dict())
            print("Q Network updated")
            print(opt_act[:10])
        
    
    
    