# @Filename:    model.py.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/9/22 3:37 AM

import gym
import torch
from torch import nn
from attention import MultiheadAttention
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment import StreetFighterEnv
import os

class BiasNet(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, num_heads: int = 8):
        super().__init__(observation_space, features_dim)
        input_size = observation_space.shape[0]
        # print(input_size)
        c, h, w = (4, 84, 84)

        if h != w:
            raise ValueError("Input shape must be square")
        if h != 84:
            raise ValueError("Input shape must be 84x84")

        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
            # relational module
            # MultiheadAttention(64, 64, num_heads),


        )
        with torch.no_grad():
            n_flatten = self.net(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        self.norm1 = nn.LayerNorm(5184)
        self.norm2 = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # print(x.shape)
        x = self.net(x)
        x = x + self.dropout(x)
        x = self.norm1(x)

        linear_out = self.linear(x)
        # print(x.shape, linear_out.shape)
        # x = x + self.dropout(linear_out)
        x = self.norm2(linear_out)
        return x

class CustomCNN(BaseFeaturesExtractor):


    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128)
)
#
# LOG_DIR = './logs/'
# OPT_DIR = './opt/'
# CHECKPOINT_DIR = './train/'
# TRAIN_LOG_DIR = './train_logs/'
# RECORD_DIR = './record/'
#
# for i in [LOG_DIR, OPT_DIR, CHECKPOINT_DIR, TRAIN_LOG_DIR, RECORD_DIR]:
#     # if os.path.exists(i):
#     import shutil
#     shutil.rmtree(i)
#     # os.rmdir(i, ignore_errors=True)
#     os.mkdir(i)
#
# env = StreetFighterEnv(RECORD_DIR)
# env = Monitor(env, LOG_DIR)
# env = DummyVecEnv([lambda: env])
# env = VecFrameStack(env, 4, channels_order='last')
# model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, policy_kwargs=policy_kwargs)
# model.learn(total_timesteps=1000000, log_interval=10)