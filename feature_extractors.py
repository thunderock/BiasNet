# @Filename:    feature_extractors.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/14/22 4:49 AM

import gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class CNNExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
        features_dim: int = 512,
    ):
        super(CNNExtractor, self).__init__(observation_space=observation_space, features_dim=features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (8,8), stride=(4,4), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.linear(x)
        # print(x.shape)
        return x


class CNNExtractorWithAttention(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
        features_dim: int = 512,
    ):
        super(CNNExtractorWithAttention, self).__init__(observation_space=observation_space, features_dim=features_dim)
        # there is already a cnn in the base class, NatureCNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (8,8), stride=(4, 4), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=(1,1), stride=(1, 1)),
            nn.ReLU(),
        )
        with torch.no_grad():
            features_out = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape
            features_out = features_out[1]

        self.self_attention = nn.MultiheadAttention(features_out, num_heads=4)
        self.max_pool = nn.Sequential(nn.MaxPool2d(kernel_size=(4, 4)), nn.Flatten(), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(36, 64), nn.ReLU(), nn.Linear(64, features_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, *_ = x.shape
        x = self.cnn(x)
        y = x.reshape(x.size(2)*x.size(0), x.size(2), x.size(1))
        y, _ = self.self_attention(y, y, y)
        y = self.max_pool(y)
        y = y.reshape(batch_size, -1)

        x = self.linear(y)
        return x


