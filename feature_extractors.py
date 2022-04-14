# @Filename:    feature_extractors.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/14/22 4:49 AM

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
from torch import nn

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from environment import StreetFighterEnv
import os
import shutil
import time

from environment import StreetFighterEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np

class CustomFeatureExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
        features_dim: int = 512
    ):
        super(CustomFeatureExtractor, self).__init__(observation_space=observation_space, features_dim=features_dim)


        # there is already a cnn in the base class, NatureCNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (8,8), stride=(4,4), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.linear(self.cnn(x))


