# @Filename:    actor_critic.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/12/22 6:03 AM
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
from torch import nn

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import NatureCNN
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

class CustomNetwork(nn.Module):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        feature_dim = observation_space.shape[0]

        # there is already a cnn in the base class, NatureCNN
        self.cnn = nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # self.linear = nn.Sequential(nn.Linear(n_flatten, n_flatten), nn.ReLU())

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(512, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(512, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(x), self.value_net(x)



class CustomActorCriticPolicy(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        feature_extractor_class: Type[nn.Module] = NatureCNN,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.features_extractor_class = feature_extractor_class

    # def _build(self, lr_schedule: Schedule) -> None:
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.observation_space)

