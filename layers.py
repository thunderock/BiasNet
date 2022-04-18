# @Filename:    layers.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/18/22 2:17 AM
from constants import *
import torch
import torch.nn as nn
from typing import Callable, Dict, Any, Optional, Type, Tuple
import gym

class ActorCriticLayer(nn.Module):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(ActorCriticLayer, self).__init__()

        # these two variables are required by baselines
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # feature_dim = observation_space.shape[0]

        self.pi_network = nn.Sequential(
            nn.Linear(512, last_layer_dim_pi), nn.ReLU()
        ).to(DEVICE)

        self.vf_network = nn.Sequential(
            nn.Linear(512, last_layer_dim_vf), nn.ReLU()
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pi_network(x), self.vf_network(x)

    # functions required by baselines
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.pi_network(features)

    # functions required by baselines
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.vf_network(features)