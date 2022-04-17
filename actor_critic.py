# @Filename:    actor_critic.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/12/22 6:03 AM
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
from torch import nn

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import NatureCNN



class A2CCNNPolicy(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        feature_extractor_class: Type[nn.Module] = NatureCNN,
        *args,
        **kwargs,
    ):

        super(A2CCNNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.features_extractor_class = feature_extractor_class



