# @Filename:    actor_critic.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/12/22 6:03 AM
from typing import Callable, Dict, Any, Optional, Type, Tuple
import gym
from torch import nn
from stable_baselines3.common.policies import ActorCriticCnnPolicy

class A2CCNNPolicy(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        actor_critic_class: Type[nn.Module],
        features_extractor_class: Type[nn.Module],
        features_extractor_kwargs: Optional[Dict[str, Any]] = dict(),
        *args,
        **kwargs,
    ):
        self.actor_critic_layer = actor_critic_class
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
        self.features_extractor = features_extractor_class(self.observation_space, **features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = self.actor_critic_layer(self.observation_space)