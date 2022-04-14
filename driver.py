# @Filename:    driver.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/14/22 1:43 AM
from environment import StreetFighterEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from actor_critic import CustomActorCriticPolicy
from feature_extractors import CustomFeatureExtractor
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
import gym
import numpy as np

policy_kwargs = dict(
    features_extractor_class=CustomFeatureExtractor,
    # features_extractor_kwargs=dict(observation_space=gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)),
)

policy = CustomActorCriticPolicy
env = StreetFighterEnv()
env = Monitor(env, "\tmp\monitor", allow_early_resets=True)
env = DummyVecEnv([lambda: env])

model = A2C(policy, env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(5000)