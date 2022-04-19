# @Filename:    driver.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/18/22 8:41 PM
from constants import *
from utils import evaluate_model_policy, plot_study, plot_fig
from trainer import get_trained_model
import optuna
from environment import StreetFighterEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from actor_critic import A2CCNNPolicy
from feature_extractors import CNNExtractorWithAttention, CNNExtractor
from tuner import Tuner
import os
from layers import ActorCriticLayer


TIMESTEPS = 50000000
N_TRIALS = 1
PLOTLY_CONFIG = {"staticPlot": True}

model = A2C
model_dir = 'models/bias'
env = StreetFighterEnv(capture_movement=False, training=True)
policy_network = A2CCNNPolicy

policy_kwargs = dict(
    features_extractor_class=CNNExtractor,
    features_extractor_kwargs=dict(features_dim=512,),
    actor_critic_class=ActorCriticLayer
)
tuner = Tuner(model=model, env=env, policy_network=policy_network, policy_args=policy_kwargs,
              timesteps=TIMESTEPS, save_dir=model_dir)

model_params = {'gamma': 0.9021710921259072, 'learning_rate': 2.4218633452543628e-04, 'gae_lambda': 0.8689432986110721}
print(tuner._evaluate_model(model_params, 0))