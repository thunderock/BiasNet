# @Filename:    tuner.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/16/22 11:10 PM

from constants import *
from utils import evaluate_model_policy
from trainer import get_trained_model
import optuna
from environment import StreetFighterEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from actor_critic import A2CCNNPolicy
from feature_extractors import CNNExtractorWithAttention, CNNExtractor
from constants import *


class Tuner(object):
    def __init__(self, model, env, policy_network, policy_args, frame_size=1, timesteps=100000):
        self.model = model
        self.env = env
        self.policy_network = policy_network
        self.policy_args = policy_args
        self.frame_size = frame_size
        self.timesteps = timesteps
        self.study = optuna.create_study(direction='maximize')

    @staticmethod
    def _get_trial_values(trial):
        return {
            'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
            'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        }

    def tune(self, n_trials=1):
        self.study.optimize(lambda trial: evaluate_model_policy(self.env, get_trained_model(
            env=self.env, policy_network=self.policy_network, feature_extractor_kwargs=self.policy_args,
            model=self.model, timesteps=self.timesteps, frame_size=self.frame_size, model_params=self._get_trial_values(trial))), n_trials=n_trials, n_jobs=1)

env = StreetFighterEnv()
model = A2C
policy_network = A2CCNNPolicy
frame_size = 1
timesteps = 10
policy_kwargs = dict(
    features_extractor_class=CNNExtractorWithAttention,
    # features_extractor_kwargs=dict(observation_space=gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)),
)

tuner = Tuner(model=model, env=env, policy_network=policy_network, policy_args=policy_kwargs, frame_size=frame_size)

tuner.tune(n_trials=2)
