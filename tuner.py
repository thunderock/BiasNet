# @Filename:    tuner.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/16/22 11:10 PM

from constants import *
from utils import evaluate_model_policy
from trainer import get_trained_model
import optuna
import os
from botorch.settings import suppress_botorch_warnings, validate_input_scaling
import shutil

suppress_botorch_warnings(False)
validate_input_scaling(True)

class Tuner(object):
    def __init__(self, model, env, policy_network, policy_args, frame_size=1, timesteps=100000, save_dir='/tmp'):
        self.model = model
        self.env = env
        self.policy_network = policy_network
        self.policy_args = policy_args
        self.frame_size = frame_size
        self.timesteps = timesteps
        self.save_dir = save_dir
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    @staticmethod
    def _get_trial_values(trial):
        return {
            'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
            'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        }


    def get_model_path(self, trial_number):
        return os.path.join(self.save_dir, 'trial_{}_best_model'.format(trial_number))



    def __tune_model(self, trial_params):
        model_params = self._get_trial_values(trial_params)

        model = get_trained_model(
            env=self.env, policy_network=self.policy_network, feature_extractor_kwargs=self.policy_args,
            model=self.model, timesteps=self.timesteps, frame_size=self.frame_size, model_params=model_params)
        reward = evaluate_model_policy(self.env, model)
        # print("Total Reward: {} for params: {}".format(reward, model_params))
        model.save(self.get_model_path(trial_params.number))
        return reward

    def get_model(self, study):
        best_iteration = study.best_trial.number
        best_model_path = self.get_model_path(best_iteration)
        return self.model.load(best_model_path)
    
    
    def tune_study(self, n_trials=1):
        sampler = optuna.integration.BoTorchSampler()
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(lambda trial: self.__tune_model(trial), n_trials=n_trials, n_jobs=1, show_progress_bar=True)
        self.env.close()
        return study


# model = A2C
# env = StreetFighterEnv(record_file='models/{}'.format(model.__module__))
#
# policy_network = A2CCNNPolicy
# frame_size = 1
# timesteps = 1000000
# policy_kwargs = dict(
#     features_extractor_class=CNNExtractorWithAttention
# )
#
# tuner = Tuner(model=model, env=env, policy_network=policy_network, policy_args=policy_kwargs, frame_size=frame_size, timesteps=timesteps, save_dir='models')
#
# best_model = tuner.tune(n_trials=20, )
