# @Filename:    tuner.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/16/22 11:10 PM


from botorch.settings import suppress_botorch_warnings, validate_input_scaling
import shutil
from constants import *
from utils import evaluate_model_policy, plot_study, plot_fig, load_study
from trainer import get_trained_model
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import os


suppress_botorch_warnings(False)
validate_input_scaling(True)

class Tuner(object):

    def __init__(self, model, env, policy_network, policy_args, timesteps=1000000, save_dir='/tmp/models'):
        self.model = model
        self.env = env
        self.policy_network = policy_network
        self.policy_args = policy_args
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

    def _tune_model(self, trial_params):
        model_params = self._get_trial_values(trial_params)
        return self._evaluate_model(model_params, trial_params.number)

    def _evaluate_model(self, model_params, trial_number):
        model = get_trained_model(
            env=self.env, policy_network=self.policy_network, feature_extractor_kwargs=self.policy_args,
            model=self.model, timesteps=self.timesteps, model_params=model_params)
        reward = evaluate_model_policy(self.env, model)
        model.save(self.get_model_path(trial_number))
        return reward

    def get_model(self, study):
        best_iteration = study.best_trial.number
        best_model_path = self.get_model_path(best_iteration)
        return self.model.load(best_model_path)
    
    def tune_study(self, n_trials=1, study_name='study', study_dir=None):
        wandb_kwargs = {"project": study_name}
        wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
        if not study_dir:
            study_dir = "sqlite:///{}/example.db".format(self.save_dir)
        sampler = optuna.integration.BoTorchSampler()
        study = optuna.create_study(study_name=study_name, storage=study_dir, direction='maximize', sampler=sampler)
        study.optimize(lambda trial: self._tune_model(trial), n_trials=n_trials, n_jobs=1, show_progress_bar=True, callbacks=[wandbc])
        self.env.close()
        return study, (study_name, study_dir)

# TIMESTEPS = 2
# N_TRIALS = 2
# # FRAME_SIZE = 4


# from environment import StreetFighterEnv
# from stable_baselines3 import A2C
# from actor_critic import A2CCNNPolicy
# from feature_extractors import CNNExtractorWithAttention, CNNExtractor
# from layers import ActorCriticLayer
# ########################################################################################################################
# for extractor in [CNNExtractorWithAttention]:
#     model = A2C
#     model_dir = 'models/bias'
#     env = StreetFighterEnv(capture_movement=False)
#     policy_network = A2CCNNPolicy
#
#     policy_kwargs = dict(
#         features_extractor_class=CNNExtractorWithAttention,
#         features_extractor_kwargs=dict(features_dim=512, ),
#         actor_critic_class=ActorCriticLayer
#     )
#     tuner = Tuner(model=model, env=env, policy_network=policy_network, policy_args=policy_kwargs,
#                   timesteps=TIMESTEPS, save_dir=model_dir)
#
#     study, (study_name, study_location) = tuner.tune_study(n_trials=N_TRIALS, study_name="study")
#
#     loaded_study = load_study(study_name, study_location)
#
#     print(study.best_trial.number, study.best_params)
#     assert len(loaded_study.trials) == len(study.trials)