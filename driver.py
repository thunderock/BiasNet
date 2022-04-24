# @Filename:    driver.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/18/22 8:41 PM
from constants import *
from utils import plot_study, GameState, record_model_playing
from environment import StreetFighterEnv
from stable_baselines3 import PPO, A2C
from actor_critic import A2CCNNPolicy
from feature_extractors import CNNExtractorWithAttention, CNNExtractor
from tuner import Tuner
import os
from layers import ActorCriticLayer
from torch.multiprocessing import Pool, set_start_method
try:
     set_start_method('spawn')
except RuntimeError as e:
    print(e)



def recorder(model_path, capture_movement, state, model_name, render, record_dir):
    print("model_path: {}, capture_movement: {}, state: {}, model_name: {}, render: {}, record_dir: {}".format(model_path, capture_movement, state, model_name, render, record_dir))
    env = StreetFighterEnv(capture_movement=capture_movement, state=state, training=False, record_file=record_dir)
    model = A2C if model_name == "A2C" else PPO
    model = model.load(model_path)
    record_model_playing(env=env, model=model, render=render)


def _tuner_wrapper(bias, capture_movement, time_steps, model_dir, model_name, trials, state):


    model = A2C if model_name == "A2C" else PPO
    feature_extractor_class = CNNExtractorWithAttention if bias else CNNExtractor
    policy_network = A2CCNNPolicy
    policy_kwargs = dict(features_extractor_class=feature_extractor_class,
                         features_extractor_kwargs=dict(features_dim=512, ), actor_critic_class=ActorCriticLayer)
    tuner = Tuner(model=model, capture_movement=capture_movement, state=state.value, policy_network=policy_network,
                  policy_args=policy_kwargs,
                  timesteps=time_steps, save_dir=os.path.join(model_dir, state.name))
    study, (study_name, study_location) = tuner.tune_study(n_trials=trials, study_name=model_name + "_" + state.name)
    plot_study(study, path=os.path.join(model_dir, state.name))
    print("state: {}, study: {}".format(state.name, study_name))


def tuner(bias, capture_movement, time_steps, model_dir, model_name, trials, n_jobs):
    print("bias: {}, capture_movement: {}, time_steps: {}, model_dir: {}, model_name: {}, trials: {}".format(bias, capture_movement, time_steps, model_dir, model_name, trials))
    assert bias in [True, False] and capture_movement in [True, False] and model_name in ["A2C", "PPO"] and time_steps > 0
    pool = Pool(n_jobs)
    pool.starmap(_tuner_wrapper,
                 [(bias, capture_movement, time_steps, model_dir, model_name, trials, state) for state in GameState])


def _train_wrapper(bias, capture_movement, model_params, time_steps, model_dir, model_name, state):
    model = A2C if model_name == "A2C" else PPO
    feature_extractor_class = CNNExtractorWithAttention if bias else CNNExtractor
    policy_network = A2CCNNPolicy
    policy_kwargs = dict(features_extractor_class=feature_extractor_class,
                         features_extractor_kwargs=dict(features_dim=512, ), actor_critic_class=ActorCriticLayer)
    env = StreetFighterEnv(capture_movement=capture_movement, state=state.value, training=True)
    tuner = Tuner(model=model, capture_movement=capture_movement, state=state, policy_network=policy_network,
                  policy_args=policy_kwargs, timesteps=time_steps, save_dir=os.path.join(model_dir, state.name))
    reward, model = tuner._evaluate_model(env, model_params, 0, return_model=True, save_model=False)
    model.save(os.path.join(model_dir, model_name + "_" + state.name))
    print("state: {}, reward: {}".format(state.name, reward))


def trainer(bias, capture_movement, model_params, time_steps, model_dir, model_name, n_jobs):
    print("bias: {}, capture_movement: {}, model_params: {}, time_steps: {}, model_dir: {}, model_name: {}".format(bias, capture_movement, model_params, time_steps, model_dir, model_name))
    assert bias in [True, False] and capture_movement in [True, False] and isinstance(model_params, dict) and model_name in ["A2C", "PPO"] and time_steps > 0

    pool = Pool(processes=n_jobs)
    pool.starmap(_train_wrapper, [(bias, capture_movement, model_params, time_steps, model_dir, model_name, state) for state in GameState])



