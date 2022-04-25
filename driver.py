# @Filename:    driver.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/18/22 8:41 PM
from constants import *
from utils import plot_study, GameState, record_model_playing
from environment import StreetFighterEnv
from trainer import _get_model
from callbacks import get_eval_callback
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


def fine_tune(model_name, model_path, tensorboard_path, model_params, state, time_steps, bias, capture_movement, model_save_path):
    print("model_name: {}, model_path: {}, tensorboard_path: {}, model_params: {}, state: {}, time_steps: {}, bias: {}, capture_movement: {}, model_save_path: {}".format(model_name, model_path, tensorboard_path, model_params, state, time_steps, bias, capture_movement, model_save_path))
    assert model_name in ["A2C", "PPO"]
    model = A2C if model_name == "A2C" else PPO
    feature_extractor_class = CNNExtractorWithAttention if bias else CNNExtractor
    policy_network = A2CCNNPolicy
    policy_kwargs = dict(features_extractor_class=feature_extractor_class,
                         features_extractor_kwargs=dict(features_dim=512, ), actor_critic_class=ActorCriticLayer)
    env = StreetFighterEnv(capture_movement=capture_movement, state=state, training=True)
    callback = get_eval_callback(env, model_save_path)
    model = _get_model(model_type=model, env=env, policy_network=policy_network, feature_extractor_kwargs=policy_kwargs, log_dir=tensorboard_path, verbose=1, model_params=model_params)
    model.load(model_path)
    # print(env)
    model.learn(total_timesteps=time_steps, callback=callback)


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
    tuner = Tuner(model=model, capture_movement=capture_movement, state=state, policy_network=policy_network,
                  policy_args=policy_kwargs,
                  timesteps=time_steps, save_dir=os.path.join(model_dir, state))
    study, (study_name, study_location) = tuner.tune_study(n_trials=trials, study_name=model_name + "_" + state)
    plot_study(study, path=os.path.join(model_dir, state))
    print("state: {}, study: {}".format(state, study_name))


def _train_wrapper(bias, capture_movement, model_params, time_steps, model_dir, model_name, state):
    model = A2C if model_name == "A2C" else PPO
    feature_extractor_class = CNNExtractorWithAttention if bias else CNNExtractor
    policy_network = A2CCNNPolicy
    policy_kwargs = dict(features_extractor_class=feature_extractor_class,
                         features_extractor_kwargs=dict(features_dim=512, ), actor_critic_class=ActorCriticLayer)
    env = StreetFighterEnv(capture_movement=capture_movement, state=state, training=True)
    tuner = Tuner(model=model, capture_movement=capture_movement, state=state, policy_network=policy_network,
                  policy_args=policy_kwargs, timesteps=time_steps, save_dir=os.path.join(model_dir, state))
    callback = get_eval_callback(env, 'models/{}/'.format(state), 15000)
    reward, model = tuner._evaluate_model(env, model_params, 0, return_model=True, save_model=False, callbacks=callback)
    model.save(os.path.join(model_dir, model_name + "_" + state))
    print("state: {}, reward: {}".format(state, reward))


def tuner(bias, capture_movement, time_steps, model_dir, model_name, trials, n_jobs, states):
    print("bias: {}, capture_movement: {}, time_steps: {}, model_dir: {}, model_name: {}, trials: {}".format(bias, capture_movement, time_steps, model_dir, model_name, trials))
    assert bias in [True, False] and capture_movement in [True, False] and model_name in ["A2C", "PPO"] and time_steps > 0
    if isinstance(states, str):
        states = [states]
    pool = Pool(n_jobs)
    pool.starmap(_tuner_wrapper,
                 [(bias, capture_movement, time_steps, model_dir, model_name, trials, state) for state in states])


def trainer(bias, capture_movement, model_params, time_steps, model_dir, model_name, n_jobs, states=None):
    print("bias: {}, capture_movement: {}, model_params: {}, time_steps: {}, model_dir: {}, model_name: {}".format(bias, capture_movement, model_params, time_steps, model_dir, model_name))
    assert bias in [True, False] and capture_movement in [True, False] and isinstance(model_params, dict) and model_name in ["A2C", "PPO"] and time_steps > 0

    if isinstance(states, str):
        states = [states]

    pool = Pool(processes=n_jobs)
    pool.starmap(_train_wrapper, [(bias, capture_movement, model_params, time_steps, model_dir, model_name, state) for state in states])



