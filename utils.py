# @Filename:    utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/16/22 8:50 PM

from constants import *
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
import optuna
import plotly
from environment import StreetFighterEnv
import time
from enum import Enum


class GameState(Enum):
    GUILE = STATE_GUILE
    ZANGIEF = STATE_ZANGIEF
    DAHLISM = STATE_DAHLISM
    EHDONA = STATE_EHDONA
    CHUNLI = STATE_CHUNLI
    BLANKA = STATE_BLANKA
    KEN = STATE_KEN
    RYU = STATE_RYU


def record_model_playing(model_path, record_path, capture_movement, render=False):
    env = StreetFighterEnv(record_file=record_path, capture_movement=capture_movement, training=False)
    model = A2C.load(model_path)
    obs = env.reset()
    iteration = 0
    done = False
    total_reward = 0
    for game in range(1):
        while not done:
            iteration += 1
            if done:
                obs = env.reset()
            if render:
                env.render()
                time.sleep(0.01)
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            if reward != 0: print(reward)
            total_reward += reward
    print("iterations: ", iteration)
    print("total reward: ", total_reward)
    return True


def load_study(study_name, path):
    return optuna.load_study(study_name, path)


def evaluate_model_policy(env, model, n_eval_episodes=5):
    """
    Evaluate a policy

    :param env: (Gym Environment) The environment to evaluate the policy on
    :param model: (BaseRLModel object) the policy, whose type depends on the environment.
    :param n_eval_episodes: (int) number of episodes to evaluate the policy
    :return: (float) Mean reward for the `n_eval_episodes` episodes
    """
    score = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)[0]
    return score

def plot_fig(fig):
    plotly_config = {"staticPlot": True}
    plotly.offline.plot(fig, config=plotly_config)


def plot_study(study):
    # only to be used with jupyter notebook
    return [optuna.visualization.plot_parallel_coordinate(study),
            optuna.visualization.plot_contour(study),
            optuna.visualization.plot_slice(study),
            # optuna.visualization.plot_param_importances(study),
            optuna.visualization.plot_edf(study),
            optuna.visualization.plot_optimization_history(study)]