# @Filename:    utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/16/22 8:50 PM
import os
from constants import *
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import time
from enum import Enum
from matplotlib import pyplot as plt


class GameState(Enum):
    GUILE = STATE_GUILE
    ZANGIEF = STATE_ZANGIEF
    DAHLISM = STATE_DAHLISM
    EHDONA = STATE_EHDONA
    CHUNLI = STATE_CHUNLI
    BLANKA = STATE_BLANKA
    KEN = STATE_KEN
    RYU = STATE_RYU


def record_model_playing(env, model, render=False):
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

            # if reward != 0: print(reward)
            total_reward += reward
    print("iterations: ", iteration)
    print("total reward: ", total_reward)
    env.close()
    return True


def load_study(study_name, path):
    if 'sqlite' not in path:
        path = os.path.join('sqlite:///', path)
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


def plot_study(study, path=None):
    plots = [optuna.visualization.matplotlib.plot_parallel_coordinate,
            optuna.visualization.matplotlib.plot_contour,
            optuna.visualization.matplotlib.plot_slice,
            optuna.visualization.matplotlib.plot_param_importances,
            optuna.visualization.matplotlib.plot_edf,
            optuna.visualization.matplotlib.plot_optimization_history]
    for plot in plots:
        _ = plot(study)
        if path is None:
            plt.show()
        else:
            p = os.path.join(path, plot.__name__ + '.png')
            print("writing fig at ", str(p))
            try:
                plt.savefig(p)
            except ValueError:
                pass
    return