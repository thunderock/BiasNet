# @Filename:    callbacks.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/24/22 2:18 AM
import os
from stable_baselines3.common.callbacks import EvalCallback


def get_eval_callback(env, model_save_path, freq=5000):
    return EvalCallback(env, best_model_save_path=model_save_path, eval_freq=freq, deterministic=True, render=False)