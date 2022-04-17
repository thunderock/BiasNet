# @Filename:    utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/16/22 8:50 PM


from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import plotly

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