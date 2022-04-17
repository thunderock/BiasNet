# @Filename:    utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/16/22 8:50 PM


from stable_baselines3.common.evaluation import evaluate_policy
import plotly.io as pio
pio.renderers.default='notebook'



def evaluate_model_policy(env, model, n_eval_episodes=5):
    """
    Evaluate a policy

    :param env: (Gym Environment) The environment to evaluate the policy on
    :param model: (BaseRLModel object) the policy, whose type depends on the environment.
    :param n_eval_episodes: (int) number of episodes to evaluate the policy
    :return: (float) Mean reward for the `n_eval_episodes` episodes
    """
    return evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)[0]