# @Filename:    trainer.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/14/22 1:43 AM

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from constants import *


def _get_model(model_type, env, policy_network, feature_extractor_kwargs, log_dir=None, model_params=None, seed=SEED, verbose=0, n_steps=128):
    return model_type(policy_network, env, policy_kwargs=feature_extractor_kwargs,
                      seed=seed, verbose=verbose, tensorboard_log=log_dir, n_steps=n_steps, **model_params)


def get_trained_model(env, policy_network, feature_extractor_kwargs, model, timesteps,
                      model_params, monitor_log_file=None, log_dir=None, seed=SEED, frame_size=1):
    # print(model_params)
    env = Monitor(env, monitor_log_file, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    # # only using black and white images
    # env = VecFrameStack(env, n_stack=frame_size, channels_order='last')
    model = _get_model(model, env, policy_network, feature_extractor_kwargs,
                       model_params=model_params, log_dir=log_dir, seed=seed)
    model.learn(timesteps)
    return model

# # # to include the custom feature extractor
# policy_kwargs = dict(
#     features_extractor_class=CNNExtractor,
#     features_extractor_kwargs=dict(frame_size=4,)
# )
#
# model_params = {'gamma': 0.8858167808442736, 'learning_rate': 2.629936406785739e-05, 'gae_lambda': 0.8408608675098352}
# policy = A2CCNNPolicy
# env = StreetFighterEnv()
# model0 = get_trained_model(env=env, policy_network=policy,
#                            feature_extractor_kwargs=policy_kwargs, model=A2C, timesteps=2,
#                            model_params=model_params, frame_size=4, monitor_log_file=None, log_dir=None)
#
