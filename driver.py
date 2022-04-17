# @Filename:    driver.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/14/22 1:43 AM
from environment import StreetFighterEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from actor_critic import A2CCNNPolicy
from feature_extractors import CNNExtractorWithAttention, CNNExtractor



def get_trained_model(env, policy_network, feature_extractor_kwargs,
                      frame_size=1, monitor_log_file=None, log_dir=None, model=A2C, timesteps=1000000):
    """
    :param env: Environment
    :param policy_network: Policy network
    :param policy_kwargs: Feature extractor along with its arguments
    :param frame_size: Number of frames to stack for the input
    :param monitor_log_file: Log file for monitoring the training, its a csv
    :param log_dir: log directory for monitoring the training in tensorboard
    :param model: model type itself
    :param timesteps: timesteps to train
    :return:
    """
    env = Monitor(env, monitor_log_file, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=frame_size)
    model = model(policy_network, env, verbose=0, policy_kwargs=feature_extractor_kwargs, tensorboard_log=log_dir)
    model.learn(timesteps)
    return model

# # to include the custom feature extractor
# policy_kwargs = dict(
#     features_extractor_class=CNNExtractorWithAttention,
#     # features_extractor_kwargs=dict(observation_space=gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)),
# )
#
# policy = A2CCNNPolicy
# env = StreetFighterEnv()
# model0 = get_trained_model(env, policy, policy_kwargs, frame_size=1, monitor_log_file=None, log_dir=None, model=A2C, timesteps=10)
# model1 = get_trained_model(env, policy, policy_kwargs, frame_size=4, monitor_log_file=None, log_dir=None, model=A2C, timesteps=10)
#
#
# policy_kwargs = dict(
#     features_extractor_class=CNNExtractor,
# )
#
# model2 = get_trained_model(env, policy, policy_kwargs, frame_size=1, monitor_log_file=None, log_dir=None, model=A2C, timesteps=10)
#
# model3= get_trained_model(env, policy, policy_kwargs, frame_size=4, monitor_log_file=None, log_dir=None, model=A2C, timesteps=10)