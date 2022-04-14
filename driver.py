# @Filename:    driver.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/14/22 1:43 AM
from environment import StreetFighterEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from actor_critic import CustomActorCriticPolicy


policy = CustomActorCriticPolicy
env = StreetFighterEnv()
env = Monitor(env, "\tmp\monitor", allow_early_resets=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 1, channels_order='last')
model = A2C(policy, env, verbose=1)
model.learn(5000)