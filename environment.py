# @Filename:    environment.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/9/22 1:21 AM

import numpy as np
import retro
from gym import Env
from gym.spaces import MultiBinary, Box
import cv2


class StreetFighterEnv(Env):
    def __init__(self, record_file=None):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        if record_file:
            self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED, record=record_file)
        else:
            self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)

    def get_reward(self, info, reward=None): return info['score'] - self.score

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess(obs)
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs
        reward = self.get_reward(info)
        self.score = info['score']
        return frame_delta, reward, done, info

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def reset(self):
        obs = self.env.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        return obs

    def close(self):
        self.env.close()

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resized, (84, 84, 1))