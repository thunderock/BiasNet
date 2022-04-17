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
    def __init__(self, record_file=None, state=None, capture_movement=False, image_size=84):

        super().__init__()
        self.image_size = image_size
        state = "Champion.Level1.RyuVsGuile" if state is None else state
        self.observation_space = Box(low=0, high=255, shape=(self.image_size, self.image_size, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        # to capture only the movement in last frame
        self.movement_capture = capture_movement
        if record_file:
            self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis',
                                  use_restricted_actions=retro.Actions.FILTERED, record=record_file, state=state)
        else:
            self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis',
                                  use_restricted_actions=retro.Actions.FILTERED, state=state)

    # reward function
    def get_reward(self, info, reward=None): return info['score'] - self.score

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess(obs)

        if self.movement_capture:
            # if we want images only which capture movement
            obs = obs - self.previous_frame
        self.previous_frame = obs
        reward = self.get_reward(info, reward)
        self.score = info['score']
        return obs, reward, done, info

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
        resized = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resized, (self.image_size, self.image_size, 1))