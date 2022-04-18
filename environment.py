# @Filename:    environment.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/9/22 1:21 AM
import os
import numpy as np
import retro
from gym import Env
from gym.spaces import MultiBinary, Box
import cv2
import shutil

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
            shutil.rmtree(record_file, ignore_errors=True)
            os.mkdir(record_file)
            self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis',
                                  use_restricted_actions=retro.Actions.FILTERED, record=record_file, state=state)
        else:
            self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis',
                                  use_restricted_actions=retro.Actions.FILTERED, state=state)

    # reward function
    def get_reward(self, info, win_reward=10000, getting_hit_penalty=.5, hitting_enemy_reward=1):
        # reward for hitting enemy
        score = info['score'] - self.score
        # penalty for losing health
        health = info['health'] - self.health  # between [-176, 0]
        # reward for winning
        win = 0
        if info['enemy_health'] == 0:
            win = 1
        elif info['health'] == 0:
            win = -1
        return hitting_enemy_reward * score + win * win_reward + health * getting_hit_penalty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess(obs)

        if self.movement_capture:
            # if we want images only which capture movement
            obs = obs - self.previous_frame
        self.previous_frame = obs
        reward = self.get_reward(info)
        self.score, self.enemy_health, self.health = info['score'], info['enemy_health'], info['health']
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def reset(self):
        obs = self.env.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        self.health = 176
        self.enemy_health = 176
        return obs

    def close(self):
        self.env.close()

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resized, (self.image_size, self.image_size, 1))