# @Filename:    record_model_playing.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/18/22 3:01 PM

from stable_baselines3 import A2C
from environment import StreetFighterEnv
import sys


def main(model_path, record_path, capture_movement):
    env = StreetFighterEnv(record_file=record_path, capture_movement=capture_movement)
    model = A2C.load(model_path)
    obs = env.reset()
    iteration = 0
    done = False
    for game in range(1):
        while not done:
            iteration += 1
            if done:
                obs = env.reset()
            env.render()
            action, critic = model.predict(obs)
            assert critic is None
            obs, reward, done, info = env.step(action)
            # time.sleep(0.01)
            if reward != 0: print(reward)
    print("iterations: ", iteration)
    return True

if __name__ == '__main__':
    model_path = sys.argv[1]
    record_path = sys.argv[2]
    capture_movement = False
    main(model_path, record_path, capture_movement)