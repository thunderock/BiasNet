# @Filename:    main.py.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/23/22 10:22 PM
import argparse, ast
from constants import *
from utils import GameState
from driver import tuner, recorder, trainer, fine_tune


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, choices=["train", "tune", "record", "fine_tune"], required=True)
    parser.add_argument("--bias", type=ast.literal_eval, default=True)
    parser.add_argument("--capture_movement", type=ast.literal_eval, default=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--render", type=ast.literal_eval, default=False)
    parser.add_argument("--state", type=str, default=STATE_GUILE, choices=list(GameState._value2member_map_.keys()))
    parser.add_argument("--record_path", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=PARALLEL_ENV_COUNT)
    args = parser.parse_args()
    model_params = {'gamma': 0.8074138106735396, 'learning_rate': 0.0001, 'gae_lambda': 0.8787060424267222}
    model_name = "A2C"
    model_dir = "models/{}_{}".format("biased" if args.bias else "unbiased", "capture_movement" if args.capture_movement else "no_capture_movement")
    time_steps = 5000000
    trials = 2
    N_JOBS = args.n_jobs
    if args.command == "train":
        trainer(bias=args.bias, capture_movement=args.capture_movement, model_params=model_params, time_steps=time_steps, model_dir=model_dir, model_name=model_name, n_jobs=args.n_jobs, states=args.state)
    elif args.command == "tune":
        tuner(bias=args.bias, capture_movement=args.capture_movement, time_steps=time_steps,model_dir=model_dir, model_name=model_name, trials=trials, n_jobs=args.n_jobs)
    elif args.command == "record":
        recorder(model_path=args.model_path, capture_movement=args.capture_movement, state=args.state, model_name=model_name, render=args.render, record_dir=args.record_path)
    elif args.command == "fine_tune":
        model_params = {'gamma': 0.8074138106735396, 'gae_lambda': 0.8787060424267222, 'learning_rate': 1e-5}
        fine_tune(model_name=model_name, model_path=args.model_path, tensorboard_path='models/', model_params=model_params, state=args.state, time_steps=30000, bias=args.bias, capture_movement=args.capture_movement, model_save_path='models/')
    else:
        raise ValueError("Invalid command")
