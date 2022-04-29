# BiasNet
Deep Reinforcement Learning for Street Fighter II

* All experiment notebooks are in experiments directory. To run them ou might have to move them to root directory first and then run them. 
* Also some of them were not run after the recent changes. (probably should not include this in final version).

* Proposal is in proposal directory.
* Final Report is in report directory.
* tranfer knowledge from one player to other.


## Instructions to run experiments

* Command to generate bk2 given a trained model:
```
python main.py --command record --bias False --capture_movement True --record_path /tmp/record/ --render True --model_path experiments/final_models/unbiased_capture_movement/A2C_GUILE.zip --state guile.state
```

* Command to tune model using optuna, for all states:

It uses Bayesian Hyper parameter tuning from [BoTorch](https://botorch.org/tutorials/) sampler.

```
python main.py --command tune --n_jobs 4 --bias False --capture_movement True

```

for one state:
```
python main.py --command tune --bias True --capture_movement True --n_jobs 1 --state chunli.state  --model_path experiments/final_models/bias_capture_movement/A2C_CHUNLI.zip --trials 4
```

Currently range of hyperparameters to train are hardcoded in tuner class. Those are 
```
{
            'gamma': trial.suggest_loguniform('gamma', 0.8, 0.85),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 4e-4),
            'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.9)
}
```
* Command to train model, for all states:
```
python main.py --command train --bias True --capture_movement False --n_jobs 1

```
for one state
```
python main.py --command train --bias True --capture_movement False --n_jobs 1 --state guile.state

```

Currently all model params are hardcoded in main.py itself for training and ease of training.

* Command to fine tune a trained model for a state:
```
python main.py --command fine_tune --model_path experiments/final_models/biased_capture_movement/A2C_CHUNLI.zip --state chunli.state --bias True --capture_movement True
```


## Few useful retro commands

* Command to add a new environment:
```
python -m retro.import .
```

* Command to generate mp4 from bk2
```
python -m retro.scripts.playback_movie *bk2
```

## Results and experiments are in experiments directory.

## Videos for "CHUNLI vs. AGENT" are [here](experiments/final_recordings).
