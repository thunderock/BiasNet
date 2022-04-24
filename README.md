# BiasNet
Reinforcement Learning for Street Fighter using GNNs

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

* Command to train model, only for all states:
```
python driver.py --command tuner --n_jobs 4 --bias False --capture_movement True

```

*Command to train model, for all states:
```
python driver.py --command train --bias True --capture_movement False --n_jobs 1

```


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
