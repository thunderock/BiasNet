# BiasNet
Reinforcement Learning for Street Fighter using GNNs

* All experiment notebooks are in experiments directory. To run them ou might have to move them to root directory first and then run them. 
* Also some of them were not run after the recent changes. (probably should not include this in final version).

* Proposal is in proposal directory.
* Final Report is in report directory.
* tranfer knowledge from one player to other.
## Few useful retro commands

* Command to add a new environment:
```
python -m retro.import .
```

* Command to generate mp4 from bk2
```
python -m retro.scripts.playback_movie *bk2
```


* Command to generate bk2 given a trained model:
```
python main.py --command record --bias False --capture_movement True --record_path /tmp/record/ --render True --model_path experiments/final_models/unbiased_capture_movement/A2C_GUILE.zip --state guile.state
```
