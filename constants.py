# @Filename:    constants.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/16/22 9:33 PM

import warnings
warnings.filterwarnings("ignore")
from stable_baselines3.common.utils import get_device, set_random_seed

SEED = 2000728661
DEVICE = get_device("auto")
set_random_seed(SEED)
BATCH_SIZE = 256
VERBOSE = 0

STATE_GUILE = 'guile.state'
STATE_ZANGIEF = 'zangief.state'
STATE_DAHLISM = 'dahlism.state'
STATE_EHDONA = 'ehonda.state'
STATE_CHUNLI = 'chunli.state'
STATE_BLANKA = 'blanka.state'
STATE_KEN = 'ken.state'
STATE_RYU = 'ryu.state'

PARALLEL_ENV_COUNT = 1
