# @Filename:    constants.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/16/22 9:33 PM

import warnings
warnings.filterwarnings("ignore")
from stable_baselines3.common.utils import get_device

SEED = 2000728661
DEVICE = get_device("auto")