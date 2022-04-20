# @Filename:    record_model_playing.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/18/22 3:01 PM

import sys
from utils import record_model_playing


if __name__ == '__main__':
    model_path = sys.argv[1]
    capture_movement = False
    render = False
    record_path = './recordings/'
    try:
        record_path = sys.argv[2]
        capture_movement = sys.argv[3]
        render = sys.argv[4]
    except IndexError:
        pass
    record_model_playing(model_path, record_path, capture_movement, render)