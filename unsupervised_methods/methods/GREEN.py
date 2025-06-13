# unsupervised_methods/methods/GREEN.py

import numpy as np
from unsupervised_methods import utils

def GREEN(frames):
    """
    frames: numpy array, shape = (T, H, W, 3)
    Returns: BVP signal, shape = (T,)
    """
    precessed_data = utils.process_video(frames)  # shape = (T, 3)
    BVP = precessed_data[:, 1]  # green 채널만 사용
    return BVP.copy()
