# unsupervised_methods/methods/LGI.py

import numpy as np
from unsupervised_methods import utils

def LGI(frames):
    """
    frames: numpy array, shape = (T, H, W, 3)
    Returns: BVP signal (shape = (T,))
    """
    precessed_data = utils.process_video(frames)  # (T, 3)

    # SVD는 (3, T) 형태로 수행 → precessed_data.T
    U, _, _ = np.linalg.svd(precessed_data.T, full_matrices=False)  # U shape = (3, 3)
    # 가장 첫 번째 주성분 벡터(3,)을 구함
    S = U[:, 0]  # (3,)
    # (T,3) × (3,) = (T,)
    bvp = np.dot(precessed_data, S)  # shape = (T,)
    return bvp.copy()
