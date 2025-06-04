# unsupervised_methods/methods/PBV.py

import math
import numpy as np
from unsupervised_methods import utils

def PBV(frames):
    """
    frames: numpy array, shape = (T, H, W, 3)
    Returns: BVP signal (shape = (T,))
    """
    precessed_data = utils.process_video(frames)  # (T, 3)
    data_mean = np.mean(precessed_data, axis=0)   # shape = (3,)

    # 각 채널별 정규화
    R_norm = precessed_data[:, 0] / (data_mean[0] + 1e-8)  # (T,)
    G_norm = precessed_data[:, 1] / (data_mean[1] + 1e-8)  # (T,)
    B_norm = precessed_data[:, 2] / (data_mean[2] + 1e-8)  # (T,)

    PBV_n = np.array([np.std(R_norm), np.std(G_norm), np.std(B_norm)])  # (3,)
    PBV_d = math.sqrt(np.var(R_norm) + np.var(G_norm) + np.var(B_norm) + 1e-8)

    pbv = PBV_n / (PBV_d + 1e-8)  # (3,)

    # linear system Q·W = pbv
    C = precessed_data         # (T,3)
    Q = np.dot(C.T, C) + 1e-8 * np.eye(3)  # (3,3)
    W = np.linalg.solve(Q, pbv.reshape(3, 1)).reshape(3,)  # (3,)

    Numerator = np.dot(C, W)                 # (T,)
    Denominator = np.dot(pbv, W) + 1e-8      # scalar
    bvp = Numerator / Denominator            # (T,)
    return bvp.copy()

def PBV2(frames):
    """
    Alternative PBV 구현 (동일한 결과를 다른 방식으로 구함)
    """
    precessed_data = utils.process_video(frames)  # (T, 3)
    data_mean = np.mean(precessed_data, axis=0)

    R_norm = precessed_data[:, 0] / (data_mean[0] + 1e-8)
    G_norm = precessed_data[:, 1] / (data_mean[1] + 1e-8)
    B_norm = precessed_data[:, 2] / (data_mean[2] + 1e-8)

    PBV_n = np.array([np.std(R_norm), np.std(G_norm), np.std(B_norm)])  # (3,)
    PBV_d = math.sqrt(np.var(R_norm) + np.var(G_norm) + np.var(B_norm) + 1e-8)

    PBV = PBV_n / (PBV_d + 1e-8)

    C = precessed_data         # (T,3)
    Q = np.dot(C.T, C) + 1e-8 * np.eye(3)
    W = np.linalg.solve(Q, PBV.reshape(3,1)).reshape(3,)

    Numerator = np.dot(C, W)    # (T,)
    Denominator = np.dot(PBV, W) + 1e-8
    BVP = Numerator / Denominator
    return BVP.copy()
