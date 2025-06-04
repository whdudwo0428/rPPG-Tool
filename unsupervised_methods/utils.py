# unsupervised_methods/utils.py

import numpy as np
from scipy import sparse


def detrend(input_signal, lambda_value):
    """
    input_signal: 1D numpy array (length = signal_length)
    lambda_value: 스무딩 파라미터 (예: 100)

    반환값: detrended_signal (1D numpy array, shape = (signal_length,))

    - 박막평활 기반 detrending을 이용해 저주파 추세를 제거합니다.
    """
    signal_length = input_signal.shape[0]
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                       (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))),
        input_signal
    )
    return filtered_signal


def process_video(frames):
    """
    frames: numpy array, shape = (T, H, W, 3), dtype=np.uint8 또는 np.float32(0~255)

    반환값: RGB_ts (numpy array, shape = (T, 3))

    - 각 프레임마다 H×W 영역의 평균 RGB 값을 계산하여 시계열 평균 RGB를 만듭니다.
    """
    RGB = []
    for frame in frames:
        arr = frame.astype(np.float32)
        sum_per_channel = np.sum(np.sum(arr, axis=0), axis=0)
        avg_per_channel = sum_per_channel / (frame.shape[0] * frame.shape[1])
        RGB.append(avg_per_channel)
    return np.asarray(RGB)  # shape = (T, 3)
