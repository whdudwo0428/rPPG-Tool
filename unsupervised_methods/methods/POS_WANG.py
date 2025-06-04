# unsupervised_methods/methods/POS_WANG.py

import math
import numpy as np
from scipy import signal
from unsupervised_methods import utils

def POS_WANG(frames, fs):
    """
    기존 방식: 윈도우별로 윈도우 평균으로 나누는 정규화
    """
    WinSec = 1.6
    RGB = utils.process_video(frames)  # shape = (T, 3)
    N = RGB.shape[0]
    H = np.zeros((1, N))               # BVP 누적용
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            window_RGB = RGB[m:n, :]                # shape=(l, 3)
            # 기존: 윈도우 평균 나누기
            Cn = window_RGB / (np.mean(window_RGB, axis=0)[None, :] + 1e-8)
            Cn = Cn.T                                # shape=(3, l)
            S  = np.dot(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)  # shape=(2, l)

            h = S[0, :] + (np.std(S[0, :]) / (np.std(S[1, :]) + 1e-8)) * S[1, :]
            h = h - np.mean(h)
            H[0, m:n] += h

    BVP = H.flatten()
    BVP = utils.detrend(BVP, 100)
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(float))
    return BVP.copy()


def POS_WANG_minmax(frames, fs):
    """
    POS_WANG 알고리즘 (Min–Max 정규화 버전)

    - frames: numpy array, shape=(T, H, W, 3)
    - fs:    sampling rate (예: 30.0)
    Returns: BVP signal (shape = (T,))

    윈도우별로 Min–Max 정규화를 적용한 뒤 S0 + α·S1 조합을 수행하고,
    최종적으로 detrend + bandpass 필터링을 적용합니다.
    """
    WinSec = 1.6
    # 1) 프레임별 평균(R, G, B)을 구하는 유틸 함수 호출 → shape=(T, 3)
    RGB = utils.process_video(frames)  # (T, 3)
    N = RGB.shape[0]
    H = np.zeros((1, N), dtype=float)  # BVP 누적용 배열
    l = math.ceil(WinSec * fs)  # 윈도우 프레임 수

    # 2) 각 시점 n에 대해, 과거 l프레임(window_RGB)을 꺼내서 Min–Max 정규화 수행
    for n in range(N):
        m = n - l
        if m >= 0:
            # (2.1) l×3 크기의 window_RGB를 추출
            window_RGB = RGB[m:n, :]  # shape=(l, 3)

            # (2.2) 각 채널별 최소값·최대값 계산
            min_rgb = np.min(window_RGB, axis=0)[None, :]  # shape=(1,3)
            max_rgb = np.max(window_RGB, axis=0)[None, :]  # shape=(1,3)
            range_rgb = max_rgb - min_rgb + 1e-8  # shape=(1,3)

            # (2.3) Min–Max 정규화: [0,1] 범위로 스케일링
            window_mm = (window_RGB - min_rgb) / range_rgb  # shape=(l, 3)

            # (3) 채널별 조합 (Chrominance 연산)
            Cn = window_mm.T  # (3, l)

            # S0, S1 신호 계산: shape=(2, l)
            S = np.dot(np.array([[0, 1, -1],
                                 [-2, 1, 1]]), Cn)

            # (4) α 계산: std(S0)/std(S1)
            std0 = np.std(S[0, :]) + 1e-8
            std1 = np.std(S[1, :]) + 1e-8
            alpha = std0 / std1

            # (5) h 신호 계산 및 평균 제거
            h = S[0, :] + alpha * S[1, :]  # shape=(l,)
            h = h - np.mean(h)  # zero-mean

            # (6) 누적: H[0, m:n]에 h를 더함
            H[0, m:n] += h

    # (7) 누적된 신호를 1차원 BVP로 변경
    BVP = H.flatten()  # shape=(N,)

    # (8) Detrend (λ=100)
    BVP = utils.detrend(BVP, 100)  # shape=(N,)

    # (9) 1차 Butterworth Bandpass 필터: 0.75–3 Hz
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(float))

    return BVP.copy()  # shape=(N,)


def POS_WANG_zscore(frames, fs):
    """
    새로 추가: 윈도우별로 z-score 표준화 (평균 빼고, 분산으로 나누기) 사용
    """
    WinSec = 1.6
    RGB = utils.process_video(frames)  # shape=(T, 3)
    N = RGB.shape[0]
    H = np.zeros((1, N), dtype=float)
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            window_RGB = RGB[m:n, :]                           # (l, 3)
            mean_rgb = np.mean(window_RGB, axis=0)[None, :]    # shape=(1, 3)
            std_rgb  = np.std(window_RGB, axis=0)[None, :] + 1e-8
            window_z = (window_RGB - mean_rgb) / std_rgb        # (l, 3)

            Cn = window_z.T                                    # (3, l)
            S  = np.dot(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)  # (2, l)

            h = S[0, :] + (np.std(S[0, :]) / (np.std(S[1, :]) + 1e-8)) * S[1, :]
            h = h - np.mean(h)
            H[0, m:n] += h

    BVP = H.flatten()
    BVP = utils.detrend(BVP, 100)
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(float))
    return BVP.copy()