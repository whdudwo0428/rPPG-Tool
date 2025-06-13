# File: dataset/preprocessing/normalization.py

import numpy as np

# 이미지 배열에서 채널이 어느 축에 있는지 설정하세요 (예: -1 혹은 1)
CHANNEL_AXIS = -1


def standardize_frames(frames: np.ndarray) -> np.ndarray:
    """
    Channel-wise Z-score over (T, H, W) for each channel.
    - dtype을 float32로 통일
    - std가 너무 작을 경우 eps로 클램핑
    - NaN은 0으로 대체
    """
    eps = 1e-6
    # 1) dtype 통일
    frames = frames.astype(np.float32, copy=False)

    # 2) μ, σ 계산 (채널 제외 모든 차원)
    axis = tuple(i for i in range(frames.ndim) if i != CHANNEL_AXIS)
    mu = frames.mean(axis=axis, keepdims=True)
    raw_sigma = frames.std(axis=axis, keepdims=True)
    sigma = np.where(raw_sigma > eps, raw_sigma, eps)

    # 3) 정규화 및 NaN 제거
    out = (frames - mu) / sigma
    out[np.isnan(out)] = 0
    return out


def diff_normalize_frames(frames: np.ndarray) -> np.ndarray:
    """
    Frame difference normalized by global std, pad last frame with zeros.
    - dtype을 float32로 통일
    - 분모가 너무 작을 경우 eps로 클램핑
    - global std가 작을 경우 eps로 클램핑
    - NaN은 0으로 대체
    """
    eps = 1e-6
    frames = frames.astype(np.float32, copy=False)

    # 1) 차분 비율 계산 (denom 클램핑)
    denom = frames[1:] + frames[:-1]
    denom = np.where(np.abs(denom) > eps, denom, eps)
    diffs = (frames[1:] - frames[:-1]) / denom

    # 2) 채널별(std over time+H+W)로 정규화
    # diffs.shape = (T-1, C, H, W)
    axis = (0, 2, 3)  # time, height, width
    raw_sigma = diffs.std(axis=axis, keepdims=True)  # shape (1, C, 1, 1)
    sigma = np.where(raw_sigma > eps, raw_sigma, eps)  # clamp for stability
    diffs = diffs / sigma  # broadcasting

    # 3) 마지막 프레임 0으로 패딩
    pad = np.zeros((1, *diffs.shape[1:]), dtype=diffs.dtype)
    out = np.concatenate((diffs, pad), axis=0)
    out[np.isnan(out)] = 0
    return out


def standardize_labels(bvp: np.ndarray):
    """
    1D BVP 라벨 Z-score 정규화
    - return: (normalized_labels, μ, σ)
    """
    eps = 1e-6
    bvp = bvp.astype(np.float32, copy=False)

    mu = bvp.mean()
    raw_sigma = bvp.std()
    sigma = raw_sigma if raw_sigma > eps else eps

    lbl = (bvp - mu) / sigma
    lbl[np.isnan(lbl)] = 0
    return lbl, mu, sigma


def diff_normalize_labels(bvp: np.ndarray):
    """
    1D BVP 라벨 차분 정규화
    - return: (normalized_diff_labels (length T), μ_diff, σ_diff)
    """
    eps = 1e-6
    bvp = bvp.astype(np.float32, copy=False)

    diffs = np.diff(bvp)             # length T-1
    mu_diff = diffs.mean()
    raw_sigma_diff = diffs.std()
    sigma_diff = raw_sigma_diff if raw_sigma_diff > eps else eps

    lbl = (diffs - mu_diff) / sigma_diff
    lbl = np.append(lbl, [0.0], axis=0)  # 마지막 원소 패딩
    lbl[np.isnan(lbl)] = 0
    return lbl, mu_diff, sigma_diff
