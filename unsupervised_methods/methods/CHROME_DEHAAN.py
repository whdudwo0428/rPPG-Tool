# unsupervised_methods/methods/CHROME_DEHAAN.py

import math
import numpy as np
from scipy import signal
from unsupervised_methods import utils

def CHROME_DEHAAN(frames, FS):
    """
    frames: numpy array, shape = (T, H, W, 3)
    FS    : sampling rate (예: 30.0)
    Returns: BVP signal, shape ~ (totallen,)
    """
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6

    # (T,3) 배열 얻기
    RGB = utils.process_video(frames)  # shape = (T, 3)
    FN = RGB.shape[0]
    NyquistF = FS / 2.0
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], btype='bandpass')

    # 윈도우 길이(프레임 수)
    WinL = math.ceil(WinSec * FS)
    if (WinL % 2) == 1:
        WinL += 1
    overlap = WinL // 2
    # 총 윈도우 개수 (half-overlap)
    NWin = math.floor((FN - overlap) / overlap)

    WinS = 0
    WinM = overlap
    WinE = WinS + WinL
    totallen = overlap * (NWin + 1)
    S = np.zeros(totallen)

    for i in range(NWin):
        # 윈도우 내 평균으로 정규화
        RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)  # shape = (3,)
        RGBNorm = RGB[WinS:WinE, :] / RGBBase[None, :]  # shape = (WinL, 3)

        Xs = 3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1]           # shape = (WinL,)
        Ys = 1.5 * RGBNorm[:, 0] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 2]  # (WinL,)

        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys, axis=0)

        Alpha = np.std(Xf) / (np.std(Yf) + 1e-8)
        SWin = Xf - Alpha * Yf  # (WinL,)
        SWin = SWin * signal.hanning(WinL)

        S[WinS:WinM] += SWin[:overlap]
        S[WinM:WinE]  = SWin[overlap:]
        WinS = WinM
        WinM = WinS + overlap
        WinE = WinS + WinL

    return S  # shape ~ (totallen,)