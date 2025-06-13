# unsupervised_methods/methods/ICA_POH.py

import math
import numpy as np
from scipy import linalg, signal
from unsupervised_methods import utils

def ICA_POH(frames, FS):
    """
    frames: numpy array, shape = (T, H, W, 3)
    FS:    sampling rate
    Returns: BVP signal (shape = (T,))
    """
    LPF = 0.7
    HPF = 2.5

    # (T,3) 배열 얻기
    RGB = utils.process_video(frames)  # (T, 3)
    NyquistF = FS / 2.0

    # R/G/B 각각 detrend + z-score 정규화
    BGRNorm = np.zeros_like(RGB)  # (T, 3)
    Lambda = 100
    for c in range(3):
        detrended = utils.detrend(RGB[:, c], Lambda)  # (T,)
        BGRNorm[:, c] = (detrended - np.mean(detrended)) / (np.std(detrended) + 1e-8)

    # ICA 분리 (JADE)
    _, S = ica(np.mat(BGRNorm).H, 3)  # S shape = (3, T)

    # 파워 스펙트럼 계산하여 “가장 파워가 큰” 컴포넌트 선택
    MaxPx = np.zeros((3,))
    for c in range(3):
        FF = np.fft.fft(S[c, :])       # (T,)
        FF = FF[1:]                   # DC 제외
        Px = np.abs(FF) ** 2
        Px = Px / (np.sum(Px) + 1e-8)
        MaxPx[c] = np.max(Px)

    MaxComp = int(np.argmax(MaxPx))
    BVP_I = S[MaxComp, :]             # shape = (T,)

    # bandpass 필터링
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], btype='bandpass')
    BVP_F = signal.filtfilt(B, A, np.real(BVP_I).astype(float))

    return BVP_F.copy()  # (T,)

#---------------------------------------
def ica(X, Nsources, Wprev=0):
    """
    X:        np.matrix 형태, shape = (3, T)
    Nsources: 분리할 source 개수 (3)
    Wprev:    초기 가중치 (0이면 identity)
    Returns:  W(혼합행렬), Zhat(분리된 신호, shape=(3, T))
    """
    nRows, nCols = X.shape
    if nRows > nCols:
        print("[ICA_POH] Warning: rows > cols (행렬 비정방형), 필요 시 전치 권장.")
    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print(f"[ICA_POH] Warning: Nsources > 데이터 채널 수, 줄여서 사용: {Nsources}")

    Winv, Zhat = jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)
    return W, Zhat

#---------------------------------------
def jade(X, m, Wprev):
    """
    X:        np.matrix of shape (3, T)
    m:        분리할 컴포넌트 개수
    Wprev:    지난번 가중치 (없으면 0)
    Returns:  Winv (혼합행렬), S (분리된 신호, shape=(m, T))
    """
    n, T = X.shape
    nem = m
    seuil = 1 / math.sqrt(T) / 100

    # (1) 백색화(Whitening)
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, X.H) / T)
        idx = np.argsort(D)
        pu = D[idx]
        ibl = np.sqrt(pu[n-m:n] - np.mean(pu[:n-m]))
        bl = 1.0 / (ibl + 1e-8)
        W = np.dot(np.diag(bl), np.transpose(U[:, idx[n-m:n]]))
        IW = np.dot(U[:, idx[n-m:n]], np.diag(ibl + 1e-8))
    else:
        IW = linalg.sqrtm(np.matmul(X, X.H) / T)
        W = np.linalg.inv(IW + 1e-8)

    Y = np.dot(W, X)
    R = np.matmul(Y, Y.H) / T
    C = np.matmul(Y, Y.T) / T

    # (2) 4차 cumulant 큐브 구해서 Q 벡터화
    Q = np.zeros((m*m*m*m, 1), dtype=complex)
    index = 0
    for lx in range(m):
        Yl = Y[lx, :]
        for kx in range(m):
            Yk = Y[kx, :]
            for jx in range(m):
                Yj = Y[jx, :]
                for ix in range(m):
                    Q[index] = (
                        np.dot((Yl * np.conj(Yk)), (Yj * np.conj(Y[ix, :])).conj().T) / T
                        - R[ix, jx] * R[lx, kx]
                        - R[ix, kx] * R[lx, jx]
                        - C[ix, lx] * np.conj(C[jx, kx])
                    )
                    index += 1

    # (3) eigen-decomposition & 축소 과정
    D, U = np.linalg.eig(Q.reshape(m*m, m*m))
    Diag = np.abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem*m), dtype=complex)
    h = m*m - 1
    for u in range(0, nem*m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u:u+m] = la[h] * Z
        h -= 1

    # (4) 반복적 Givens 회전
    B = np.array([[1, 0, 0], [0, 1, 1], [0, -1j, 1j]])
    Bt = B.conj().T
    encore = True
    if Wprev == 0:
        V = np.eye(m, dtype=complex)
    else:
        V = np.linalg.inv(Wprev + 1e-8)

    while encore:
        encore = False
        for p in range(m-1):
            for q in range(p+1, m):
                Ip = np.arange(p, nem*m, m)
                Iq = np.arange(q, nem*m, m)
                g = np.vstack([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.dot(g, g.conj().T)
                temp2 = np.dot(B, temp1)
                temp = np.dot(temp2, Bt)
                Dd, vcp = np.linalg.eig(np.real(temp))
                kk = np.argsort(Dd)
                angles = vcp[:, kk[-1]]
                if angles[0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0]/2)
                s = 0.5 * (angles[1] - 1j*angles[2]) / c

                if np.abs(s) > seuil:
                    encore = True
                    G = np.array([[c, -np.conj(s)], [s, c]])
                    V[:, [p, q]] = np.dot(V[:, [p, q]], G)
                    M[[p, q], :] = np.dot(G.conj().T, M[[p, q], :])
                    temp_p = c * M[:, Ip] + s * M[:, Iq]
                    temp_q = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    M[:, Ip] = temp_p
                    M[:, Iq] = temp_q

    A = np.dot(IW, V)
    S = np.dot(V.conj().T, Y)  # shape = (m, T)
    return A, S
