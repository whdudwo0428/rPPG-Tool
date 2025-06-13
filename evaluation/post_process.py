# evaluation/post_process.py

"""The post processing files for calculating heart rate using FFT or peak detection.
The file also includes helper funcs such as detrend, mag2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import medfilt, butter, welch, filtfilt
from scipy.sparse import spdiags
import logging
logger = logging.getLogger(__name__)
from config import _C as config


def get_hr(y, sr, min_hr, max_hr):
    # nfft는 신호 길이에 맞춰 2의 거듭제곱, nperseg는 전체 길이
    nfft = _next_power_of_2(len(y))
    nperseg = len(y)
    p, q = welch(y, sr, nfft=nfft, nperseg=nperseg)
    mask = (p > min_hr / 60) & (p < max_hr / 60)
    if not np.any(mask):
        return 0.0
    idx = np.argmax(q[mask])
    return p[mask][idx] * 60


def get_psd(y, sr=30, min_hr=45, max_hr=150):
    p, q = welch(y, sr, nfft=int(1e5 / sr), nperseg=np.min((len(y) - 1, 256)))
    mask = (p > min_hr / 60.0) & (p < max_hr / 60.0)
    return p[mask], q[mask]


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def mag2db(mag):
    """Convert magnitude to dB, underflow 방지를 위해 small값 clip."""
    mag = np.clip(mag, 1e-12, None)
    return 20. * np.log10(mag)


def _calculate_fft_hr(ppg_signal, fs, low_cut, high_cut):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_cut) & (f_ppg <= high_cut))
    if fmask_ppg.size == 0:
        return 0.0
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    if len(ppg_peaks) < 2:
        return 0.0
    return 60.0 / (np.mean(np.diff(ppg_peaks)) / fs)


def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=None, high_pass=None):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the
    first and second harmonics of the ground truth HR frequency to the area under the curve of the
    remainder of the frequency spectrum, from 0.75 Hz to 2.5 Hz.

    Args:
        pred_ppg_signal(np.array): predicted PPG signal
        hr_label(float): ground truth HR in bpm
        fs(int or float): sampling rate of the video
    Returns:
        SNR(float): Signal-to-Noise Ratio in dB
    """
    first_harmonic_freq = hr_label / 60.0
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6.0 / 60.0  # 6 bpm → Hz

    pred_ppg_signal = np.squeeze(pred_ppg_signal)  # (T,)
    nperseg = min(len(pred_ppg_signal), 256)
    f_ppg, pxx_ppg = scipy.signal.welch(
        pred_ppg_signal,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False
    )

    idx_harmonic1 = np.argwhere(
        (f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere(
        (f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) &
                                ~((f_ppg >= (first_harmonic_freq - deviation)) & (
                                        f_ppg <= (first_harmonic_freq + deviation))) &
                                ~((f_ppg >= (second_harmonic_freq - deviation)) & (
                                        f_ppg <= (second_harmonic_freq + deviation))))

    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # ratio: (1st+2nd harmonic power) / (remainder power), 데시벨 단위로 변환
    eps = 1e-12
    ratio = (signal_power_hm1 + signal_power_hm2) / (signal_power_rem + eps)
    SNR_db = 10.0 * np.log10(ratio)
    return float(SNR_db)


def calculate_metric_per_video(predictions, labels, fs,
                               diff_flag, use_bandpass,
                               hr_method, min_hr, max_hr,
                               low_cut, high_cut):
    """Calculate video-level HR and SNR."""
    # 1) Detrend
    if diff_flag:
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)

    # 2) Bandpass
    if use_bandpass:
        order = config.TEST.DATA.PREPROCESS.BANDPASS.ORDER
        b, a = butter(order, [low_cut / fs * 2, high_cut / fs * 2], btype='bandpass')
        padlen = 3 * (len(a) - 1)
        if len(predictions) <= padlen or len(labels) <= padlen:
            logger.warning(f"Skipping clip: length {len(predictions)} ≤ padlen {padlen}")
            return np.nan, np.nan, np.nan
        try:
            predictions = filtfilt(b, a, predictions.astype(np.double))
            labels = filtfilt(b, a, labels.astype(np.double))
        except ValueError as e:
            logger.warning(f"filtfilt error: {e} — skipping clip.")
            return np.nan, np.nan, np.nan

    # 3) Hamming window (both pred & label)
    window = np.hamming(len(predictions))
    seg_pred = predictions * window
    seg_label = labels * window

    # 4) HR 계산
    if hr_method == 'FFT':
        hr_pred = get_hr(seg_pred, sr=fs, min_hr=min_hr, max_hr=max_hr)
        hr_label = get_hr(seg_label, sr=fs, min_hr=min_hr, max_hr=max_hr)
    elif hr_method == 'Peak':
        # Peak 방식: 배열→스칼라
        raw_pred = _calculate_peak_hr(predictions, fs=fs)
        raw_label = _calculate_peak_hr(labels, fs=fs)
        hr_pred = float(np.mean(raw_pred))
        hr_label = float(np.mean(raw_label))
    else:
        raise ValueError("hr_method must be 'FFT' or 'Peak'")

    # 5) Median smoothing (FFT일 때만 배열 리턴)
    if isinstance(hr_pred, np.ndarray):
        hr_pred = medfilt(hr_pred, kernel_size=5)
        hr_label = medfilt(hr_label, kernel_size=5)
        # 다시 스칼라로
        hr_pred = float(hr_pred)
        hr_label = float(hr_label)

    # 6) SNR: 외부에서 받은 low_cut/high_cut을 그대로 사용
    SNR = _calculate_SNR(
        predictions,
        hr_label,
        fs=fs,
        low_pass=low_cut,
        high_pass=high_cut
    )

    return hr_label, hr_pred, float(SNR)


def calculate_hr(predictions, labels, fs=30, diff_flag=False, min_hr=45, max_hr=150):
    """Calculate video‐level HR (bpm) only."""
    if diff_flag:
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)

    # bandpass filter using provided HR bounds (bpm → Hz)
    order = config.TEST.DATA.PREPROCESS.BANDPASS.ORDER
    low = min_hr / 60.0
    high = max_hr / 60.0
    b, a = butter(order, [low / fs * 2, high / fs * 2], btype='bandpass')
    predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
    labels = scipy.signal.filtfilt(b, a, np.double(labels))

    hr_pred = get_hr(predictions, sr=fs, min_hr=min_hr, max_hr=max_hr)
    hr_label = get_hr(labels, sr=fs, min_hr=min_hr, max_hr=max_hr)
    return float(hr_pred), float(hr_label)


def calculate_psd(predictions, labels, fs=30, diff_flag=False, low_cut=0.75, high_cut=2.5):
    """Calculate video‐level PSD (FFT) only."""
    if diff_flag:
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)

    order = config.TEST.DATA.PREPROCESS.BANDPASS.ORDER
    b, a = butter(order, [low_cut / fs * 2.0, high_cut / fs * 2.0], btype='bandpass')
    predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
    labels = scipy.signal.filtfilt(b, a, np.double(labels))

    psd_pred = get_psd(predictions, sr=fs, min_hr=low_cut * 60.0, max_hr=high_cut * 60.0)
    psd_label = get_psd(labels, sr=fs, min_hr=low_cut * 60.0, max_hr=high_cut * 60.0)
    return psd_pred, psd_label
