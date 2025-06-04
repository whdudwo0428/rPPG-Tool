# evaluation/post_process.py

"""The post processing files for calculating heart rate using FFT or peak detection.
The file also includes helper funcs such as detrend, mag2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter, welch
from scipy.sparse import spdiags


def get_hr(y, sr=30, min=45, max=150):
    # nfft는 신호 길이에 맞춰 2의 거듭제곱, nperseg는 전체 길이
    nfft = _next_power_of_2(len(y))
    nperseg = len(y)
    p, q = welch(y, sr, nfft=nfft, nperseg=nperseg)
    mask = (p > min/60) & (p < max/60)
    if not np.any(mask):
        return 0.0
    idx = np.argmax(q[mask])
    return p[mask][idx] * 60


def get_psd(y, sr=30, min=45, max=150):
    p, q = welch(y, sr, nfft=int(1e5/sr), nperseg=np.min((len(y)-1, 256)))
    mask = (p > min/60) & (p < max/60)
    return q[mask]


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


def _calculate_fft_hr(ppg_signal, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
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


def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
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

    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) &
                               ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) &
                               ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    eps = 1e-12
    ratio = (signal_power_hm1 + signal_power_hm2) / (signal_power_rem + eps)
    SNR = mag2db(ratio + eps)
    return float(SNR)


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR."""
    if diff_flag:
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)

    if use_bandpass:
        b, a = butter(3, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))

    if hr_method == 'FFT':
        hr_pred = get_hr(predictions, sr=fs)
        hr_label = get_hr(labels, sr=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError("Please use hr_method='FFT' or 'Peak'")

    SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    return float(hr_label), float(hr_pred), float(SNR)


def calculate_hr(predictions, labels, fs=30, diff_flag=False):
    """Calculate video‐level HR (bpm) only."""
    if diff_flag:
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)

    b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
    labels = scipy.signal.filtfilt(b, a, np.double(labels))

    hr_pred = get_hr(predictions, sr=fs)
    hr_label = get_hr(labels, sr=fs)
    return float(hr_pred), float(hr_label)


def calculate_psd(predictions, labels, fs=30, diff_flag=False):
    """Calculate video‐level PSD (FFT) only."""
    if diff_flag:
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)

    b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
    labels = scipy.signal.filtfilt(b, a, np.double(labels))

    psd_pred = get_psd(predictions, sr=fs)
    psd_label = get_psd(labels, sr=fs)
    return psd_pred, psd_label
