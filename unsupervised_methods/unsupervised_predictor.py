# unsupervised_methods/unsupervised_predictor.py

import numpy as np
from tqdm import tqdm

from evaluation.post_process import calculate_metric_per_video
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.ICA_POH import ICA_POH
from unsupervised_methods.methods.LGI import LGI
from unsupervised_methods.methods.PBV import PBV
from unsupervised_methods.methods.POS_WANG import POS_WANG

def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset. """
    if data_loader.get("unsupervised") is None:
        raise ValueError("No data for unsupervised method predicting")

    print(f"=== Unsupervised Method ({method_name}) Predicting ===")

    predict_hr_peak_all = []
    gt_hr_peak_all      = []
    predict_hr_fft_all  = []
    gt_hr_fft_all       = []
    SNR_all             = []

    eval_method = config.INFERENCE.EVALUATION_METHOD.strip().lower()
    use_peak    = 'peak' in eval_method

    diff_flag = ("DiffNormalized" in config.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE)
    win_s  = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE
    step_s = getattr(config.INFERENCE.EVALUATION_WINDOW, 'WINDOW_STEP_SIZE', win_s)

    fs     = config.UNSUPERVISED.DATA.FS

    win_f  = int(win_s * fs)
    step_f = int(step_s * fs)

    for batch in tqdm(data_loader["unsupervised"], ncols=80):
        # batch[0]: tensor of shape (B, T, C, H, W) when DATA_FORMAT='NDCHW'
        frames = batch[0]  # torch.Tensor
        # (B, T, C, H, W) → (B, T, H, W, C)
        if frames.ndim == 5:
            frames = frames.permute(0, 1, 3, 4, 2)
        frames_batch = frames.cpu().numpy()  # now (B, T, H, W, C)
        labels_batch = batch[1].cpu().numpy()  # (B, T)
        B = frames_batch.shape[0]

        for i in range(B):
            data_input   = frames_batch[i]  # (T, H, W, 3)
            labels_input = labels_batch[i]  # (T,)

            # 1) rPPG 추정
            name = method_name.strip().lower()
            # 지원하는 메소드 이름을 모두 매핑
            if name in ("pos", "pos_wang"):
                bvp_signal = POS_WANG(data_input, fs)
            elif name in ("chrom", "chrome", "chrome_dehaan"):
                bvp_signal = CHROME_DEHAAN(data_input, fs)
            elif name in ("ica", "ica_poh"):
                bvp_signal = ICA_POH(data_input, fs)
            elif name == "green":
                bvp_signal = GREEN(data_input)
            elif name == "lgi":
                bvp_signal = LGI(data_input)
            elif name == "pbv":
                bvp_signal = PBV(data_input)
            else:
                raise ValueError(f"Unsupported unsupervised method: {method_name}")

            # 2) 슬라이딩 윈도우
            T = bvp_signal.shape[0]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window = min(win_f, T)
            else:
                window = T if T < win_f else win_f

            for start in range(0, T - window + 1, step_f):
                seg_bvp   = bvp_signal[start : start + window]
                seg_label = labels_input[start : start + window]

                # padlen 기반 최소 길이 확인 (예: padlen = 3*(order-1))
                order = config.UNSUPERVISED.DATA.PREPROCESS.BANDPASS.ORDER
                padlen = 3 * (order - 1)
                if seg_bvp.shape[0] < padlen + 1:
                    continue

                # ——— Bandpass/HR 설정 한 번 읽기 ———
                pre = config.UNSUPERVISED.DATA.PREPROCESS.BANDPASS
                use_bp, low_cut, high_cut = pre.APPLY, pre.LOW_CUT, pre.HIGH_CUT
                min_hr, max_hr = low_cut * 60, high_cut * 60

                if use_peak:
                    # Peak 방식으로 HR & SNR 계산
                    hr_gt, hr_pr, snr = calculate_metric_per_video(
                        seg_bvp, seg_label,
                        fs=fs,
                        diff_flag=diff_flag,
                        use_bandpass=use_bp,
                        hr_method='Peak',
                        min_hr=min_hr,
                        max_hr=max_hr,
                        low_cut=low_cut,
                        high_cut=high_cut
                    )
                    gt_hr_peak_all.append(hr_gt)
                    predict_hr_peak_all.append(hr_pr)
                else:
                    # FFT 방식으로 HR & SNR 계산
                    hr_gt, hr_pr, snr = calculate_metric_per_video(
                        seg_bvp, seg_label,
                        fs=fs,
                        diff_flag=diff_flag,
                        use_bandpass=use_bp,
                        hr_method='FFT',
                        min_hr=min_hr,
                        max_hr=max_hr,
                        low_cut=low_cut,
                        high_cut=high_cut
                    )
                    gt_hr_fft_all.append(hr_gt)
                    predict_hr_fft_all.append(hr_pr)

                # SNR 저장 (Peak/FFT 각각 분기 내에서만 정의됨)
                SNR_all.append(snr if np.isfinite(snr) else 0.0)

    print(f"Used Unsupervised Method: {method_name}")

    # 3) 최종 메트릭 출력
    SNR_all = np.array(SNR_all)
    if use_peak:
        preds = np.array(predict_hr_peak_all)
        gts   = np.array(gt_hr_peak_all)
        tag   = "Peak"
    else:
        preds = np.array(predict_hr_fft_all)
        gts   = np.array(gt_hr_fft_all)
        tag   = "FFT"

    n = len(preds)
    for metric in config.UNSUPERVISED.METRICS:
        if metric == "MAE":
            vals = np.abs(preds - gts)
            v  = vals.mean() if n > 0 else float('nan')
            se = vals.std()  / np.sqrt(n) if n > 0 else float('nan')
            print(f"{tag} MAE: {v:.3f} ± {se:.3f}")
        elif metric == "RMSE":
            errs = preds - gts
            v    = np.sqrt((errs**2).mean()) if n > 0 else float('nan')
            se   = errs.std() / np.sqrt(n) if n > 0 else float('nan')
            print(f"{tag} RMSE: {v:.3f} ± {se:.3f}")
        elif metric == "MAPE":
            per = np.abs((preds - gts) / (gts + 1e-8)) * 100
            v  = per.mean() if n > 0 else float('nan')
            se = per.std() / np.sqrt(n) if n > 0 else float('nan')
            print(f"{tag} MAPE: {v:.2f}% ± {se:.2f}%")
        elif metric == "Pearson":
            if n < 3 or np.std(preds) == 0 or np.std(gts) == 0:
                print(f"{tag} Pearson: skipped (insufficient data or zero variance)")
            else:
                corr = np.corrcoef(preds, gts)[0, 1]
                se   = np.sqrt((1 - corr**2) / (n - 2))
                print(f"{tag} Pearson: {corr:.3f} ± {se:.3f}")
        elif metric == "SNR":
            v  = SNR_all.mean() if n > 0 else float('nan')
            se = SNR_all.std() / np.sqrt(n) if n > 0 else float('nan')
            print(f"{tag} SNR: {v:.3f} ± {se:.3f} dB")
        else:
            raise ValueError(f"Unsupported metric: {metric}")
