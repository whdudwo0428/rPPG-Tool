# File: evaluation/metrics.py

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset.data_loader.Base_Loader import BaseLoader
from dataset.data_loader.PURE_Loader import PURELoader
from evaluation.BlandAltmanPy import BlandAltman
from evaluation.post_process import calculate_metric_per_video


def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv(f"label/{dataset}_Comparison.csv")
    out_dict = df.to_dict(orient='index')
    return {str(v['VideoID']): v for v in out_dict.values()}

def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    if index.startswith('subject'):
        index = index[7:]
    video_dict = feed_dict[index]
    pref = video_dict['Preferred']
    return index, video_dict['Peak Detection' if pref == 'Peak Detection' else 'FFT']

def _reform_data_from_dict(data, flatten=True):
    """Reformat predictions/labels from dicts into a flat NumPy array."""
    tensors = [v for _, v in sorted(data.items(), key=lambda x: x[0])]
    cat = torch.cat(tensors, dim=0)
    arr = cat.cpu().numpy()
    return arr.reshape(-1) if flatten else arr

def calculate_metrics(predictions, config):
    """Calculate rPPG metrics and save per-video arrays."""
    predict_hr_fft_all  = []
    gt_hr_fft_all       = []
    predict_hr_peak_all = []
    gt_hr_peak_all      = []
    snr_list            = []

    # 결과 저장 디렉터리 (프로젝트 루트 ./output/rPPGandLabel)
    save_dir = "./output/rPPGandLabel"
    os.makedirs(save_dir, exist_ok=True)

    raw_root = config.TEST.DATA.RAW_DATA_PATH
    for vid in tqdm(predictions.keys(), ncols=80):
        # 1) 모델 예측 PPG 시계열
        pred_arr = _reform_data_from_dict(predictions[vid])

        # 2) vid "0101" → 폴더명 "01-01", JSON 경로 생성
        # PURE 전용 매핑
        folder = vid[:2] + '-' + vid[2:]
        json_path = os.path.join(raw_root, folder, f"{folder}.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Cannot find raw JSON: {json_path}")
        raw_bvps = PURELoader.read_wave(json_path)  # 원본 BVP 읽기
        lbl_arr = BaseLoader.resample_ppg(raw_bvps, pred_arr.shape[0])

        # 원본 배열 저장
        np.save(os.path.join(save_dir, f"prediction_{vid}.npy"), pred_arr)
        np.save(os.path.join(save_dir, f"label_{vid}.npy"),      lbl_arr)

        length = pred_arr.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            w = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            window = min(w, length)
        else:
            window = length

        diff_flag = (config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized")
        method    = config.INFERENCE.EVALUATION_METHOD.lower()
        hr_method = 'Peak' if 'peak' in method else 'FFT'

        for start in range(0, length, window):
            pw = pred_arr[start:start+window]
            lw = lbl_arr [start:start+window]
            if pw.shape[0] < 9:
                print(f"Ignored window of length {pw.shape[0]} < 9")
                continue

            gt_hr, pr_hr, snr = calculate_metric_per_video(
                pw, lw, diff_flag=diff_flag,
                fs=config.TEST.DATA.FS, hr_method=hr_method
            )
            snr_list.append(snr)
            if hr_method == 'FFT':
                gt_hr_fft_all.append(gt_hr)
                predict_hr_fft_all.append(pr_hr)
            else:
                gt_hr_peak_all.append(gt_hr)
                predict_hr_peak_all.append(pr_hr)

    # 파일명 ID 결정
    if config.TOOLBOX_MODE == 'train_and_test':
        fid = config.TRAIN.MODEL_FILE_NAME
    else:
        root = os.path.basename(config.INFERENCE.MODEL_PATH).split('.pth')[0]
        fid  = f"{root}_{config.TEST.DATA.DATASET}"

    # FFT 결과 처리
    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        arr_g   = np.array(gt_hr_fft_all)
        arr_p   = np.array(predict_hr_fft_all)
        arr_snr = np.array(snr_list)
        n       = len(arr_p)

        for m in config.TEST.METRICS:
            if m == "MAE":
                v  = np.mean(np.abs(arr_p - arr_g))
                se = np.std(np.abs(arr_p - arr_g)) / np.sqrt(n)
                print(f"FFT MAE: {v:.3f} ± {se:.3f}")
            elif m == "RMSE":
                errs = arr_p - arr_g
                v     = np.sqrt(np.mean(errs**2))
                se    = np.std(errs) / np.sqrt(n)
                print(f"FFT RMSE: {v:.3f} ± {se:.3f}")
            elif m == "MAPE":
                v  = np.mean(np.abs((arr_p - arr_g)/arr_g))*100
                se = np.std(np.abs((arr_p - arr_g)/arr_g))*100/np.sqrt(n)
                print(f"FFT MAPE: {v:.2f}% ± {se:.2f}%")
            elif m == "Pearson":
                if n < 3 or np.std(arr_p)==0 or np.std(arr_g)==0:
                    print("FFT Pearson: skipped (insufficient samples or zero variance)")
                else:
                    corr = np.corrcoef(arr_p, arr_g)[0,1]
                    se   = np.sqrt((1-corr**2)/(n-2))
                    print(f"FFT Pearson: {corr:.3f} ± {se:.3f}")
            elif m == "SNR":
                v  = np.mean(arr_snr)
                se = np.std(arr_snr)/np.sqrt(n)
                print(f"FFT SNR: {v:.3f} ± {se:.3f} dB")
            elif m == "BA":
                if n < 3:
                    print("⚠️ Skipped FFT BA plots: insufficient samples")
                else:
                    try:
                        # BA 전용 디렉터리 생성
                        ba_dir = os.path.join(save_dir, 'bland_altman_plots')
                        os.makedirs(ba_dir, exist_ok=True)

                        ba = BlandAltman(arr_g, arr_p, config, averaged=True)
                        scatter_path = os.path.join(ba_dir, f'{fid}_FFT_BA_Scatter.pdf')
                        diff_path = os.path.join(ba_dir, f'{fid}_FFT_BA_Diff.pdf')

                        ba.scatter_plot(
                            x_label='GT HR [bpm]', y_label='rPPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            file_name=scatter_path
                        )
                        ba.difference_plot(
                            x_label='Difference [bpm]', y_label='Average [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            file_name=diff_path
                        )
                    except np.linalg.LinAlgError as e:
                        print(f"⚠️ Skipped FFT BA plots: {e}")

    # Peak detection 결과 처리
    else:
        arr_g   = np.array(gt_hr_peak_all)
        arr_p   = np.array(predict_hr_peak_all)
        arr_snr = np.array(snr_list)
        n       = len(arr_p)

        for m in config.TEST.METRICS:
            if m == "MAE":
                v  = np.mean(np.abs(arr_p - arr_g))
                se = np.std(np.abs(arr_p - arr_g)) / np.sqrt(n)
                print(f"Peak MAE: {v:.3f} ± {se:.3f}")
            elif m == "RMSE":
                errs = arr_p - arr_g
                v     = np.sqrt(np.mean(errs**2))
                se    = np.std(errs) / np.sqrt(n)
                print(f"Peak RMSE: {v:.3f} ± {se:.3f}")
            elif m == "MAPE":
                v  = np.mean(np.abs((arr_p - arr_g)/arr_g))*100
                se = np.std(np.abs((arr_p - arr_g)/arr_g))*100/np.sqrt(n)
                print(f"Peak MAPE: {v:.2f}% ± {se:.2f}%")
            elif m == "Pearson":
                if n < 3 or np.std(arr_p)==0 or np.std(arr_g)==0:
                    print("Peak Pearson: skipped (insufficient samples or zero variance)")
                else:
                    corr = np.corrcoef(arr_p, arr_g)[0,1]
                    se   = np.sqrt((1-corr**2)/(n-2))
                    print(f"Peak Pearson: {corr:.3f} ± {se:.3f}")
            elif m == "SNR":
                v  = np.mean(arr_snr)
                se = np.std(arr_snr)/np.sqrt(n)
                print(f"Peak SNR: {v:.3f} ± {se:.3f} dB")
            elif m == "BA":
                if n < 3:
                    print("⚠️ Skipped Peak BA plots: insufficient samples")
                else:
                    try:
                        ba_dir = os.path.join(save_dir, 'bland_altman_plots')
                        os.makedirs(ba_dir, exist_ok=True)

                        ba = BlandAltman(arr_g, arr_p, config, averaged=True)
                        scatter_path = os.path.join(ba_dir, f'{fid}_Peak_BA_Scatter.pdf')
                        diff_path = os.path.join(ba_dir, f'{fid}_Peak_BA_Diff.pdf')

                        ba.scatter_plot(
                            x_label='GT HR [bpm]', y_label='rPPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            file_name=scatter_path
                        )
                        ba.difference_plot(
                            x_label='Difference [bpm]', y_label='Average [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            file_name=diff_path
                        )
                    except np.linalg.LinAlgError as e:
                        print(f"⚠️ Skipped Peak BA plots: {e}")

#
# # ----------
#     # FFT 결과 처리
#     if config.INFERENCE.EVALUATION_METHOD == "FFT":
#         arr_g   = np.array(gt_hr_fft_all)
#         arr_p   = np.array(predict_hr_fft_all)
#         arr_snr = np.array(snr_list)
#         n       = len(arr_p)
#
#         for m in config.TEST.METRICS:
#             if m == "MAE":
#                 v  = np.mean(np.abs(arr_p - arr_g))
#                 se = np.std(np.abs(arr_p - arr_g)) / np.sqrt(n)
#                 print(f"FFT MAE: {v:.3f} ± {se:.3f}")
#             elif m == "RMSE":
#                 v  = np.sqrt(np.mean((arr_p - arr_g)**2))
#                 se = np.std((arr_p - arr_g)**2) / np.sqrt(n)
#                 print(f"FFT RMSE: {v:.3f} ± {se:.3f}")
#             elif m == "MAPE":
#                 v  = np.mean(np.abs((arr_p - arr_g)/arr_g))*100
#                 se = np.std(np.abs((arr_p - arr_g)/arr_g))*100/np.sqrt(n)
#                 print(f"FFT MAPE: {v:.2f}% ± {se:.2f}%")
#             elif m == "Pearson":
#                 corr = np.corrcoef(arr_p, arr_g)[0,1]
#                 se   = np.sqrt((1-corr**2)/(n-2))
#                 print(f"FFT Pearson: {corr:.3f} ± {se:.3f}")
#             elif m == "SNR":
#                 v  = np.mean(arr_snr)
#                 se = np.std(arr_snr)/np.sqrt(n)
#                 print(f"FFT SNR: {v:.3f} ± {se:.3f} dB")
#             elif m == "BA":
#                 ba = BlandAltman(arr_g, arr_p, config, averaged=True)
#                 try:
#                     ba.scatter_plot(
#                         x_label='GT HR [bpm]', y_label='rPPG HR [bpm]',
#                         show_legend=True, figure_size=(5,5),
#                         the_title=f'{fid}_FFT_BA_Scatter',
#                         file_name=os.path.join(save_dir, f'{fid}_FFT_BA_Scatter.pdf')
#                     )
#                     ba.difference_plot(
#                         x_label='Difference [bpm]', y_label='Average [bpm]',
#                         show_legend=True, figure_size=(5,5),
#                         the_title=f'{fid}_FFT_BA_Diff',
#                         file_name=os.path.join(save_dir, f'{fid}_FFT_BA_Diff.pdf')
#                     )
#                 except np.linalg.LinAlgError as e:
#                     print(f"⚠️ Skipped FFT BA plots: {e}")
#
#     # Peak detection 결과 처리
#     else:
#         arr_g   = np.array(gt_hr_peak_all)
#         arr_p   = np.array(predict_hr_peak_all)
#         arr_snr = np.array(snr_list)
#         n       = len(arr_p)
#
#         for m in config.TEST.METRICS:
#             if m == "MAE":
#                 v  = np.mean(np.abs(arr_p - arr_g))
#                 se = np.std(np.abs(arr_p - arr_g)) / np.sqrt(n)
#                 print(f"Peak MAE: {v:.3f} ± {se:.3f}")
#             elif m == "RMSE":
#                 v  = np.sqrt(np.mean((arr_p - arr_g)**2))
#                 se = np.std((arr_p - arr_g)**2) / np.sqrt(n)
#                 print(f"Peak RMSE: {v:.3f} ± {se:.3f}")
#             elif m == "MAPE":
#                 v  = np.mean(np.abs((arr_p - arr_g)/arr_g))*100
#                 se = np.std(np.abs((arr_p - arr_g)/arr_g))*100/np.sqrt(n)
#                 print(f"Peak MAPE: {v:.2f}% ± {se:.2f}%")
#             elif m == "Pearson":
#                 corr = np.corrcoef(arr_p, arr_g)[0,1]
#                 se   = np.sqrt((1-corr**2)/(n-2))
#                 print(f"Peak Pearson: {corr:.3f} ± {se:.3f}")
#             elif m == "SNR":
#                 v  = np.mean(arr_snr)
#                 se = np.std(arr_snr)/np.sqrt(n)
#                 print(f"Peak SNR: {v:.3f} ± {se:.3f} dB")
#             elif m == "BA":
#                 ba = BlandAltman(arr_g, arr_p, config, averaged=True)
#                 try:
#                     ba.scatter_plot(
#                         x_label='GT HR [bpm]', y_label='rPPG HR [bpm]',
#                         show_legend=True, figure_size=(5,5),
#                         the_title=f'{fid}_Peak_BA_Scatter',
#                         file_name=os.path.join(save_dir, f'{fid}_Peak_BA_Scatter.pdf')
#                     )
#                     ba.difference_plot(
#                         x_label='Difference [bpm]', y_label='Average [bpm]',
#                         show_legend=True, figure_size=(5,5),
#                         the_title=f'{fid}_Peak_BA_Diff',
#                         file_name=os.path.join(save_dir, f'{fid}_Peak_BA_Diff.pdf')
#                     )
#                 except np.linalg.LinAlgError as e:
#                     print(f"⚠️ Skipped Peak BA plots: {e}")
