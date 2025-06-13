# File: dataset/data_loader/Base_Loader.py

import glob
import logging

logger = logging.getLogger(__name__)

import os
from math import ceil
from multiprocessing import Process, Manager

import cv2
import numpy as np
import pandas as pd
from scipy import signal
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.post_process import calculate_hr
from unsupervised_methods.methods.POS_WANG import POS_WANG_zscore

from dataset.preprocessing.normalization import (standardize_frames,
                                                 diff_normalize_frames,
                                                 standardize_labels,
                                                 diff_normalize_labels)
from config import _C as config


def apply_bandpass(signal_1d: np.ndarray, lowcut: float, highcut: float, fs: float, order: None = None) -> np.ndarray:
    nyq = 0.5 * fs
    # 1) config 에서 ORDER 읽어오기 (None 이면 YAML 값 사용)
    if order is None:
        order = config.TEST.DATA.PREPROCESS.BANDPASS.ORDER
    Wn = [lowcut / nyq, highcut / nyq]
    b, a = signal.butter(order, Wn, btype='bandpass')
    return signal.filtfilt(b, a, signal_1d)


class BaseLoader(Dataset):
    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument("--cached_path", default=None, type=str)
        parser.add_argument("--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, dataset_name, raw_data_path, config_data):
        """
        Inits dataloader with lists of files.
        Args:
            dataset_name(str): name of the dataloader.
            raw_data_path(string): path to the folder containing all raw data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.inputs = []
        self.labels = []
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        self.config_data = config_data

        # PREPROCESS 섹션 검증
        pre = config_data.PREPROCESS
        assert hasattr(pre, 'DATA_AUG'), "Missing DATA_AUG in DATA.PREPROCESS"
        assert hasattr(pre, 'USE_PSEUDO_PPG_LABEL'), "Missing USE_PSEUDO_PPG_LABEL in DATA.PREPROCESS"

        # train/val split 비율 검증
        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN >= 0 and config_data.END <= 1)

        if config_data.DO_PREPROCESS:
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
            self.preprocess_dataset(self.raw_data_dirs,
                                    config_data.PREPROCESS,
                                    config_data.BEGIN,
                                    config_data.END)

        else:
            if not os.path.exists(self.cached_path):
                raise ValueError(self.dataset_name,
                                 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
                self.build_file_list_retroactive(self.raw_data_dirs,
                                                 config_data.BEGIN,
                                                 config_data.END)
                print('File list generated.', end='\n\n')
            self.load_preprocessed_data()

        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Returns a clip of video and its label plus precomputed GT HR.
        """
        data_file = self.inputs[index]
        label_file = self.labels[index]

        data = np.load(data_file)  # 저장 시 NDHWC (T, H, W, C)
        label = np.load(label_file)

        # 저장 시 NDHWC → 원하는 포맷으로 변환 (예: NDCHW)
        if self.data_format == 'NDCHW':
            # data: (T, H, W, C) → (T, C, H, W)
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            # data: (T, H, W, C) → (C, T, H, W)
            data = np.transpose(data, (3, 0, 1, 2))
        # 기본값 NDHWC 그대로 두면 (T, H, W, C)

        data = data.astype(np.float32)
        label = label.astype(np.float32)

        # hr 경로: “_input” → “_hr”
        hr_file = data_file.replace("_input", "_hr")
        hr_gt = float(np.load(hr_file))

        # filename 및 chunk_id 추출
        item_name = os.path.basename(data_file)
        split_idx = item_name.rindex('_')
        filename = item_name[:split_idx]
        chunk_id = item_name[split_idx + 6:].split('.')[0]

        # mu/σ 복원 정보 로드
        mu_sigma_file = data_file.replace("_input", "_mu_sigma").replace(".npy", ".npz")
        ms = np.load(mu_sigma_file)
        mu, sigma = float(ms["mu"]), float(ms["sigma"])

        return data, label, mu, sigma, hr_gt, filename, chunk_id

    def get_raw_data(self, raw_data_path):
        """Returns a list of raw-data directories (override 필요)."""
        raise NotImplementedError("'get_raw_data' Not Implemented")

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs based on split (override 필요)."""
        raise NotImplementedError("'split_raw_data' Not Implemented")

    def read_npy_video(self, video_file):
        """Reads a video file in numpy format, returns frames as uint8 RGB."""
        frames = np.load(video_file[0])
        if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
            processed_frames = [frame.astype(np.uint8)[..., :3] for frame in frames]
        elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
            processed_frames = [(np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
        else:
            raise Exception(f'Loaded frames are of an incorrect type or range of values! '
                            f'Received frames of type {frames.dtype} and range {np.min(frames)} to {np.max(frames)}.')
        return np.asarray(processed_frames)

    def generate_pos_pseudo_labels(self, frames, fs=30):
        """
        Generated POS-based PPG Pseudo Labels For Training
        """
        bvp_raw = POS_WANG_zscore(frames, fs)  # shape = (T,)
        analytic_signal = signal.hilbert(bvp_raw)  # shape = (T,)
        amplitude_envelope = np.abs(analytic_signal) + 1e-8
        env_norm_bvp = bvp_raw / amplitude_envelope
        return np.array(env_norm_bvp)  # shape = (T,)

    def crop_face_resize(self, frames, cfg_crop):
        """
        frames: (T,H,W,3) ndarray
        cfg_crop: CfgNode (config.CROP)
        """
        logger.debug(f"[Facemesh] CropFaceResize 시작 (T={frames.shape[0]})")

        # ─────────────────────────────────────────────────────────────
        # STEP 1. FaceMesh ⇢ Tracker ⇢ Cheek+Forehead Mask 생성
        # ─────────────────────────────────────────────────────────────
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        mesh = mp_face_mesh.FaceMesh(
            static_image_mode=not cfg_crop.DO_DYNAMIC_DETECTION,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        T = frames.shape[0]
        det_freq = cfg_crop.DYNAMIC_DETECTION_FREQUENCY if cfg_crop.DO_DYNAMIC_DETECTION else T
        face_boxes = []
        masks = []

        # 1) detection 주기마다 landmark 추출 → box, mask 저장
        for det_idx in range(ceil(T / det_freq)):
            idx = min(det_idx * det_freq, T - 1)
            rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            H, W = frames.shape[1], frames.shape[2]

            # a) landmark → mask
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                pts = np.array([[int(p.x * W), int(p.y * H)] for p in lm], np.int32)
                f_idx = [10, 338, 297, 332, 284, 251]
                lc_idx, rc_idx = [234, 93, 132, 58, 172, 136], [454, 323, 361, 288, 397, 375]
                m = np.zeros((H, W), np.uint8)
                cv2.fillPoly(m, [pts[f_idx]], 1)
                cv2.fillPoly(m, [pts[lc_idx]], 1)
                cv2.fillPoly(m, [pts[rc_idx]], 1)
            else:
                m = np.ones((H, W), np.uint8)
            masks.append(m)

            # b) box 계산
            if res.multi_face_landmarks and cfg_crop.USE_HYBRID:
                xs = np.array([p.x for p in lm]) * W
                ys = np.array([p.y for p in lm]) * H
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                box = [x1, y1, x2 - x1, y2 - y1]
            else:
                box = [0, 0, W, H]
            face_boxes.append(box)

        mesh.close()

        # USE_MEDIAN_FACE_BOX 적용
        if cfg_crop.USE_MEDIAN_FACE_BOX:
            mb = np.median(np.stack(face_boxes), axis=0).astype(int)
            face_boxes = [mb] * len(face_boxes)

        # tracker 초기화 (Hybrid 모드)
        tracker = None
        if cfg_crop.USE_HYBRID:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frames[0], tuple(face_boxes[0]))

        # ─────────────────────────────────────────────────────────────
        # STEP 2. Crop → Resize → Mask 적용
        # ─────────────────────────────────────────────────────────────
        out = np.zeros((T, cfg_crop.HEIGHT, cfg_crop.WIDTH, 3), frames.dtype)
        for t in range(T):
            # 1) box 업데이트
            det_idx = min(t // det_freq, len(face_boxes) - 1)
            if cfg_crop.USE_HYBRID:
                if t % det_freq == 0:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frames[t], tuple(face_boxes[det_idx]))
                ok, b = tracker.update(frames[t])
                if ok:
                    box = list(map(int, b))
                else:
                    box = face_boxes[det_idx]
            else:
                box = face_boxes[det_idx]

            x, y, wb, hb = box
            roi = frames[t][y:y + hb, x:x + wb]
            face = cv2.resize(roi, (cfg_crop.WIDTH, cfg_crop.HEIGHT))

            # 2) mask 업데이트
            m = masks[det_idx]
            m_crop = m[y:y + hb, x:x + wb]
            m_res = cv2.resize(m_crop, (cfg_crop.WIDTH, cfg_crop.HEIGHT),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            face[~m_res] = 0
            out[t] = face

        logger.debug(f"[Facemesh] CropFaceResize 완료 (T={T}, "f"mode={'Hybrid' if cfg_crop.USE_HYBRID else 'Dynamic'})")
        return out

    def chunk(self, frames, bvps, chunk_length):
        """
        Chunk the data into small chunks.

        Args:
            frames(np.array): video frames.
            bvps(np.array): 1D PPG 라벨.
            chunk_length(int): length of each chunk.
        Returns:
            frames_clips(np.ndarray): 크롭+리사이즈된 얼굴 프레임의 청크 배열, shape = (N_clips, chunk_length, H, W, C)
            bvp_clips(np.ndarray): 1D PPG 청크 배열, shape = (N_clips, chunk_length)
        """
        total_len = frames.shape[0]
        clips_f = []
        clips_b = []

        for start in range(0, total_len, chunk_length):
            end = start + chunk_length
            if end <= total_len:
                clip_f = frames[start:end]
                clip_b = bvps[start:end]
            else:
                # 마지막 꼬리: 부족한 부분 edge-padding
                clip_f = frames[start:]
                clip_b = bvps[start:]
                pad = chunk_length - clip_f.shape[0]
                clip_f = np.pad(clip_f, ((0, pad), (0, 0), (0, 0), (0, 0)), mode='edge')
                clip_b = np.pad(clip_b, (0, pad), mode='edge')
            clips_f.append(clip_f)
            clips_b.append(clip_b)

            if end >= total_len:
                break
        return np.array(clips_f, dtype=object), np.array(clips_b, dtype=object)

    def save(self, frames_clips, bvps_clips, filename):
        """
        Save all the chunked data (싱글 프로세스용).

        Args:
            frames_clips(np.ndarray): 얼굴 프레임 청크들
            bvps_clips(np.ndarray): BVP (PPG) 청크들
            filename(str): 원본 영상 인덱스 (예: “0101”)
        Returns:
            count(int): 저장된 청크 개수
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + f"{filename}_input{count}.npy"
            label_path_name = self.cached_path + os.sep + f"{filename}_label{count}.npy"
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return count

    def save_multi_process(self, frames_clips, bvps_clips, filename, mu=None, sigma=None):
        os.makedirs(self.cached_path, exist_ok=True)
        input_list, label_list = [], []
        mu_val, sig_val = (mu or 0.0), (sigma or 1.0)

        for idx, (f_c, b_c) in enumerate(zip(frames_clips, bvps_clips)):
            inp = os.path.join(self.cached_path, f"{filename}_input{idx}.npy")
            lbl = os.path.join(self.cached_path, f"{filename}_label{idx}.npy")
            np.save(inp, f_c)
            np.save(lbl, b_c)
            input_list.append(inp)
            label_list.append(lbl)

            # mu/sigma 저장
            ms = os.path.join(self.cached_path, f"{filename}_mu_sigma{idx}.npz")
            np.savez(ms, mu=mu_val, sigma=sig_val)

            # 미리 계산된 HR 저장 (normalized 신호 → denorm → HR 계산)
            if sigma is not None and mu is not None:
                raw = b_c * sigma + mu
            else:
                raw = b_c
            pre = self.config_data.PREPROCESS
            low = pre.BANDPASS.LOW_CUT
            high = pre.BANDPASS.HIGH_CUT
            min_hr, max_hr = low * 60, high * 60
            # 실제 LABEL_TYPE에 맞춰 diff_flag 동적 설정
            diff_flag = ("DiffNormalized" in pre.LABEL_TYPE) if hasattr(pre, 'LABEL_TYPE') else False
            _, hr = calculate_hr(
                raw, raw,
                fs=self.config_data.FS,
                diff_flag=diff_flag,
                min_hr=min_hr,
                max_hr=max_hr
            )
            hrp = os.path.join(self.cached_path, f"{filename}_hr{idx}.npy")
            np.save(hrp, np.array(hr, dtype=np.float32))

        return input_list, label_list

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=2):
        """
        multi_process_quota=8
        Allocate dataset preprocessing across multiple processes.

        Args:
            data_dirs(list): raw data 디렉토리 리스트
            config_preprocess(CfgNode): preprocessing 설정
            multi_process_quota(int): 동시에 사용할 프로세스 최대 개수
        Returns:
            file_list_dict(dict): {process_idx: [input_path1, input_path2, ...], ...}
        """
        file_num = len(data_dirs)
        choose_range = range(0, file_num)
        pbar = tqdm(list(choose_range))

        manager = Manager()
        file_list_dict = manager.dict()
        p_list = []
        running_num = 0

        for i in choose_range:
            process_flag = True
            while process_flag:
                if running_num < multi_process_quota:
                    p = Process(target=self.preprocess_dataset_subprocess,
                                args=(data_dirs, config_preprocess, i, file_list_dict))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()
        file_list = dict(file_list_dict)
        manager.shutdown()
        return file_list

    def build_file_list(self, file_list_dict):
        """
        Build a list of files used by the dataloader for split(train/val/test).
        그리고 CSV로 저장.

        Args:
            file_list_dict(dict): {process_idx: [input_path, ...], ...}
        """
        file_list = []
        print(file_list_dict)
        for process_num, file_paths in file_list_dict.items():
            file_list += file_paths

        if not file_list:
            raise ValueError(self.dataset_name, 'No files in file list')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """
        If a file list has not already been generated, build it retroactively.

        Args:
            data_dirs(list): raw data 디렉토리 리스트
            begin(float): split 시작 비율
            end(float): split 끝 비율
        """
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)
        filename_list = [d['index'] for d in data_dirs_subset]
        filename_list = list(set(filename_list))

        file_list = []
        for fname in filename_list:
            processed_file_data = glob.glob(self.cached_path + os.sep + f"{fname}_input*.npy")
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.dataset_name,
                             'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)

    def load_preprocessed_data(self):
        """
        Loads the preprocessed data listed in the file list (.csv).

        Returns:
            None (inputs, labels 리스트를 채움)
        """
        file_list_df = pd.read_csv(self.file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)
        labels = [inp.replace("input", "label") for inp in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    def normalize_and_chunk(self, raw_frames: np.ndarray, raw_bvp: np.ndarray, cfg_pre):
        if raw_bvp.shape[0] != raw_frames.shape[0]:
            raw_bvp = self.resample_ppg(raw_bvp, raw_frames.shape[0])
        # 1) 프레임 정규화: 설정이 리스트든 문자열이든 모두 안전하게 처리
        dtypes = cfg_pre.DATA_TYPE if isinstance(cfg_pre.DATA_TYPE, (list, tuple)) else [cfg_pre.DATA_TYPE]
        if "DiffNormalized" in dtypes:
            frames = diff_normalize_frames(raw_frames)
        elif "Standardized" in dtypes:
            frames = standardize_frames(raw_frames)
        else:
            frames = raw_frames

        # 2) 라벨 정규화
        ltypes = cfg_pre.LABEL_TYPE if isinstance(cfg_pre.LABEL_TYPE, (list, tuple)) else [cfg_pre.LABEL_TYPE]
        if "DiffNormalized" in ltypes:
            lbl, mu, sigma = diff_normalize_labels(raw_bvp)
        elif "Standardized" in ltypes:
            lbl, mu, sigma = standardize_labels(raw_bvp)
        else:
            lbl, mu, sigma = raw_bvp, None, None

        # 3) 청크 분할
        clips_f, clips_b = self.chunk(frames, lbl, cfg_pre.CHUNK_LENGTH)
        return clips_f, clips_b, mu, sigma

    @staticmethod
    def resample_ppg(input_signal, target_length):
        """
        Resample a 1D PPG sequence into a specific length by linear interpolation.

        Args:
            input_signal(np.ndarray): shape = (T_raw,)
            target_length(int): 원하는 타겟 길이 (예: 원본 프레임 수)
        Returns:
            resampled(np.ndarray): shape = (target_length,)
        """
        return np.interp(
            np.linspace(1, input_signal.shape[0], target_length),
            np.linspace(1, input_signal.shape[0], input_signal.shape[0]),
            input_signal
        )

    @staticmethod
    def simple_roi_bandpass(frames, masks, fs: float, lowcut: float, highcut: float, order: int = None):
        """
        Crop+Resize된 얼굴 시퀀스와 동일 길이의 boolean masks 리스트(masks)에서
        Green 채널을 평균 내어 1차원 시계열을 얻고,
        YAML 설정에 맞춘 Band-Pass 필터를 적용하여 노이즈를 억제한 1D PPG 신호를 반환.

        Args:
            frames (np.ndarray): (T, H, W, 3) 크롭+리사이즈된 얼굴 시퀀스 (RGB)
            fs (float): 샘플링 레이트 (예: 30.0)
            lowcut (float): Band-Pass 하한 (Hz)
            highcut (float): Band-Pass 상한 (Hz)
            order (int): Butterworth 필터 차수
        Returns:
            filtered (np.ndarray): shape = (T,), Band-Pass 후 1D PPG 시그널
        """
        # ─── 1) 조명 보정: 프레임별 전체 밝기 평준화 ─────────────────────────
        frames = frames.astype(np.float32)
        lum = frames.mean(axis=(1, 2, 3), keepdims=True)
        frames = frames / (lum + 1e-6)

        # ─── 2) Skin-Only Mask 적용 (Mediapipe FaceMesh landmarks 재사용) ──────
        # 이미 crop_face_resize 에서 FaceMesh landmark를 썼다면,
        # frames 속성으로 landmark를 붙여 두었다고 가정합니다.
        # (만약 없으면, frames[...,1] 전체 평균으로 fallback)

        green_ts = []
        for frame, mask in zip(frames, masks):
            vals = frame[..., 1][mask] if mask is not None else frame[..., 1].ravel()
            green_ts.append(vals.mean())

        green_ts = np.array(green_ts, dtype=np.float32)
        # Zero-mean 제거
        green_ts = green_ts - np.mean(green_ts)

        # ─── 2) Butterworth Band-Pass 필터 설계 ─────────────────────────────────────
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # order가 None이면 config에 정의된 기본값 사용
        if order is None:
            from config import _C as config
            order = config.TEST.DATA.PREPROCESS.BANDPASS.ORDER
        # 항상 필터 계수(b, a) 계산
        b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')

        # 3) 필터 적용 (padlen을 충분히 주어 edge effect 최소화)
        padlen = 3 * (max(len(a), len(b)) - 1)
        filtered = signal.filtfilt(b, a, green_ts, padlen=padlen)

        return filtered  # shape = (T,)
