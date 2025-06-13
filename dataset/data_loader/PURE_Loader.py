# File: dataset/data_loader/PURE_Loader.py

import glob
import json
import logging
import os
import time

import cv2
import numpy as np

from dataset.data_loader.Base_Loader import BaseLoader
from evaluation.post_process import calculate_hr

from dataset.preprocessing.normalization import (standardize_frames,
                                                 diff_normalize_frames,
                                                 standardize_labels,
                                                 diff_normalize_labels)

logger = logging.getLogger(__name__)


class PURELoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, name, raw_data_path, config_data):
        """
        Initializes an PURE dataloader.
        기본 폴더 구조:
            rppg_dataset/PURE/
            ├── 01-01/
            │   ├── img0001.png, img0002.png, ...
            │   └── 01-01.json
            ├── 01-02/
            │   ├── img0001.png, img0002.png, ...
            │   └── 01-02.json
            ...
        """
        super().__init__(name, raw_data_path, config_data)

    def preprocess(self, frames_face: np.ndarray, bvp_signal: np.ndarray, config_preprocess):
        """
        - frames_face: crop+resize된 얼굴 프레임 시퀀스, shape = (T, H, W, 3)
        - bvp_signal: simple_roi_bandpass로 얻은 1D PPG 시퀀스, shape = (T,)
        - config_preprocess: 데이터 전처리 관련 설정(CfgNode)

        Returns:
            frames_clips (np.ndarray): shape = (N_clips, chunk_length, H, W, C)
            bvp_clips    (np.ndarray): shape = (N_clips, chunk_length)
        """
        # ── 1) DATA_TYPE에 따른 프레임 정규화 ─────────────────────────
        if 'DiffNormalized' in getattr(config_preprocess, 'DATA_TYPE', ''):
            frames_face = diff_normalize_frames(frames_face)
        elif 'Standardized' in getattr(config_preprocess, 'DATA_TYPE', ''):
            frames_face = standardize_frames(frames_face)

        # LABEL_TYPE 분기: 리스트/문자열 모두 처리
        ltypes = config_preprocess.LABEL_TYPE if isinstance(config_preprocess.LABEL_TYPE, (list, tuple)) else [
            config_preprocess.LABEL_TYPE]
        if "DiffNormalized" in ltypes:
            bvp_signal, mu, sigma = diff_normalize_labels(bvp_signal)
        elif "Standardized" in ltypes:
            bvp_signal, mu, sigma = standardize_labels(bvp_signal)
        else:
            # 예외 시 기본값
            mu, sigma = 0.0, 1.0

        # ── 3) chunking ─────────────────────────
        chunk_length = config_preprocess.CHUNK_LENGTH
        frames_clips, bvp_clips = self.chunk(frames_face, bvp_signal, chunk_length)

        # μ·σ를 함께 반환
        return frames_clips, bvp_clips, mu, sigma

    def get_raw_data(self, raw_data_path):
        """
        Returns data directories under the path (for PURE dataset).
        """
        data_dirs = glob.glob(raw_data_path + os.sep + "*-*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = []
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('-', '')
            index = subject_trail_val
            subject = int(subject_trail_val[0:2])
            dirs.append({"index": index, "path": data_dir, "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """
        Returns a subset of data dirs, split with begin and end values,
        and ensures no overlapping subjects between splits.
        """
        if begin == 0 and end == 1:
            return data_dirs

        data_info = {}
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            if subject not in data_info:
                data_info[subject] = []
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = sorted(list(data_info.keys()))
        num_subjs = len(subj_list)

        subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files

        return data_dirs_new

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """
        Parses and preprocesses all the raw data based on split.

        - multi_process_quota를 1로 낮춰 디스크 I/O 경합을 최소화합니다.
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)
        file_list_dict = self.multi_process_manager(
            data_dirs_split,
            config_preprocess,
            multi_process_quota=1  # ▶ 한 번에 프로세스 1개만 I/O 사용
        )
        # ── DEBUG: multi_process_manager 결과 확인
        if 0 in file_list_dict:
            paths0 = file_list_dict[0]
            print(f"[DEBUG] Process 0 (첫 번째 디렉토리) 저장된 clip 수: {len(paths0)}, 예시: {paths0[:3]} …")
        self.build_file_list(file_list_dict)
        self.load_preprocessed_data()
        print("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """
        Invoked by preprocess_dataset for multiprocess preprocessing of 한 개의 디렉토리.
        # ── 1) frames.npy 캐시 확인 및 로드
        # ── 3) Forehead+Cheek 영역 Mask 생성 (검출 실패 시 전체 ROI)
        # ── 4) ROI 적용 →
        # ── 5) JSON에서 GT PPG/HR 로드 → bvp_signal
        # ── 6) normalize + chunk → frames_clips, bvp_clips, mu, sigma
        # ── 7) save_multi_process → npy + μ·σ + HR 저장
        # ── 8) JSON GT HR 덮어쓰기
        """
        video_dir = data_dirs[i]['path']
        total_png = len(glob.glob(os.path.join(video_dir, '*.png')))
        print(f"[DEBUG] Subprocess {i} start: {video_dir} (총 PNG 개수: {total_png}장)")
        filename = os.path.basename(video_dir)
        saved_filename = data_dirs[i]['index']

        logger.info(f"[Subprocess {i}] Starting preprocessing for {filename}")

        try:
            # ── 1) frames.npy 캐시 확인 및 로드/생성
            frames_npy_path = os.path.join(video_dir, "frames.npy")
            t0 = time.time()
            if os.path.exists(frames_npy_path):
                # 이미 변환된 npy가 있으면 바로 불러오기
                frames = np.load(frames_npy_path)  # shape = (T, H, W, 3)
                t1 = time.time()
                print(f"[DEBUG_TIME] {video_dir}: load frames.npy: {t1 - t0:.2f} sec (T={frames.shape[0]})")
            else:
                # frames.npy가 없으면 PNG를 순차적으로 읽어서 한 번만 생성
                all_png = sorted(glob.glob(os.path.join(video_dir, '*.png')))
                frames_list = []
                for idx, png_path in enumerate(all_png):
                    # ▶ IMREAD_REDUCED_COLOR_2 옵션: 원본의 절반 크기로 바로 디코딩 (더 빠름)
                    img = cv2.imread(png_path, cv2.IMREAD_REDUCED_COLOR_2)
                    if img is None:
                        logger.warning(f"[WARN] {video_dir}: {png_path}을(를) 읽을 수 없어 스킵합니다.")
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames_list.append(img_rgb)
                    if idx % 500 == 0 and idx > 0:
                        print(f"[DEBUG_READ] {video_dir}: {idx}/{total_png} 프레임 로딩 완료")

                frames = np.stack(frames_list, axis=0)  # shape = (T, H_reduced, W_reduced, 3)
                t1 = time.time()
                print(f"[DEBUG_TIME] {video_dir}: Read from PNG: {t1 - t0:.2f} sec (T={frames.shape[0]})")

                # ▶ 한 번만 npy로 저장
                np.save(frames_npy_path, frames)
                print(f"[DEBUG] {video_dir}: frames.npy 생성됨")

            # ── 2) Crop+Mask+Resize (Base_Loader.crop_face_resize 에서 모두 처리)
            t0 = time.time()
            frames_face = self.crop_face_resize(frames, config_preprocess.CROP_FACE.DETECTION)
            t1 = time.time()
            print(f"[DEBUG_TIME] {video_dir}: Crop+Mask+Resize 완료: {t1 - t0:.2f} sec (T'={frames_face.shape[0]})")

            # ── 5) JSON에서 GT PPG/HR 로드 및 bvp_signal 결정
            t0 = time.time()
            json_path = os.path.join(video_dir, f"{filename}.json")
            with open(json_path, 'r') as f:
                meta = json.load(f)

            # GT PPG 파형 추출 ("/FullPackage" 아래 frame별 waveform)
            gt_ppg = np.asarray(
                [item['Value']['waveform'] for item in meta['/FullPackage']],
                dtype=np.float32
            )
            # GT HR: JSON에 HR 필드가 있으면 사용, 없으면 calculate_hr로 재계산
            if 'HR' in meta:
                gt_hr = float(meta['HR'])
            else:
                _, gt_hr = calculate_hr(
                    gt_ppg, gt_ppg,
                    fs=self.config_data.FS,
                    diff_flag=False,
                    min_hr=config_preprocess.BANDPASS.LOW_CUT * 60,
                    max_hr=config_preprocess.BANDPASS.HIGH_CUT * 60
                )

            # 이후 전처리 단계에 JSON GT PPG 사용
            bvp_signal = gt_ppg
            t1 = time.time()
            print(f"[DEBUG_TIME] {video_dir}: load JSON GT PPG: {t1 - t0:.2f} sec")

            # ── 6) normalize + chunk → frames_clips, bvp_clips, mu·σ 반환
            t0 = time.time()
            frames_clips, bvp_clips, mu, sigma = self.preprocess(frames_face, bvp_signal, config_preprocess)
            t1 = time.time()
            print(f"[DEBUG_TIME] {video_dir}: normalize_and_chunk: {t1 - t0:.2f} sec (clips={len(frames_clips)})")

            # ── 7) save_multi_process → 입력/레이블 npy, μ·σ 저장
            t0 = time.time()
            input_name_list, label_name_list = self.save_multi_process(frames_clips, bvp_clips, saved_filename, mu,
                                                                       sigma)
            t1 = time.time()
            print(
                f"[DEBUG_TIME] {video_dir}: save_multi_process: {t1 - t0:.2f} sec (저장된 clip 수={len(input_name_list)})")

            file_list_dict[i] = input_name_list

            # ── 8) 저장된 clip별로 JSON GT HR 덮어쓰기
            for idx in range(len(input_name_list)):
                hr_path = os.path.join(self.cached_path, f"{saved_filename}_hr{idx}.npy")
                np.save(hr_path, np.array(gt_hr, dtype=np.float32))

            logger.info(f"[Subprocess {i}] Finished {filename}: {len(input_name_list)} clips saved")

        except Exception as e:
            logger.error(f"[Subprocess {i}] Error processing '{filename}': {e}", exc_info=True)

    @staticmethod
    def read_video(video_file):
        """
        Reads a folder of PNG frames, returns np.ndarray of shape (T, H, W, 3) (RGB).
        """
        frames = []
        all_png = sorted(glob.glob(os.path.join(video_file, '*.png')))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)
