# File: dataset/data_loader/PURE_Loader.py

import glob
import json
import logging
import os

import cv2
import numpy as np

from dataset.data_loader.Base_Loader import BaseLoader

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
        name(str): name of the dataloader
        config_data(CfgNode): data settings(ref:config.py)
        """
        super().__init__(name, raw_data_path, config_data)

    def get_raw_data(self, raw_data_path):
        """
        Returns data directories under the path (for PURE dataset).

        raw_data_path 하위에 “XX-YY” 형태의 폴더들이 있어야 함.
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

        begin, end는 각각 0~1 사이의 비율.
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

        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """
        Invoked by preprocess_dataset for multi-process preprocessing of a single directory.

        1) Read PNG frames → frames (RGB)
        2) FaceMesh Crop + Resize → frames_face (shape = (T', 128, 128, 3))
        3) simple_roi_bandpass(frames_face) → bvp_signal (1D PPG, shape = (T',))
        4) resample bvp_signal → target_length = T'
        5) self.preprocess(frames_face, bvp_signal) → frames_clips, bvp_clips
        6) save_multi_process(frames_clips, bvp_clips, saved_filename)
        """
        video_dir = data_dirs[i]['path']
        filename = os.path.basename(video_dir)
        saved_filename = data_dirs[i]['index']

        logger.info(f"[Subprocess {i}] Starting preprocessing for {filename}")

        try:
            # ── 1) Read all PNG frames from directory ───────────────────────────
            all_png = sorted(glob.glob(os.path.join(video_dir, '*.png')))
            frames = []
            for png_path in all_png:
                img = cv2.imread(png_path)                   # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # RGB
                frames.append(img)
            frames = np.asarray(frames)  # shape = (T, H, W, 3)

            # ── 2) FaceMesh 기반 Crop + Resize → frames_face ───────────────────
            frames_face = self.crop_face_resize(
                frames,
                config_preprocess.CROP_FACE.DO_CROP_FACE,
                config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
                config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
                config_preprocess.RESIZE.W,
                config_preprocess.RESIZE.H
            )
            # frames_face shape = (T', 128, 128, 3)

            # ── 3) ROI Band-Pass → raw 1D PPG signal 생성 ─────────────────────
            fs = self.config_data.FS  # 예: 30
            bvp_signal = BaseLoader.simple_roi_bandpass(frames_face, fs)  # shape = (T',)

            # ── 4) Resample PPG to match frame count ──────────────────────────
            target_length = frames_face.shape[0]
            bvp_signal = BaseLoader.resample_ppg(bvp_signal, target_length)
            # bvp_signal shape = (T',) (이미 band-pass & zero-mean 처리됨)

            # ── 5) 클립 단위로 Chunk → Save ─────────────────────────────────
            frames_clips, bvp_clips = self.preprocess(frames_face,
                                                      bvp_signal,
                                                      config_preprocess)
            input_name_list, label_name_list = self.save_multi_process(
                frames_clips, bvp_clips, saved_filename
            )
            file_list_dict[i] = input_name_list

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

    @staticmethod
    def read_wave(bvp_file):
        """
        Reads a bvp signal JSON file (PURE의 /FullPackage 화일 구조를 가정).
        """
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)
