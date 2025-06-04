# File: dataset/data_loader/Base_Loader.py

import glob
import logging
import os
from math import ceil
from multiprocessing import Process, Manager

import cv2
import mediapipe as mp  # MediaPipe Face Mesh 사용
import numpy as np
import pandas as pd
from scipy import signal
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.post_process import calculate_hr
from unsupervised_methods.methods.POS_WANG import POS_WANG_zscore

logger = logging.getLogger(__name__)


def apply_bandpass(signal_1d: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    1D 신호에 Butterworth Band-Pass 필터를 적용하여 반환합니다.
    """
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='bandpass')
    # edge effect를 줄이기 위해 filtfilt 사용
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

        # 반드시 PREPROCESS 섹션이 존재해야 함
        assert hasattr(config_data, 'PREPROCESS'), "Missing PREPROCESS section in DATA config"
        pre = config_data.PREPROCESS
        assert hasattr(pre, 'DATA_AUG'), "Missing DATA_AUG in DATA.PREPROCESS"
        assert hasattr(pre, 'USE_PSEUDO_PPG_LABEL'), "Missing USE_PSEUDO_PPG_LABEL in DATA.PREPROCESS"

        # train/val split 비율 검증
        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN >= 0)
        assert (config_data.END <= 1)

        if config_data.DO_PREPROCESS:
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
            self.preprocess_dataset(self.raw_data_dirs,
                                    config_data.PREPROCESS,
                                    config_data.BEGIN,
                                    config_data.END)
        else:
            # 이미 전처리된 데이터가 존재해야 함
            if not os.path.exists(self.cached_path):
                logger.info(f"CACHED_PATH: {self.cached_path}")
                raise ValueError(self.dataset_name,
                                 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
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

        Returns:
            data: np.ndarray of shape (T, C, H, W) 혹은 (C, T, H, W) 등 지정된 포맷
            label: np.ndarray of shape (T,) 혹은 (T,1)
            hr_gt: scalar float (사전에 계산된 GT HR)
            filename: string (예: “01-01”)
            chunk_id: string (예: “0”)
        """
        import os
        import numpy as np

        data_file = self.inputs[index]
        label_file = self.labels[index]

        data = np.load(data_file)
        label = np.load(label_file)

        # 저장 시 NDHWC → 원하는 포맷으로 변환
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        # 기본값 NDHWC 그대로 두면 (T, H, W, C) 형태

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

        return data, label, hr_gt, filename, chunk_id

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

        Args:
            frames (np.ndarray): 크롭 & 리사이즈된 얼굴 프레임들, shape = (T, H, W, 3)
            fs (int or float): 비디오의 샘플링 레이트 (예: 30)

        Returns:
            env_norm_bvp (np.ndarray): Hilbert envelope normalization이 적용된
                                      POS PPG 신호, shape = (T,)
        """
        # ── 1) Min–Max 기반 POS_WANG 으로 raw BVP 생성 ─────────────────────────────────────────
        #    POS_WANG_zscore 함수 내부에서
        #    frames → RGB_ts → chrominance 연산 → detrend → bandpass 필터링 등을 수행.
        bvp_raw = POS_WANG_zscore(frames, fs)  # shape = (T,)

        # ── 2) Hilbert Envelope Normalization (선택사항) ─────────────────────────────────────
        analytic_signal = signal.hilbert(bvp_raw)  # 복소형 analytic signal, shape = (T,)
        amplitude_envelope = np.abs(analytic_signal) + 1e-8  # envelope, shape = (T,)
        env_norm_bvp = bvp_raw / amplitude_envelope  # 진폭 정규화

        return np.array(env_norm_bvp)  # 최종 POS PPG Pseudo-Label (shape = (T,))

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """
        Parses and preprocesses all the raw data based on split.

        1) split_raw_data → data_dirs_split
        2) multi_process_manager → preprocess_dataset_subprocess 호출
        3) build_file_list → file_list.csv 생성
        4) load_preprocessed_data → inputs/labels 리스트 갱신
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess)
        self.build_file_list(file_list_dict)
        self.load_preprocessed_data()
        print("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

    def preprocess(self, frames, bvps, config_preprocess):
        """
        Preprocesses a pair of frames (크롭+리사이즈된 얼굴)과 bvps (1D PPG).
        1) 시간 분할 (BEGIN~END)
        2) 이미 frames는 crop_face_resize에서 크롭+리사이즈 완료된 상태
        3) DATA_TYPE, LABEL_TYPE에 따라 DiffNormalized / Standardized 적용
        4) DO_CHUNK=True일 때 chunk 단위로 자름
        """
        # ─── 시간 분할: config_data.BEGIN~END 비율로 자르기 ─────────────────────────
        T = frames.shape[0]
        start = int(self.config_data.BEGIN * T)
        end = int(self.config_data.END * T)
        frames = frames[start:end]
        bvps = bvps[start:end]

        # ─── 표준화/차분(normalize) 등 데이터 변환 ──────────────────────────────────
        data_list = []
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data_list.append(f_c)
            elif data_type == "DiffNormalized":
                data_list.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data_list.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data_list, axis=-1)  # 마지막 채널 축으로 concat

        # ─── 라벨 변환: Raw / DiffNormalized / Standardized ─────────────────────────
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        # ─── 청크 단위로 자르기 ─────────────────────────────────────────────────
        if config_preprocess.DO_CHUNK:
            frames_clips, bvps_clips = self.chunk(data, bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips

    def face_detection(self, frame, backend, use_larger_box=False, larger_box_coef=1.0):
        """
        얼굴 검출 함수 (확장형)

        Args:
            frame (np.ndarray): (H, W, 3) 컬러 프레임 (BGR 또는 RGB 상관없음)
            backend (str): "HC" / "MTCNN" / "FACEMESH" 중 하나
            use_larger_box (bool): True면 검출된 박스를 larger_box_coef만큼 확대
            larger_box_coef (float): 박스 확대 비율
        Returns:
            face_box_coor (List[int]): [x, y, w, h] 형태의 얼굴 바운딩 박스
        """
        # ── 1) Haar Cascade (HC) 백엔드 ───────────────────────────────────────
        if backend == "HC":
            xml_path = os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "haarcascade_frontalface_default.xml"
            )
            xml_path = os.path.normpath(xml_path)
            detector = cv2.CascadeClassifier(xml_path)

            faces = detector.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) == 0:
                Hf, Wf = frame.shape[:2]
                face_box_coor = [0, 0, Wf, Hf]
            elif len(faces) == 1:
                x, y, w_box, h_box = faces[0]
                face_box_coor = [int(x), int(y), int(w_box), int(h_box)]
            else:
                # 여러 얼굴 중 가장 큰 영역 선택
                areas = [w * h for (_, _, w, h) in faces]
                idx_largest = int(np.argmax(areas))
                x, y, w_box, h_box = faces[idx_largest]
                face_box_coor = [int(x), int(y), int(w_box), int(h_box)]

        # ── 2) MTCNN 백엔드 (미구현) ─────────────────────────────────────
        elif backend == "MTCNN":
            raise NotImplementedError("MTCNN backend is not implemented yet.")

        # ── 3) MediaPipe Face Mesh 백엔드 ───────────────────────────────────
        elif backend == "FACEMESH":
            mp_face_mesh = mp.solutions.face_mesh
            # with 블록 안에서 FaceMesh 인스턴스를 생성 → 사용 → 종료
            with mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
            ) as face_mesh:
                # MediaPipe는 RGB를 기대하므로 BGR→RGB 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if not results.multi_face_landmarks:
                    # 랜드마크 검출 실패 시: 전체 프레임 fallback
                    Hf, Wf = frame.shape[:2]
                    face_box_coor = [0, 0, Wf, Hf]
                else:
                    # 첫 번째 얼굴(468개 랜드마크)만 사용
                    landmarks = results.multi_face_landmarks[0].landmark
                    h, w, _ = frame.shape

                    # 모든 랜드마크 x, y 좌표를 픽셀 단위로 변환
                    all_x = np.array([lm.x for lm in landmarks]) * w
                    all_y = np.array([lm.y for lm in landmarks]) * h

                    x_min = int(np.min(all_x))
                    x_max = int(np.max(all_x))
                    y_min = int(np.min(all_y))
                    y_max = int(np.max(all_y))

                    # 얼굴 윤곽에 10% 정도 padding 추가
                    pad_w = int(0.1 * (x_max - x_min))
                    pad_h = int(0.1 * (y_max - y_min))

                    x1 = max(x_min - pad_w, 0)
                    y1 = max(y_min - pad_h, 0)
                    x2 = min(x_max + pad_w, w)
                    y2 = min(y_max + pad_h, h)

                    face_box_coor = [x1, y1, x2 - x1, y2 - y1]
        else:
            raise ValueError(f"Unsupported face detection backend: {backend}")

        # ── 검출된 영역을 확대할지 여부 ─────────────────────────────────────────
        if use_larger_box:
            x, y, w_box, h_box = face_box_coor
            cx = x + w_box // 2
            cy = y + h_box // 2
            new_w = int(w_box * larger_box_coef)
            new_h = int(h_box * larger_box_coef)
            new_x = max(cx - new_w // 2, 0)
            new_y = max(cy - new_h // 2, 0)
            new_w = min(new_w, frame.shape[1] - new_x)
            new_h = min(new_h, frame.shape[0] - new_y)
            face_box_coor = [new_x, new_y, new_w, new_h]

        return face_box_coor

    def crop_face_resize(self,
                         frames,
                         use_face_detection,
                         use_larger_box,
                         larger_box_coef,
                         use_dynamic_detection,
                         detection_freq,
                         use_median_box,
                         width,
                         height):
        """
        Crop face and resize frames.

        Args:
            frames(np.ndarray): 원본 프레임 배열, shape=(T, H, W, 3)
            use_face_detection(bool): True면 얼굴만 자름
            use_larger_box(bool): True면 박스를 larger_box_coef 배 확대
            larger_box_coef(float): 박스 확대율
            use_dynamic_detection(bool): True면 매 detection_freq 프레임마다 얼굴 검출
            detection_freq(int): 얼굴 검출 주기(프레임 단위)
            use_median_box(bool): True면 검출된 여러 박스의 중앙값 사용
            width(int), height(int): 리사이즈 타겟 크기
        Returns:
            resized_frames(np.ndarray): (T, height, width, 3) 크롭+리사이즈된 프레임
        """
        global face_region_median
        if use_dynamic_detection:
            num_dynamic_det = ceil(frames.shape[0] / detection_freq)
        else:
            num_dynamic_det = 1

        face_region_all = []
        for idx in range(num_dynamic_det):
            if use_face_detection:
                backend_name = self.config_data.PREPROCESS.CROP_FACE.BACKEND
                box = self.face_detection(
                    frames[detection_freq * idx],
                    backend_name,
                    use_larger_box,
                    larger_box_coef
                )
                face_region_all.append(box)
            else:
                Hf, Wf = frames.shape[1], frames.shape[2]
                face_region_all.append([0, 0, Wf, Hf])

        face_region_all = np.asarray(face_region_all, dtype='int')
        if use_median_box:
            face_region_median = np.median(face_region_all, axis=0).astype('int')

        resized_frames = np.zeros((frames.shape[0], height, width, 3), dtype=np.uint8)
        for i in range(frames.shape[0]):
            frame = frames[i]
            if use_dynamic_detection:
                reference_index = i // detection_freq
            else:
                reference_index = 0

            if use_face_detection:
                if use_median_box:
                    face_region = face_region_median
                else:
                    face_region = face_region_all[reference_index]
                x, y, w_box, h_box = face_region
                cropped = frame[
                          max(y, 0): min(y + h_box, frame.shape[0]),
                          max(x, 0): min(x + w_box, frame.shape[1])
                          ]
            else:
                cropped = frame

            resized_frames[i] = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)

        return resized_frames

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

        return np.array(clips_f), np.array(clips_b)

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

    def save_multi_process(self, frames_clips, bvps_clips, filename):
        """
        Save all the chunked data (멀티프로세스용).

        Args:
            frames_clips(np.ndarray): 얼굴 프레임 청크들
            bvps_clips(np.ndarray): BVP (PPG) 청크들
            filename(str): 원본 영상 인덱스
        Returns:
            input_path_name_list(list): 모든 저장 경로 리스트 (입력)
            label_path_name_list(list): 모든 저장 경로 리스트 (라벨)
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)

        input_path_name_list = []
        label_path_name_list = []
        count = 0

        for clip_idx in range(len(bvps_clips)):
            # 1) 입력 프레임 저장
            input_path = os.path.join(self.cached_path, f"{filename}_input{count}.npy")
            np.save(input_path, frames_clips[clip_idx])
            input_path_name_list.append(input_path)

            # 2) 라벨 (1D PPG) 저장
            label_path = os.path.join(self.cached_path, f"{filename}_label{count}.npy")
            np.save(label_path, bvps_clips[clip_idx])
            label_path_name_list.append(label_path)

            # 3) 사전 계산된 GT HR 저장
            _, hr_gt = calculate_hr(
                bvps_clips[clip_idx], bvps_clips[clip_idx],
                fs=self.config_data.FS,
                diff_flag=False
            )
            hr_path = os.path.join(self.cached_path, f"{filename}_hr{count}.npy")
            np.save(hr_path, np.array(hr_gt, dtype=np.float32))

            count += 1

        return input_path_name_list, label_path_name_list

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=8):
        """
        Allocate dataset preprocessing across multiple processes.

        Args:
            data_dirs(list): raw data 디렉토리 리스트
            config_preprocess(CfgNode): preprocessing 설정
            multi_process_quota(int): 동시에 사용할 프로세스 최대 개수
        Returns:
            file_list_dict(dict): {process_idx: [input_path1, input_path2, ...], ...}
        """
        print('Preprocessing dataset...')
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

    @staticmethod
    def diff_normalize_data(data):
        """
        Calculate discrete difference in video data along the time-axis
        and normalize by its standard deviation.
        """
        n, h, w, c = data.shape
        diff_len = n - 1
        diffnormalized_data = np.zeros((diff_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diff_len):
            diffnormalized_data[j] = (data[j + 1] - data[j]) / (data[j + 1] + data[j] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    @staticmethod
    def diff_normalize_label(label):
        """
        Calculate discrete difference in label (1D PPG) along time-axis
        and normalize by its standard deviation.
        """
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @staticmethod
    def standardized_data(data):
        """
        Z-score standardization for video data.
        """
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        """
        Z-score standardization for 1D PPG label.
        """
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label

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
    def simple_roi_bandpass(frames, fs, lowcut=0.75, highcut=3.0):
        """
        Crop+Resize된 얼굴 시퀀스(frames, shape=(T, H, W, 3))에서
        Green 채널을 평균 내어 1차원 시계열을 얻고,
        0.75–3 Hz Band-Pass 필터를 적용하여 노이즈를 억제한 1D PPG 신호를 반환.

        Args:
            frames(np.ndarray): (T, H, W, 3) 크롭+리사이즈된 얼굴 시퀀스 (RGB)
            fs(float): 샘플링 레이트 (예: 30.0)
            lowcut(float): Band-Pass 하한 (Hz), 기본 0.75
            highcut(float): Band-Pass 상한 (Hz), 기본 3.0
        Returns:
            filtered(np.ndarray): shape = (T,), Band-Pass 후 1D PPG 시그널
        """
        # 1) 각 프레임의 Green 채널을 모두 평균 내기 → 1D 시계열 (T,)
        green_ts = frames[..., 1].reshape(frames.shape[0], -1).mean(axis=1)
        # Zero-mean 제거
        green_ts = green_ts - np.mean(green_ts)

        # 2) Butterworth Band-Pass 필터 설계
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(N=1, Wn=[low, high], btype='bandpass')

        # 3) 필터 적용 (padlen을 충분히 주어 edge effect 최소화)
        padlen = 3 * (max(len(a), len(b)) - 1)
        filtered = signal.filtfilt(b, a, green_ts, padlen=padlen)

        return filtered  # shape = (T,)
