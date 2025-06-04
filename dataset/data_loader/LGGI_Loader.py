# dataset/data_loader/LGGI_Loader.py

import glob
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from dataset.data_loader.Base_Loader import BaseLoader


class LGGILoader(BaseLoader):
    """LGGI dataset 데이터 로더 클래스.

    데이터 구조 예시:
      rppg_dataset/LGGI/
        ├── alex/
        │    ├── alex_gym/
        │    │    ├── cv_camera_sensor_stream_handler.avi
        │    │    ├── cms50_stream_handler.xml
        │    │    └── cv_camera_sensor_timer_stream_handler.xml
        │    ├── alex_resting/
        │    ├── alex_rotation/
        │    └── alex_talk/
        ├── angelo/
        │    └── ...
        └── cpi/

    - 각 세션 디렉토리 이름: <subject>_<session>
    - `cv_camera_sensor_stream_handler.avi`: RGB 영상 (25fps)
    - `cms50_stream_handler.xml`: CMS50E PPG 시계열 라벨
    - `cv_camera_sensor_timer_stream_handler.xml`: 센서 타이머 정보 (선택적)
    """

    def __init__(self, name: str, data_path: str, config_data: dict):
        self.dataset_name = "LGGI"
        self.base_path = Path(data_path).resolve()
        if not self.base_path.exists():
            raise FileNotFoundError(f"{self.dataset_name} 데이터 경로를 찾을 수 없습니다: {self.base_path}")
        super().__init__(name, str(self.base_path), config_data)

    def get_raw_data(self) -> list:
        """Subject별 세션 디렉토리 목록을 반환."""
        entries = []
        # LGGI 폴더 하위의 모든 subject 디렉토리 순회
        for subj_dir in sorted(self.base_path.iterdir()):
            if not subj_dir.is_dir():
                continue
            subj_name = subj_dir.name  # ex: 'alex'
            # 각 subject 내 세션 폴더
            for sess_dir in sorted(subj_dir.iterdir()):
                if not sess_dir.is_dir():
                    continue
                idx = sess_dir.name      # ex: 'alex_gym'
                entries.append({
                    'index': idx,
                    'subject': subj_name,
                    'session': idx.split('_',1)[1] if '_' in idx else idx,
                    'path': str(sess_dir)
                })
        if not entries:
            raise ValueError(f"{self.dataset_name} 데이터가 없습니다: {self.base_path}")
        return entries

    def split_raw_data(self, data_dirs: list, begin: float, end: float) -> list:
        """데이터 디렉토리 리스트를 분할 비율(begin~end)로 반환."""
        if begin == 0 and end == 1:
            return data_dirs
        total = len(data_dirs)
        start = int(begin * total)
        stop = int(end * total)
        return data_dirs[start:stop]

    def preprocess_dataset_subprocess(self,
                                      data_dirs: list,
                                      config_preprocess: dict,
                                      idx: int,
                                      file_list_dict: dict) -> None:
        """세션별 영상/라벨 로딩, 전처리, 저장 수행."""
        entry     = data_dirs[idx]
        sess_name = entry['index']
        folder    = entry['path']
        try:
            # 1) 영상 로딩
            avi_path = os.path.join(folder, "cv_camera_sensor_stream_handler.avi")
            if 'None' in config_preprocess.DATA_AUG:
                frames = self._read_video(avi_path)
            elif 'Motion' in config_preprocess.DATA_AUG:
                frames = self.read_npy_video(glob.glob(os.path.join(folder, '*.npy')))
            else:
                print(f"[SKIP] {sess_name}: 알 수 없는 DATA_AUG={config_preprocess.DATA_AUG}")
                return

            # 2) 라벨 로딩 (XML)
            if config_preprocess.USE_PSEUDO_PPG_LABEL:
                bvps = self.generate_pos_pseudo_labels(frames, fs=self.config_data.FS)
            else:
                xml_list = glob.glob(os.path.join(folder, "cms50_stream_handler*.xml"))
                if not xml_list:
                    print(f"[SKIP] {sess_name}: XML 파일 없음")
                    return
                bvps = self._read_xml_label(xml_list[0])

            # 3) PPG 리샘플링
            bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])

            # 4) 전처리: RoI 평균, 차분, 필터링, 클립화
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

            # 5) 저장
            input_names, _ = self.save_multi_process(frames_clips, bvps_clips, sess_name)
            file_list_dict[idx] = input_names
            print(f"[DONE] {sess_name}: {len(input_names)}개 클립 저장완료")

        except Exception as e:
            print(f"[ERROR] {sess_name}: 예외 → {e}")

    @staticmethod
    def _read_video(path: str) -> np.ndarray:
        """AVI 파일에서 프레임 읽어 (T, H, W, 3) 반환."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.asarray(frames)

    @staticmethod
    def _read_xml_label(xml_file: str) -> np.ndarray:
        """XML Value 태그에서 PPG waveform 추출하여 numpy 배열로 반환."""
        tree = ET.parse(xml_file)
        values = []
        for elem in tree.getroot().iter('Value'):
            text = elem.text.strip()
            try:
                arr = np.fromstring(text, sep=',')
                values.extend(arr.tolist())
            except:
                try:
                    values.append(float(text))
                except:
                    continue
        return np.array(values, dtype=np.float32)
