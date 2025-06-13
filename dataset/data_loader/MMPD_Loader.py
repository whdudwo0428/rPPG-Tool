""" The dataloader for MMPD datasets. """

import os
import glob
import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from multiprocessing import Process, Manager
from dataset.data_loader.Base_Loader import BaseLoader
from scipy.signal import butter, filtfilt
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

class MMPDLoader(BaseLoader):
    """The data loader for the MMPD dataset."""

    def __init__(self, name, data_path, config_data):
        self.info = config_data.INFO
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, raw_data_path):
        """Returns data directories under the path (For MMPD dataset)."""
        data_dirs = sorted(glob.glob(os.path.join(raw_data_path, 'subject*')))
        if not data_dirs:
            raise ValueError(f'{self.dataset_name} data paths empty!')
        dirs = []
        for data_dir in data_dirs:
            subject = int(os.path.basename(data_dir).lstrip('subject'))
            for mat_file in sorted(os.listdir(data_dir)):
                idx = mat_file.split('_')[-1].split('.')[0]
                dirs.append({
                    'index': idx,
                    'path': os.path.join(data_dir, mat_file),
                    'subject': subject
                })
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Split data_dirs by subject, ensuring no subject overlap."""
        if begin == 0 and end == 1:
            return data_dirs

        # group by subject
        data_info = {}
        for item in data_dirs:
            s = item['subject']
            data_info.setdefault(s, []).append(item)

        subj_list = sorted(data_info.keys())
        num = len(subj_list)
        start, stop = int(begin * num), int(end * num)
        chosen = subj_list[start:stop]
        split = []
        for s in chosen:
            split.extend(data_info[s])
        print('used subject ids for split:', chosen)
        return split

    def preprocess_dataset_subprocess(self, data_dirs, cfg, idx, flist):
        """Called by multi-process manager."""
        frames, bvps, *meta = self.read_mat(data_dirs[idx]['path'])
        light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup = meta

        fname = (
            f"subject{data_dirs[idx]['subject']}"
            f"_L{light}_MO{motion}_E{exercise}"
            f"_S{skin_color}_GE{gender}_GL{glasser}"
            f"_H{hair_cover}_MA{makeup}"
        )

        # uint8 frames, resample BVP
        frames = (np.round(frames * 255)).astype(np.uint8)
        bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])

        # preprocess → save clips
        clips, labels = self.preprocess(frames, bvps, cfg)
        inames, _ = self.save_multi_process(clips, labels, fname)
        flist[idx] = inames

    def read_mat(self, mat_file):
        """Load .mat, extract video, GT_ppg and metadata."""
        mat = sio.loadmat(mat_file)
        frames = np.array(mat['video'])
        if self.config_data.PREPROCESS.USE_PSEUDO_PPG_LABEL:
            bvps = self.generate_pos_pseudo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = np.array(mat['GT_ppg']).T.reshape(-1)

        info = [
            mat['light'], mat['motion'], mat['exercise'], mat['skin_color'],
            mat['gender'], mat['glasser'], mat['hair_cover'], mat['makeup']
        ]
        light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup = \
            self.get_information(info)
        return frames, bvps, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup

    def load_preprocessed_data(self):
        """Loads preprocessed file list, filters by metadata."""
        df = pd.read_csv(self.file_list_path)
        valid = []
        for path in df['input_files']:
            parts = os.path.splitext(os.path.basename(path))[0].split('_')
            _, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup = parts
            vals = {
                'light': int(light[-1]), 'motion': int(motion[-1]),
                'exercise': int(exercise[-1]), 'skin_color': int(skin_color[-1]),
                'gender': int(gender[-1]), 'glasser': int(glasser[-1]),
                'hair_cover': int(hair_cover[-1]), 'makeup': int(makeup[-1])
            }
            if all(vals[k] in getattr(self.info, k.upper()) for k in vals):
                valid.append(path)

        if not valid:
            raise ValueError(f'{self.dataset_name} dataset loading error!')
        valid = sorted(valid)
        self.inputs = valid
        self.labels = [p.replace('_input', '_label') for p in valid]
        self.preprocessed_data_len = len(self.inputs)

    @staticmethod
    def get_information(info):
        """Map raw metadata to integer codes."""
        # light
        l = info[0]
        if l == 'LED-low':     light=1
        elif l == 'LED-high':  light=2
        elif l == 'Incandescent': light=3
        elif l == 'Nature':    light=4
        else: raise ValueError(f'Unsupported light: {l}')

        # motion
        m = info[1]
        if m in ('Stationary','Stationary (after exercise)'): motion=1
        elif m=='Rotation':    motion=2
        elif m=='Talking':     motion=3
        elif m in ('Walking','Watching Videos'): motion=4
        else: raise ValueError(f'Unsupported motion: {m}')

        # exercise
        e = info[2]
        if e=='True':          exercise=1
        elif e=='False':       exercise=2
        else: raise ValueError(f'Unsupported exercise: {e}')

        # skin_color
        sc = info[3][0][0]
        if sc in (3,4,5,6):    skin_color=sc
        else: raise ValueError(f'Unsupported skin_color: {sc}')

        # gender
        g = info[4]
        if g=='male':          gender=1
        elif g=='female':      gender=2
        else: raise ValueError(f'Unsupported gender: {g}')

        # glasser
        gl = info[5]
        glasser = 1 if gl=='True' else 2 if gl=='False' else None
        if glasser is None:    raise ValueError(f'Unsupported glasser: {gl}')

        # hair_cover
        hc = info[6]
        hair_cover = 1 if hc=='True' else 2 if hc=='False' else None
        if hair_cover is None: raise ValueError(f'Unsupported hair_cover: {hc}')

        # makeup
        ma = info[7]
        makeup = 1 if ma=='True' else 2 if ma=='False' else None
        if makeup is None:     raise ValueError(f'Unsupported makeup: {ma}')

        return light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup
