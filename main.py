# main.py
#!/usr/bin/env python3

# ─── Mediapipe CalculatorGraph pickle 방지 ───────────────────────────────────
import mediapipe.python._framework_bindings.calculator_graph as _cg
def _cg_getstate(self): return {}
def _cg_setstate(self, state): pass
_cg.CalculatorGraph.__getstate__ = _cg_getstate
_cg.CalculatorGraph.__setstate__ = _cg_setstate
# ─────────────────────────────────────────────────────────────────────────────

import multiprocessing as mp

import argparse
import random
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # only FATAL
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'kms_swrast'  # libEGL swrast 오류 억제
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='absl')          # absl warning off
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')     # mediapipe facemesh off


import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from dataset.data_loader.Base_Loader import BaseLoader
from dataset.data_loader.PURE_Loader import PURELoader
from dataset.data_loader.UBFC_rPPG_Loader import UBFCrPPGLoader
from dataset.data_loader.MMPD_Loader import MMPDLoader

from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.trainer.PhysMambaTrainer import PhysMambaTrainer

from unsupervised_methods.unsupervised_predictor import unsupervised_predict

# ─── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # 입력 크기가 고정되어 있으면 CuDNN이 최적 커널을 선택

general_generator = torch.Generator().manual_seed(RANDOM_SEED)
train_generator   = torch.Generator().manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    parser.add_argument(
        '--config_file', required=False,
        default="configs/train_configs/intra/1PURE_PhysMamba.yaml",
        type=str, help="Path to the yaml config file."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Turn on verbose logging in data loaders and trainer"
    )
    return parser


def train_and_test(config, data_loader_dict):
    """
    모델 이름이 'PhysMamba'여야만 사용 가능.
    Trainer를 생성하고, train() → test() 순으로 실행합니다.
    """
    if config.MODEL.NAME != 'PhysMamba':
        raise ValueError(f"Unsupported model: {config.MODEL.NAME}")
    trainer = PhysMambaTrainer(config, data_loader_dict)
    trainer.train(data_loader_dict)
    trainer.test(data_loader_dict)



def only_test(config, data_loader_dict):
    """
    모델 이름이 'PhysMamba'여야만 사용 가능.
    Trainer를 생성하고, test()만 실행합니다.
    """
    if config.MODEL.NAME != 'PhysMamba':
        raise ValueError(f"Unsupported model: {config.MODEL.NAME}")
    trainer = PhysMambaTrainer(config, data_loader_dict)
    trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):
    """
    config.UNSUPERVISED.METHOD에 지정된 모든 방법(method)을 순회하며 unsupervised_predict()를 실행합니다.
    """
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set UNSUPERVISED.METHOD in the yaml.")
    for method in config.UNSUPERVISED.METHOD:
        unsupervised_predict(config, data_loader, method)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')

    # ─── CLI 파서 & Config 로드 ────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    config = get_config(args)

    if args.cached_path is not None:
        for split in ["TRAIN", "VALID", "TEST", "UNSUPERVISED"]:
            data_cfg = getattr(config, split).DATA
            data_cfg.CACHED_PATH    = args.cached_path
            data_cfg.FILE_LIST_PATH = os.path.join(args.cached_path, "DataFileLists")

    data_type = config.TRAIN.DATA.PREPROCESS.DATA_TYPE[0] if config.TRAIN.DATA.PREPROCESS.DATA_TYPE else None
    print("Data type:", data_type)

    config.defrost()
    config.VERBOSE            = args.verbose
    config.TRAIN.DATA.VERBOSE = args.verbose
    config.VALID.DATA.VERBOSE = args.verbose
    config.TEST.DATA.VERBOSE  = args.verbose
    config.freeze()

    print("\nConfiguration:\n", config, "\n")

    # ─── DataLoader 준비 ───────────────────────────────────────────────────
    data_loader_dict = {}

    if config.TOOLBOX_MODE in ("train_and_test", "only_test"):
        ds = (config.TRAIN.DATA.DATASET
              if config.TOOLBOX_MODE == "train_and_test"
              else config.TEST.DATA.DATASET)

        if ds == "PURE":
            LoaderClass = PURELoader
        elif ds == "UBFC-rPPG":
            LoaderClass = UBFCrPPGLoader
        elif ds == "MMPD":
            LoaderClass = MMPDLoader
        else:
            raise ValueError(f"Unsupported dataset: {ds}")


        if config.TOOLBOX_MODE == "train_and_test":

            # 1) Train DataLoader
            train_ds = LoaderClass(
                name="train",
                raw_data_path=config.TRAIN.DATA.RAW_DATA_PATH,
                config_data=config.TRAIN.DATA
            )
            data_loader_dict['train'] = DataLoader(
                dataset=train_ds,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                num_workers=2,            # 시스템 환경에 맞춰 2~8 정도로 조정
                multiprocessing_context=ctx,
                pin_memory=False,          # CPU→GPU 복사 최적화
                persistent_workers=False,   # epoch 간 워커 유지
                worker_init_fn=seed_worker,
                generator=train_generator
            )

            # 2) Valid DataLoader
            valid_ds = LoaderClass(
                name="valid",
                raw_data_path=config.VALID.DATA.RAW_DATA_PATH,
                config_data=config.VALID.DATA
            )
            data_loader_dict['valid'] = DataLoader(
                dataset=valid_ds,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                multiprocessing_context=ctx,
                pin_memory=False,
                persistent_workers=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )

        # 3) Test DataLoader (공통)
        test_ds = LoaderClass(
            name="test",
            raw_data_path=config.TEST.DATA.RAW_DATA_PATH,
            config_data=config.TEST.DATA
        )
        data_loader_dict['test'] = DataLoader(
            dataset=test_ds,
            batch_size=config.INFERENCE.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    elif config.TOOLBOX_MODE == "unsupervised_method":
        ds = config.UNSUPERVISED.DATA.DATASET
        if ds == "PURE":
            LoaderClass = PURELoader
        elif ds == "UBFC-rPPG":
            LoaderClass = UBFCrPPGLoader
        elif ds == "MMPD":
            LoaderClass = MMPDLoader
        else:
            raise ValueError(f"Unsupported dataset: {ds}")

        unsup_ds = LoaderClass(
            name="unsupervised",
            raw_data_path=config.UNSUPERVISED.DATA.RAW_DATA_PATH,
            config_data=config.UNSUPERVISED.DATA
        )
        data_loader_dict['unsupervised'] = DataLoader(
            dataset=unsup_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            multiprocessing_context=ctx,
            pin_memory=False,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    else:
        raise ValueError(f"Unsupported TOOLBOX_MODE: {config.TOOLBOX_MODE}")

    # ─── 실행 ─────────────────────────────────────────────────────────────────
    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)

    elif config.TOOLBOX_MODE == "only_test":
        only_test(config, data_loader_dict)

    else:
        # unsupervised_method
        unsupervised_method_inference(config, data_loader_dict['unsupervised'])
