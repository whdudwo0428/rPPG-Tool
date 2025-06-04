# File: neural_methods/trainer/PhysMambaTrainer.py

import os
import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm

from scipy.signal import butter, filtfilt

from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.PhysMambaLoss import PhysMambaLoss


class PhysMambaTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        super().__init__()
        # 1) 디바이스 설정
        self.device = torch.device(config.DEVICE)
        torch.backends.cudnn.benchmark = True

        self.max_epoch_num   = config.TRAIN.EPOCHS
        self.model_dir       = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.chunk_len       = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config          = config

        # 최소 검증 손실 및 최적 epoch
        self.min_valid_loss = None
        self.best_epoch     = 0

        # diff_flag: "DiffNormalized"일 때 1
        self.diff_flag = int(config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized")

        if config.TOOLBOX_MODE == "train_and_test":
            # 2) PhysMamba 모델 & DataParallel 래핑
            base = PhysMamba().to(self.device)
            self.model = torch.nn.DataParallel(
                base, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
            )

            # 3) 손실 함수: alpha_time=1.0, alpha_freq=1.0, alpha_corr=0.5 로 조정
            self.criterion = PhysMambaLoss(alpha_time=1.0, alpha_freq=1.0, alpha_corr=0.5)
            print(f">>> [Debug] PhysMambaLoss weights: "
                  f"alpha_time={self.criterion.alpha_time:.2f}, "
                  f"alpha_freq={self.criterion.alpha_freq:.2f}, "
                  f"alpha_corr={self.criterion.alpha_corr:.2f}")

            # 5) Optimizer & Scheduler
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0
            )
            steps = len(data_loader["train"])

            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=steps
            )

        elif config.TOOLBOX_MODE == "only_test":
            # Test 전용: 모델만 로드
            base = PhysMamba().to(self.device)
            self.model = torch.nn.DataParallel(
                base, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
            )
        else:
            raise ValueError("Incorrect toolbox mode for PhysMambaTrainer")

    def train(self, data_loader):
        """
        전체 학습 루프 (클립별 표준화 후 loss 계산)
        """
        # ──────────────────── 학습/검증 손실 기록용 리스트 초기화 ────────────────────
        self.train_losses = []
        self.valid_losses = []

        for epoch in range(self.max_epoch_num):
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)

            epoch_train_loss = 0.0
            num_batches = 0

            for batch_idx, (data, labels, hr_gts, *_) in enumerate(tbar):
                # ─── 차원 순서 교정: (B, T, H, W, C) → (B, T, C, H, W)
                if data.dim() == 5 and data.shape[-1] in (1, 3):
                    data = data.permute(0, 1, 4, 2, 3).contiguous()

                data = data.float().to(self.device)  # (B, T, C, H, W)

                # ─── 라벨(원본 PPG) 처리: (B, T, 1) 형태로 가져옴
                labels = labels.float().to(self.device)
                if labels.dim() == 2:
                    labels = labels.unsqueeze(-1)  # (B, T, 1)

                B, T, _ = labels.shape

                # ─── **클립별(샘플별) 평균·표준편차** 계산하여 정규화
                lab_flat   = labels.view(-1)             # (B*T,)
                mu_clip    = lab_flat.mean(dim=0)         # 스칼라
                sigma_clip = lab_flat.std(dim=0) + 1e-8   # 스칼라

                labels_std = (labels - mu_clip) / sigma_clip  # (B, T, 1)

                # ─── HR GT (사용하지 않을 수 있지만, Forward 시 signature 맞추기 위해 넘겨줌)
                hr_gts = hr_gts.to(self.device)

                # ─── Data Augmentation (필요 시)
                if self.config.TRAIN.AUG:
                    data, labels_std = self.data_augmentation(data, labels_std)

                # ─── 순방향 & 손실 계산
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)          # (B, T)
                pred_ppg = pred_ppg.unsqueeze(-1)    # (B, T, 1)

                # [Debug: epoch0-batch0에서만 시각화 저장]
                if epoch == 0 and batch_idx == 0:
                    print(f"[Debug] epoch={epoch}, batch_idx={batch_idx}, "
                          f"data.shape={tuple(data.shape)}, labels.shape={tuple(labels.shape)}, hr_gts.shape={tuple(hr_gts.shape)}")

                    pp = pred_ppg.detach()  # (B, T, 1)
                    ls = labels_std.detach()  # (B, T, 1)

                    print(f"[Debug] pred_ppg stats: min={pp.min().item():.4f}, max={pp.max().item():.4f}, "
                          f"mean={pp.mean().item():.4f}, std={pp.std().item():.4f}")
                    print(f"[Debug] labels_std stats: min={ls.min().item():.4f}, "
                          f"max={ls.max().item():.4f}, mean={ls.mean().item():.4f}, std={ls.std().item():.4f}")

                    # Denormalize 후 비교용 시각화 (배치 첫 샘플만)
                    pp_std = pp.squeeze().cpu().numpy()  # (B, T)
                    gt_std = ls.squeeze().cpu().numpy()  # (B, T)
                    pp_den = pp_std * sigma_clip.cpu().numpy() + mu_clip.cpu().numpy()  # (B, T)
                    gt_den = gt_std * sigma_clip.cpu().numpy() + mu_clip.cpu().numpy()  # (B, T)
                    t_axis = np.arange(pp_den.shape[1]) / self.config.TRAIN.DATA.FS
                    import matplotlib.pyplot as plt
                    os.makedirs("debug_plots", exist_ok=True)
                    plt.figure(figsize=(8, 2))
                    plt.plot(t_axis, gt_den[0], label="GT PPG [0]", lw=1.0)
                    plt.plot(t_axis, pp_den[0], label="Pred PPG [0]", lw=1.0, alpha=0.7)
                    plt.xlabel("Time [s]")
                    plt.ylabel("PPG amplitude")
                    plt.title("Epoch0 Batch0: GT vs Pred")
                    plt.legend(loc="upper right")
                    plt.tight_layout()
                    plt.savefig("debug_plots/epoch0_batch0_ppg.png", dpi=200, bbox_inches="tight")
                    plt.close()
                    print("[Debug] Saved → debug_plots/epoch0_batch0_ppg.png")

                # ─── **클립별 정규화된 레이블**을 사용하여 손실 계산
                loss_per_sample = self.criterion(
                    pred_ppg,                # (B, T, 1) 정규화 예측
                    labels_std,              # (B, T, 1) 정규화 GT
                    hr_gts,                  # (B,) HR (사실 사용되지 않음)
                    epoch,                   # 현재 Epoch
                    self.config.TRAIN.DATA.FS,
                    self.diff_flag,
                    mu_clip,                 # 이 클립(샘플)의 μ
                    sigma_clip               # 이 클립(샘플)의 σ
                )
                loss = loss_per_sample.mean()

                # ─── 역전파 & 최적화
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # ─── tqdm에 손실 표시
                tbar.set_postfix(loss=f"{loss.item():.4f}")

                # ─── 배치별 손실 누적
                epoch_train_loss += loss.item()
                num_batches += 1

            # ─── 에폭 종료 시 모델 저장
            self.save_model(epoch)

            # ─── 이 epoch의 평균 Train Loss 기록
            avg_tr_loss = (epoch_train_loss / num_batches) if num_batches > 0 else float('nan')
            self.train_losses.append(avg_tr_loss)
            print(f"[Debug] Epoch {epoch} 평균 Train Loss = {avg_tr_loss:.4f}")

            # ─── Validation (USE_LAST_EPOCH=False 시)
            if not self.config.TEST.USE_LAST_EPOCH:
                vl = self.valid(data_loader)
                print(f"[Info] Validation loss (epoch {epoch}): {vl:.4f}")
                self.valid_losses.append(vl)
                if self.min_valid_loss is None or vl < self.min_valid_loss:
                    self.min_valid_loss, self.best_epoch = vl, epoch
                    print(f"[Info] Updated best model: epoch {epoch}")

        # ───────────────────────────────────────────────────────────────────────────
        # **전체 학습 종료 후: Train/Valid Loss 곡선 저장**
        import matplotlib.pyplot as plt
        os.makedirs("debug_plots", exist_ok=True)
        epochs_arr = np.arange(len(self.train_losses))
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_arr, self.train_losses, label="Train Loss")
        plt.plot(epochs_arr, self.valid_losses, label="Valid Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train & Valid Loss Curve")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("debug_plots/loss_curve.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[Debug] Saved → debug_plots/loss_curve.png")

    def valid(self, data_loader):
        """
        벡터화된 Validation 루프 (클립별 표준화 적용)
        """
        print("\n[Info] === Validating ===")
        self.model.eval()
        total_val_loss = 0.0
        count_batches  = 0

        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for batch_idx_v, (data_v, labels_v, hr_gts_v, *_ ) in enumerate(vbar):
                # ─── 차원 순서 교정 (B, T, H, W, C) → (B, T, C, H, W)
                if data_v.dim() == 5 and data_v.shape[-1] in (1, 3):
                    data_v = data_v.permute(0, 1, 4, 2, 3).contiguous()
                data_v = data_v.float().to(self.device)

                # ─── GT 레이블 (B, T, 1) 형태로
                labels_v = labels_v.float().to(self.device)
                if labels_v.dim() == 2:
                    labels_v = labels_v.unsqueeze(-1)

                B, T, _ = labels_v.shape

                # ─── **클립별 평균·표준편차** 계산
                lab_flat_v   = labels_v.view(-1)              # (B*T,)
                mu_clip_v    = lab_flat_v.mean(dim=0)          # 스칼라
                sigma_clip_v = lab_flat_v.std(dim=0) + 1e-8    # 스칼라

                labels_v_std = (labels_v - mu_clip_v) / sigma_clip_v  # (B, T, 1)

                hr_gts_v = hr_gts_v.to(self.device)

                # ─── 모델 출력
                pred_v = self.model(data_v)  # (B, T)
                pred_v = pred_v.unsqueeze(-1)  # (B, T, 1)

                # ─── 첫 배치에 한해 시각화 저장 (denorm)
                if batch_idx_v == 0:
                    pp_std_v = pred_v.detach().cpu().numpy().reshape(-1)  # (T,)
                    gt_std_v = labels_v_std.detach().cpu().numpy().reshape(-1)  # (T,)
                    pp_den_v = pp_std_v * sigma_clip_v.cpu().numpy() + mu_clip_v.cpu().numpy()
                    gt_den_v = gt_std_v * sigma_clip_v.cpu().numpy() + mu_clip_v.cpu().numpy()
                    t_v = np.arange(pp_den_v.shape[0]) / self.config.VALID.DATA.FS
                    import matplotlib.pyplot as plt
                    os.makedirs("debug_plots", exist_ok=True)
                    plt.figure(figsize=(8,2))
                    plt.plot(t_v, gt_den_v, label="Valid GT", lw=1.0)
                    plt.plot(t_v, pp_den_v, label="Valid Pred", lw=1.0, alpha=0.7)
                    plt.xlabel("Time [s]")
                    plt.ylabel("PPG amplitude")
                    plt.title("Valid Batch0: GT vs Pred")
                    plt.legend(loc="upper right")
                    plt.tight_layout()
                    plt.savefig("debug_plots/valid_batch0_ppg.png", dpi=200, bbox_inches="tight")
                    plt.close()
                    print("[Debug] Saved → debug_plots/valid_batch0_ppg.png")

                # ─── **클립별 정규화된 레이블**을 사용하여 손실 계산
                loss_vec = self.criterion(
                    pred_v,
                    labels_v_std,
                    hr_gts_v,
                    self.max_epoch_num,
                    self.config.VALID.DATA.FS,
                    self.diff_flag,
                    mu_clip_v,
                    sigma_clip_v
                )

                batch_loss = loss_vec.mean()
                total_val_loss += batch_loss.detach()
                count_batches += 1

                vbar.set_postfix(valid_loss=f"{batch_loss.item():.4f}")

        avg_val_loss = (total_val_loss / count_batches).item() if count_batches > 0 else float('nan')
        return avg_val_loss

    def test(self, data_loader):
        """
        테스트 루프 (모델 로드 → 예측 → Band-Pass → Metric 계산)
        """
        print("\n[Info] === Testing ===")

        # 1) 모델 weight 불러오기
        if self.config.TOOLBOX_MODE == "only_test":
            path = self.config.INFERENCE.MODEL_PATH
            if not os.path.isfile(path):
                raise ValueError("Invalid MODEL_PATH in yaml.")
            state = torch.load(path)
            self.model.load_state_dict(state)
        else:
            epoch = (self.max_epoch_num - 1) if self.config.TEST.USE_LAST_EPOCH else self.best_epoch
            fname = f"{self.model_file_name}_Epoch{epoch}.pth"
            path = os.path.join(self.model_dir, fname)
            self.model.load_state_dict(torch.load(path))

        self.model.to(self.device).eval()
        preds, labs = {}, {}

        # 2) Band-Pass 필터 설계 (0.75–3 Hz)
        fs = self.config.TEST.DATA.FS   # 예: 30
        nyq = 0.5 * fs
        low = 0.75 / nyq
        high = 3.0 / nyq
        b, a = butter(N=1, Wn=[low, high], btype='bandpass')

        with torch.no_grad():
            for data_t, lab_t, _, sid_list, idx_list in data_loader["test"]:
                # ─── (B, T, H, W, C) → (B, T, C, H, W)
                if data_t.dim() == 5 and data_t.shape[-1] in (1, 3):
                    data_t = data_t.permute(0, 1, 4, 2, 3).contiguous()
                data_t = data_t.float().to(self.device)

                # ─── GT 라벨 (B, T, 1) 형태로
                lab_t = lab_t.float().to(self.device)
                if lab_t.dim() == 2:
                    lab_t = lab_t.unsqueeze(-1)

                B, T, _ = lab_t.shape

                # ─── **클립별 평균·표준편차** 계산 (denorm 용)
                lab_flat_test   = lab_t.view(-1)               # (B*T,)
                mu_clip_test    = lab_flat_test.mean(dim=0)     # 스칼라
                sigma_clip_test = lab_flat_test.std(dim=0) + 1e-8  # 스칼라

                # ─── 모델 출력 (정규화된 PPG, shape=(B, T))
                out_std = self.model(data_t)    # (B, T)
                out_std = out_std.unsqueeze(-1)  # (B, T, 1)

                # ─── Denormalize: (B, T, 1)
                out_denorm = out_std * sigma_clip_test + mu_clip_test  # (B, T, 1)

                # ─── ★ Band-Pass 필터 적용 (모델 출력) ★
                #     → out_denorm은 (B, T, 1) 텐서이므로, 먼저 NumPy로 꺼낸 뒤 필터 후 다시 Tensor로 복귀
                out_np = out_denorm.squeeze(-1).cpu().numpy()  # (B, T)
                out_filt = np.zeros_like(out_np)
                for i in range(B):
                    out_filt[i, :] = filtfilt(b, a, out_np[i, :], padlen=3 * (max(len(a), len(b)) - 1))
                out_filt_torch = torch.from_numpy(out_filt).float().to(self.device).unsqueeze(-1)  # (B, T, 1)

                # ─── ★ GT 라벨도 동일하게 Band-Pass (필터링) ★
                lab_np = lab_t.squeeze(-1).cpu().numpy()  # (B, T)
                lab_filt = np.zeros_like(lab_np)
                for i in range(B):
                    lab_filt[i, :] = filtfilt(b, a, lab_np[i, :], padlen=3 * (max(len(a), len(b)) - 1))
                lab_filt_torch = torch.from_numpy(lab_filt).float().to(self.device).unsqueeze(-1)  # (B, T, 1)

                # ─── “클립 단위 저장” (chunk_len 단위로 나누어서 dict에 담아둠)
                out_flat = out_filt_torch.view(-1, 1)   # (B*T, 1)
                lab_flat = lab_filt_torch.view(-1, 1)   # (B*T, 1)

                for i in range(B):
                    sid = sid_list[i]
                    idx = int(idx_list[i])
                    start = i * self.chunk_len
                    end   = (i + 1) * self.chunk_len

                    preds.setdefault(sid, {})[idx] = out_flat[start:end]
                    labs.setdefault(sid, {})[idx]  = lab_flat[start:end]

        # ─── 3) calculate_metrics 호출 (Band-Pass 후 데이터를 넘겨 줌) ─────
        calculate_metrics(preds, self.config)

    def save_model(self, epoch):
        """모델 state_dict를 저장."""
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{epoch}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"[Info] Saved Model: {path}")

    def data_augmentation(self, data: torch.Tensor, labels: torch.Tensor):
        """
        배치 단위 Data Augmentation (GPU에서 벡터화):
         1) Flip (좌우 반전)
         2) Roll (Circular shift) → HR에 따라 shift 양 조정
        """
        # data: (B, T, C, H, W) 예상, labels: (B, T, 1)
        if data.dim() == 5 and data.shape[1] not in (1, 3) and data.shape[2] in (1, 3):
            # (B, T, C, H, W) → (B, C, T, H, W)
            data = data.permute(0, 2, 1, 3, 4).contiguous()

        B, C, T, H, W = data.shape

        # 1) 랜덤 Flip (50% 확률)
        if random.random() < 0.5:
            data = torch.flip(data, dims=[2])  # 시간축(t=2)에 대해 뒤집기
            labels = torch.flip(labels, dims=[1])

        # 2) 랜덤 Roll (Circular shift)
        augmented_data = data.clone()
        augmented_labels = labels.clone()

        for i in range(B):
            seq = labels[i].view(T).detach().cpu().numpy()
            hr_fft, _ = calculate_hr(
                seq, seq,
                diff_flag=self.diff_flag,
                fs=self.config.TRAIN.DATA.FS
            )

            if random.random() < 0.5:
                if hr_fft > 90:
                    shift = random.randint(0, T // 2 - 1)
                    augmented_data[i] = torch.roll(data[i], shifts=shift, dims=2)
                    augmented_labels[i] = torch.roll(labels[i], shifts=shift, dims=1)

                elif hr_fft < 75:
                    temp_data = torch.zeros_like(data[i])  # (C, T, H, W)
                    temp_labels = torch.zeros_like(labels[i])  # (T, 1)
                    for t in range(T):
                        if t < T // 2:
                            idx = t * 2
                            temp_data[:, t, :, :] = data[i, :, idx, :, :]
                            temp_labels[t] = labels[i, idx]
                        else:
                            temp_data[:, t, :, :] = temp_data[:, t - (T // 2), :, :]
                            temp_labels[t] = temp_labels[t - (T // 2)]
                    augmented_data[i] = temp_data
                    augmented_labels[i] = temp_labels

                else:
                    augmented_data[i] = data[i]
                    augmented_labels[i] = labels[i]
            else:
                augmented_data[i] = data[i]
                augmented_labels[i] = labels[i]

        # 모델은 (B, T, C, H, W)를 기대하므로 다시 permute
        augmented_data = augmented_data.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W) → (B, T, C, H, W)
        return augmented_data, augmented_labels
