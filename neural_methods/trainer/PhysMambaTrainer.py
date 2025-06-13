# File: neural_methods/trainer/PhysMambaTrainer.py

import os
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import random
from tqdm import tqdm

from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.PhysMambaLoss import PhysMambaLoss


class PhysMambaTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        self.min_valid_loss = None
        self.best_epoch = 0
        super().__init__()
        # Device, seeds, logging…
        self.device = torch.device(config.DEVICE)
        torch.backends.cudnn.benchmark = True
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        # 안전하게 리스트/문자열 내 포함 여부로 판별
        ltypes = config.TRAIN.DATA.PREPROCESS.LABEL_TYPE
        if not isinstance(ltypes, (list, tuple)):
            ltypes = [ltypes]
        self.diff_flag = ("DiffNormalized" in ltypes)

        if config.TOOLBOX_MODE == "train_and_test":
            base = PhysMamba().to(self.device)
            self.model = torch.nn.DataParallel(
                base, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
            )

            # → config에서 받은 값으로 초기화
            self.criterion = PhysMambaLoss(config)

            print(f">>> [Debug] PhysMambaLoss weights: "
                  f"alpha_time={self.criterion.alpha_time:.2f}, "
                  f"alpha_denorm={self.criterion.alpha_denorm:.2f}, "
                  f"alpha_freq={self.criterion.alpha_freq:.2f}, "
                  f"alpha_corr={self.criterion.alpha_corr:.2f}")

            # Optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.TRAIN.LR,
                betas=tuple(config.TRAIN.OPTIMIZER.BETAS),
                eps=config.TRAIN.OPTIMIZER.EPS,
                weight_decay=0
            )
            steps = len(data_loader["train"])
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=steps
            )
            self.scaler = GradScaler()

        elif config.TOOLBOX_MODE == "only_test":
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
        import matplotlib.pyplot as plt
        # ──────────────────── 학습/검증 손실 기록용 리스트 초기화 ────────────────────
        self.train_losses = []
        self.valid_losses = []

        for epoch in range(self.max_epoch_num):
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)

            epoch_train_loss = 0.0
            num_batches = 0

            for batch_idx, (data, labels, mu_batch, sigma_batch, hr_gts, *_) in enumerate(tbar):
                # ─── 차원 순서 교정: (B, T, H, W, C) → (B, T, C, H, W)
                if data.dim() == 5 and data.shape[-1] in (1, 3):
                    data = data.permute(0, 1, 4, 2, 3).contiguous()

                if batch_idx > 0 and batch_idx % 20 == 0:
                    print(f"[Info] Epoch {epoch} Batch {batch_idx}/{len(data_loader['train'])}")

                data = data.float().to(self.device)  # (B, T, C, H, W)

                # ─── 라벨(원본 PPG) 처리: (B, T, 1) 형태로 가져옴
                labels = labels.float().to(self.device)
                if labels.dim() == 2:
                    labels = labels.unsqueeze(-1)  # (B, T, 1)

                B, T, _ = labels.shape

                # ─── 로더가 표준화/차분 정규화한 레이블 + mu/sigma 그대로 사용
                labels_std = labels  # (B, T, 1)
                mu_clip = mu_batch.to(self.device)  # (B,)
                sigma_clip = sigma_batch.to(self.device)  # (B,)

                # ─── HR GT (사용하지 않을 수 있지만, Forward 시 signature 맞추기 위해 넘겨줌)
                hr_gts = hr_gts.to(self.device)

                # ─── Data Augmentation (필요 시)
                if self.config.TRAIN.AUG:
                    data, labels_std = self.data_augmentation(data, labels_std)

                # ─── 순방향 & AMP 혼합 정밀도 계산
                self.optimizer.zero_grad()

                with autocast():
                    pred_ppg = self.model(data)  # (B, T)
                    pred_ppg = pred_ppg.unsqueeze(-1)  # (B, T, 1)

                    if epoch == 0 and batch_idx == 0:
                        pp = pred_ppg.detach().squeeze(-1)[0].cpu().numpy()
                        gt = labels_std.detach().squeeze(-1)[0].cpu().numpy()
                        sigma0 = sigma_clip[0].item()
                        mu0 = mu_clip[0].item()
                        pp_den = pp * sigma0 + mu0
                        gt_den = gt * sigma0 + mu0
                        t = np.arange(pp_den.shape[0]) / self.config.TRAIN.DATA.FS
                        plt.figure(figsize=(8, 2))
                        plt.plot(t, gt_den, label="GT PPG [0]")
                        plt.plot(t, pp_den, label="Pred PPG [0]", alpha=0.7)
                        plt.xlabel("Time [s]")
                        plt.ylabel("PPG amplitude")
                        plt.title(f"Epoch{epoch} Batch{batch_idx}: GT vs Pred")
                        plt.legend(loc="upper right")
                        plt.tight_layout()
                        # ensure 디렉터리 생성 및 안전 저장
                        try:
                            os.makedirs("debug_plots", exist_ok=True)
                            plt.savefig(f"debug_plots/epoch{epoch}_batch{batch_idx}_ppg.png", dpi=200)
                            print(f"[Debug] Saved → debug_plots/epoch{epoch}_batch{batch_idx}_ppg.png")
                        except Exception as e:
                            print(f"⚠️ Failed to save debug plot: {e}")
                        finally:
                            plt.close()

                    loss_per_sample = self.criterion(pred_ppg,  # (B, T, 1) 정규화 예측
                                                     labels_std,  # (B, T, 1) 정규화 GT
                                                     hr_gts,  # (B,) HR (사실 사용되지 않음)
                                                     epoch,  # 현재 Epoch
                                                     self.config.TRAIN.DATA.FS,
                                                     self.diff_flag,
                                                     mu_clip,  # 이 클립(샘플)의 μ
                                                     sigma_clip  # 이 클립(샘플)의 σ
                                                     )
                    loss = loss_per_sample.mean()

                # ─── AMP: Scale & Backward, Optimizer step
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                # ─── tqdm에 손실 표시
                tbar.set_postfix(loss=f"{loss.item():.4f}")

                # ─── 배치별 손실 누적
                epoch_train_loss += loss.item()
                num_batches += 1

            # ─── 이 epoch의 평균 Train Loss 기록
            avg_tr_loss = (epoch_train_loss / num_batches) if num_batches > 0 else float('nan')
            self.train_losses.append(avg_tr_loss)
            print(f"[Debug] Epoch {epoch} 평균 Train Loss = {avg_tr_loss:.4f}")

            # ─── Validation & best-model tracking (매 epoch마다 실행)
            vl = self.valid(data_loader)
            print(f"[Info] Validation loss (epoch {epoch}): {vl:.4f}")
            self.valid_losses.append(vl)
            if self.min_valid_loss is None or vl < self.min_valid_loss:
                self.min_valid_loss, self.best_epoch = vl, epoch
                print(f"[Info] Updated best model: epoch {epoch}")
                # 모델 저장은 훈련 종료 후에 한 번만 수행합니다.

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
        # 마지막 에폭 모델 저장 (USE_LAST_EPOCH=True일 때)

        # 전체 학습 종료 후: 모델 저장
        # USE_LAST_EPOCH=True → 마지막 epoch, else → best epoch
        if self.config.TEST.USE_LAST_EPOCH:
            epoch_to_save = self.max_epoch_num - 1
        else:
            epoch_to_save = self.best_epoch
        self.save_model(epoch_to_save)

    def valid(self, data_loader):
        """
        벡터화된 Validation 루프 (클립별 표준화 적용)
        """
        print("\n[Info] === Validating ===")
        self.model.eval()
        total_val_loss = 0.0
        count_batches = 0

        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for batch_idx_v, (data_v, labels_v, mu_batch_v, sigma_batch_v, hr_gts_v, *_) in enumerate(vbar):
                # ─── 차원 순서 교정 (B, T, H, W, C) → (B, T, C, H, W)
                if data_v.dim() == 5 and data_v.shape[-1] in (1, 3):
                    data_v = data_v.permute(0, 1, 4, 2, 3).contiguous()
                if batch_idx_v > 0 and batch_idx_v % 20 == 0:
                    print(f"[Info] Valid Batch {batch_idx_v}/{len(data_loader['valid'])}")

                data_v = data_v.float().to(self.device)

                labels_v = labels_v.float().to(self.device)
                if labels_v.dim() == 2:
                    labels_v = labels_v.unsqueeze(-1)  # (B, T, 1)

                # ─── 로더가 준 표준화/차분 정규화 레이블 + mu/sigma 바로 사용
                labels_v_std = labels_v  # (B, T, 1)
                mu_clip_v = mu_batch_v.to(self.device)  # (B,)
                sigma_clip_v = sigma_batch_v.to(self.device)  # (B,)

                hr_gts_v = hr_gts_v.to(self.device)

                # ─── AMP로 forward
                with autocast():
                    pred_v = self.model(data_v)  # (B, T)
                    pred_v = pred_v.unsqueeze(-1)  # (B, T, 1)

                # ─── 첫 배치 디버그 플롯
                if batch_idx_v == 0:
                    pp_std_v = pred_v.detach().cpu().numpy().reshape(-1)  # (T,)
                    gt_std_v = labels_v_std.detach().cpu().numpy().reshape(-1)
                    # use scalar μ/σ from the first sample
                    mu0 = mu_clip_v[0].cpu().item()
                    sigma0 = sigma_clip_v[0].cpu().item()
                    pp_den_v = pp_std_v * sigma0 + mu0
                    gt_den_v = gt_std_v * sigma0 + mu0
                    t_v = np.arange(pp_den_v.shape[0]) / self.config.VALID.DATA.FS
                    import matplotlib.pyplot as plt, os
                    os.makedirs("debug_plots", exist_ok=True)
                    plt.figure(figsize=(8, 2))
                    plt.plot(t_v, gt_den_v, label="Valid GT")
                    plt.plot(t_v, pp_den_v, label="Valid Pred", alpha=0.7)
                    plt.xlabel("Time [s]")
                    plt.ylabel("PPG amplitude")
                    plt.title("Valid Batch0: GT vs Pred")
                    plt.legend(loc="upper right")
                    plt.tight_layout()
                    plt.savefig("debug_plots/valid_batch0_ppg.png", dpi=200, bbox_inches="tight")
                    plt.close()
                    print("[Debug] Saved → debug_plots/valid_batch0_ppg.png")

                # ─── loss 계산
                with autocast():
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

        avg_val = (total_val_loss / count_batches).item() if count_batches > 0 else float('nan')
        return avg_val

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

        print(f"[Info] Loaded model: {path}")
        print(f"[Info] # Test batches = {len(data_loader['test'])}")

        self.model.to(self.device).eval()
        preds, labs = {}, {}

        with ((torch.no_grad())):
            for batch_idx, (data_t,lab_t,mu_batch_t,sigma_batch_t,hr_gts_t,sid_list,idx_list
                            )in enumerate(data_loader["test"]):
                if batch_idx % 20 == 0 or batch_idx == len(data_loader["test"]) - 1:
                    print(f"[Info] Processing batch {batch_idx + 1}/{len(data_loader['test'])}")

                if data_t.dim() == 5 and data_t.shape[-1] in (1, 3):
                    data_t = data_t.permute(0, 1, 4, 2, 3).contiguous()
                data_t = data_t.float().to(self.device)

                lab_t = lab_t.float().to(self.device)
                if lab_t.dim() == 2:
                    lab_t = lab_t.unsqueeze(-1)
                B, T, _ = lab_t.shape

                # ─── 로더가 준 mu/sigma 로 바로 denorm
                mu_clip_test = mu_batch_t.to(self.device)  # (B,)
                sigma_clip_test = sigma_batch_t.to(self.device)  # (B,)

                with autocast():
                    # 1) 네트워크 출력: standardized or diff-normalized prediction
                    out_std = self.model(data_t)  # -> (B, T)
                    out_std = out_std.unsqueeze(-1)  # -> (B, T, 1)

                    # 2) denorm: 표준화(Standardized) vs 차분정규화(DiffNormalized) 구분
                if self.diff_flag:
                    # diff_normalize_labels 로 학습했으므로, 우선 미분 신호 복원
                    # pred_diff[t] = σ_diff * pred_std[t] + μ_diff
                    diff_pred = out_std * sigma_clip_test.view(B, 1, 1) \
                                + mu_clip_test.view(B, 1, 1)  # (B, T, 1)
                    # 3) 적분: 누적합으로 원 PPG 파형 복원 (offset은 0으로 설정)
                    out_denorm = torch.cumsum(diff_pred, dim=1)  # (B, T, 1)
                else:
                    # standardized labels 일 때는 기존 방식으로 복원하면 원파형
                    out_denorm = (out_std * sigma_clip_test.view(B, 1, 1)
                                  + mu_clip_test.view(B, 1, 1))  # (B, T, 1)

                # ─── GT 라벨도 동일한 방식으로 denorm & (diff_mode이면) 적분
                den = lab_t * sigma_clip_test.view(B, 1, 1) \
                      + mu_clip_test.view(B, 1, 1)  # (B, T, 1)
                if self.diff_flag:
                    lab_den = torch.cumsum(den, dim=1)
                else:
                    lab_den = den
                out_flat = out_denorm.reshape(-1, 1)
                lab_flat = lab_den.reshape(-1, 1)

                for i in range(B):
                    sid = sid_list[i]
                    idx = int(idx_list[i])
                    start = i * self.chunk_len
                    end = (i + 1) * self.chunk_len
                    preds.setdefault(sid, {})[idx] = out_flat[start:end]
                    labs.setdefault(sid, {})[idx] = lab_flat[start:end]

        calculate_metrics(preds, labs, self.config)

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
