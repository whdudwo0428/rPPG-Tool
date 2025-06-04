# File: neural_methods/loss/PhysMambaLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neural_methods.loss.TorchLossComputer import TorchLossComputer


class NegPearson(nn.Module):
    """
    Negative Pearson correlation loss.
    preds, labels: (T, 1) 또는 (B, T, 1) 형태의 standardized PPG 시퀀스
    """
    def __init__(self):
        super(NegPearson, self).__init__()

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # ── 배치 처리 모드: preds.dim() == 3 → (B, T, 1)
        if preds.dim() == 3:
            B = preds.size(0)
            losses = []
            for i in range(B):
                x = preds[i].view(-1)
                y = labels[i].view(-1)
                xm = x - x.mean()
                ym = y - y.mean()
                num = (xm * ym).sum()
                den = torch.sqrt((xm ** 2).sum() + 1e-8) * torch.sqrt((ym ** 2).sum() + 1e-8)
                corr = num / den
                corr = torch.clamp(corr, -1.0, 1.0)
                losses.append(1.0 - corr)
            return torch.stack(losses)  # (B,)

        # ── 단일 시퀀스 모드: preds.dim() == 2 → (T, 1)
        elif preds.dim() == 2:
            x = preds.view(-1)
            y = labels.view(-1)
            xm = x - x.mean()
            ym = y - y.mean()
            num = (xm * ym).sum()
            den = torch.sqrt((xm ** 2).sum() + 1e-8) * torch.sqrt((ym ** 2).sum() + 1e-8)
            corr = num / den
            corr = torch.clamp(corr, -1.0, 1.0)
            return 1.0 - corr  # 스칼라

        else:
            raise ValueError(f"NegPearson: preds.dim()={preds.dim()} 은 지원하지 않습니다.")


class PhysMambaLoss(nn.Module):
    """
    PhysMamba 논문에서 제안된 복합 손실 함수:
      L_total = α_time * MSE + α_freq * (CE + KL) + α_corr * NegPearson
    - 단, 기존 코드는 alpha_freq 내부에서 *5.0, alpha_corr 내부에서 *2.0 을 했으나,
      이 부분을 제거하고, 사용자가 넘겨준 alpha 그대로 적용하도록 수정했습니다.
    """
    def __init__(
        self,
        alpha_time: float = 1.0,
        alpha_freq: float = 1.0,
        alpha_corr: float = 0.5,
        sigma: float = 2.0
    ):
        super(PhysMambaLoss, self).__init__()
        # 생성자에 넘겨진 가중치를 그대로 사용
        self.alpha_time = alpha_time
        self.alpha_freq = alpha_freq
        self.alpha_corr = alpha_corr
        self.sigma      = sigma
        self.pearson    = NegPearson()

    def forward(
        self,
        pred_std: Tensor,    # (B, T, 1) or (T, 1) 또는 (T,)
        gt_std:   Tensor,    # (B, T, 1) or (T, 1) 또는 (T,)
        hr_gt:    Tensor,    # (B,) 또는 스칼라 형태의 HR GT (사용하지 않을 수 있음)
        epoch:    int,       # 현재 epoch (사용하지 않을 수 있음)
        FS:       int,       # 샘플링 주파수 (주파수 손실 계산 시 사용)
        diff_flag: bool,     # DiffNormalized 여부 (주파수 손실 내부에서 사용)
        label_mean: float,   # 이 시퀀스(클립) 복원을 위한 μ
        label_std:  float    # 이 시퀀스(클립) 복원을 위한 σ
    ) -> Tensor:
        # ── 배치 모드: pred_std.dim() == 3 → (B, T, 1)
        if pred_std.dim() == 3:
            B, T, _ = pred_std.size()

            # 1) Time-domain MSE
            pred_2d = pred_std.squeeze(-1)  # (B, T)
            gt_2d   = gt_std.squeeze(-1)    # (B, T)
            mse_loss = F.mse_loss(
                pred_2d,
                gt_2d,
                reduction='none'
            ).mean(dim=1)  # (B,)

            # 2) Pearson loss
            pearson_losses = []
            for i in range(B):
                pearson_losses.append(
                    self.pearson(pred_std[i], gt_std[i])  # (T,1) → 스칼라
                )
            pearson_loss = torch.stack(pearson_losses).view(B)  # (B,)

            # 3) Frequency-domain 손실
            pred_flat = pred_2d * label_std + label_mean  # (B, T)
            gt_flat   = gt_2d   * label_std + label_mean  # (B, T)

            freq_losses = []
            for i in range(B):
                ce_i, kl_i = TorchLossComputer.Frequency_loss(
                    pred_flat[i],     # (T,) 텐서
                    gt_flat[i],       # (T,) 텐서
                    diff_flag,
                    FS,
                    self.sigma
                )
                freq_losses.append(ce_i + kl_i)  # 스칼라
            freq_losses = torch.stack(freq_losses).view(B)  # (B,)

            # 4) 최종 손실 벡터
            loss_vec = (
                self.alpha_time * mse_loss +
                self.alpha_corr * pearson_loss +
                self.alpha_freq * freq_losses
            )  # (B,)
            return loss_vec  # (B,)

        # ── 단일 시퀀스 모드: pred_std.dim() == 2 또는 1
        elif pred_std.dim() in {2, 1}:
            if pred_std.dim() == 2:
                T = pred_std.size(0)
                pred_seq = pred_std.unsqueeze(0)  # (1, T, 1)
                gt_seq   = gt_std.unsqueeze(0)    # (1, T, 1)
            else:
                T = pred_std.size(0)
                pred_seq = pred_std.view(1, T, 1)
                gt_seq   = gt_std.view(1, T, 1)

            if isinstance(hr_gt, torch.Tensor):
                hr_val = hr_gt.detach().cpu()
                if hr_val.numel() == 1:
                    hr_seq = torch.tensor([hr_val.item()], device=pred_std.device)
                else:
                    hr_seq = hr_val.view(-1)[:1].to(pred_std.device)
            else:
                hr_seq = torch.tensor([float(hr_gt)], device=pred_std.device)

            loss_vec = self.forward(
                pred_seq,
                gt_seq,
                hr_seq,
                epoch,
                FS,
                diff_flag,
                label_mean,
                label_std
            )  # (1,)
            return loss_vec.squeeze(0)

        else:
            raise ValueError(f"PhysMambaLoss: pred_std.dim()={pred_std.dim()} 은 지원하지 않습니다.")
