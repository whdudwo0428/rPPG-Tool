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
    Combined loss:
      L_total = α_time * MSE_std + α_denorm * MSE_denorm
              + α_corr * NegPearson + α_freq * (CE + KL)
    Config-driven: reads all weights and freq-mask from config
    """

    def __init__(self, config):
        super(PhysMambaLoss, self).__init__()
        self.alpha_time = config.TRAIN.LOSS.ALPHA_TIME
        self.alpha_freq = config.TRAIN.LOSS.ALPHA_FREQ
        self.alpha_corr = config.TRAIN.LOSS.ALPHA_CORR
        self.alpha_denorm = config.TRAIN.LOSS.ALPHA_DENORM
        self.sigma = config.TRAIN.LOSS.SIGMA
        self.low_cut = config.TRAIN.LOSS.LOW_CUT
        self.high_cut = config.TRAIN.LOSS.HIGH_CUT
        self.pearson = NegPearson()
        self.alpha_wave_corr = config.TRAIN.LOSS.ALPHA_WAVE_CORR

    def forward(self,
                pred_std: Tensor,
                gt_std: Tensor,
                hr_gt: Tensor,
                epoch: int,
                FS: int,
                diff_flag: bool,
                label_mean: float,
                label_std: float) -> Tensor:

        # ── 배치 모드: pred_std.dim() == 3 → (B, T, 1)
        if pred_std.dim() == 3:
            B, T, _ = pred_std.size()

            # 1) Time-domain MSE (standardized)
            pred_2d = pred_std.squeeze(-1)  # (B, T)
            gt_2d = gt_std.squeeze(-1)  # (B, T)
            mse_std = F.mse_loss(
                pred_2d,
                gt_2d,
                reduction='none'
            ).mean(dim=1)  # (B,)

            # 1.5) Denormalized MSE → “적분된” 원파형 도메인에서 계산
            label_std = label_std.view(-1, 1)  # (B,1)
            label_mean = label_mean.view(-1, 1)  # (B,1)

            # 복원된 신호 계산
            denorm_pred = pred_2d * label_std + label_mean  # (B, T)
            denorm_gt   = gt_2d   * label_std + label_mean  # (B, T)
            if diff_flag:
                # diff-normalized: 복원 후 적분하여 원파형 생성
                waveform_pred = torch.cumsum(denorm_pred, dim=1)
                waveform_gt   = torch.cumsum(denorm_gt,   dim=1)
            else:
                # standardized: 직접 복원된 원파형 비교
                waveform_pred = denorm_pred
                waveform_gt   = denorm_gt

            mse_den = F.mse_loss(
                waveform_pred,
                waveform_gt,
                reduction='none'
            ).mean(dim=1)  # (B,)

            # 2) Pearson loss
            x = pred_std.squeeze(-1)  # (B, T)
            y = gt_std.squeeze(-1)  # (B, T)
            xm = x - x.mean(dim=1, keepdim=True)
            ym = y - y.mean(dim=1, keepdim=True)
            num = (xm * ym).sum(dim=1)
            den = torch.sqrt((xm ** 2).sum(dim=1) * (ym ** 2).sum(dim=1) + 1e-8)
            pearson_loss = 1.0 - (num / den).clamp(-1.0, 1.0)  # (B,)

            # 2.5) waveform-level Pearson loss
            pcm = []
            for i in range(B):
                wp = waveform_pred[i] - waveform_pred[i].mean()
                wg = waveform_gt[i]   - waveform_gt[i].mean()
                num = (wp*wg).sum()
                den = torch.sqrt((wp**2).sum()*(wg**2).sum() + 1e-8)
                corr = (num/den).clamp(-1,1)
                pcm.append(1 - corr)
            wave_corr_loss = torch.stack(pcm)

            # 3) Frequency-domain 손실 → 적분된 원파형 사용
            pred_flat = waveform_pred   # (B, T)
            gt_flat = waveform_gt       # (B, T)

            freq_losses = []
            for i in range(B):
                ce_i, kl_i = TorchLossComputer.Frequency_loss(
                    pred_flat[i],  # (T,) 텐서
                    gt_flat[i],  # (T,) 텐서
                    diff_flag,
                    FS,
                    self.sigma,
                    self.low_cut,  # config에서 읽어온 저주파 컷오프
                    self.high_cut  # config에서 읽어온 고주파 컷오프
                )
                freq_losses.append(ce_i + kl_i)  # 스칼라
            freq_losses = torch.stack(freq_losses).view(B)  # (B,)

            # 4) 최종 손실 벡터
            loss_vec = (
                    self.alpha_time * mse_std +  # 기존 표준화 MSE
                    self.alpha_denorm * mse_den +  # ★ denorm MSE
                    self.alpha_corr * pearson_loss +
                    self.alpha_freq * freq_losses +
                    self.alpha_wave_corr * wave_corr_loss
            )  # (B,)
            return loss_vec  # (B,)

        # ── 단일 시퀀스 모드: pred_std.dim() == 2 또는 1
        elif pred_std.dim() in {2, 1}:
            if pred_std.dim() == 2:
                T = pred_std.size(0)
                pred_seq = pred_std.unsqueeze(0)  # (1, T, 1)
                gt_seq = gt_std.unsqueeze(0)  # (1, T, 1)
            else:
                T = pred_std.size(0)
                pred_seq = pred_std.view(1, T, 1)
                gt_seq = gt_std.view(1, T, 1)

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
