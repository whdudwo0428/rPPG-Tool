# File: neural_methods/loss/TorchLossComputer.py

"""
Adapted from PhysFormer/TorchLossComputer.py, modified to run fully on GPU.
- Thin‐plate spline 기반 Detrend (I - inv(I + λ^2 D^T D)) GPU 캐시
- GPU 상 FFT를 이용한 PSD 계산 → CrossEntropy + KL‐divergence 구간 손실
"""

import torch
import torch.nn.functional as F

# Detrend 연산용 캐시: key = (T, λ, device) → value = P = I - inv(I + λ^2 D^T D)
_DETREND_CACHE = {}


class TorchLossComputer:
    _DETREND_CACHE = {}

    @staticmethod
    def Frequency_loss(pred_tensor: torch.Tensor,
                       gt_tensor: torch.Tensor,
                       diff_flag: bool,
                       Fs: int,
                       std: float,
                       low_cut: float,
                       high_cut: float
                       )-> (torch.Tensor, torch.Tensor):
        """
        주파수 도메인 손실을 완전 GPU에서 계산합니다.

        Args:
            pred_tensor: (T,) 또는 (B, T) 형태의 GPU float Tensor (예측 PPG)
            gt_tensor:   (T,) 또는 (B, T) 형태의 GPU float Tensor (GT PPG)
            diff_flag:   bool, True면 차분(normalized) 신호 → 누적 후 Detrend
            Fs:          int, 샘플링 주파수 (예: 30)
            std:         float, Gaussian sigma for soft target (논문에 맞춰)

        Returns:
            ce_loss: torch.Tensor (스칼라) - CrossEntropy 손실
            kl_loss: torch.Tensor (스칼라) - KL‐divergence 손실
        """
        if pred_tensor.dim() == 1:
            return TorchLossComputer._single_freq_loss(
                pred_tensor, gt_tensor, diff_flag, Fs, std, low_cut, high_cut
            )
        elif pred_tensor.dim() == 2:
            B = pred_tensor.size(0)
            ce_list, kl_list = [], []
            for i in range(B):
                ce_i, kl_i = TorchLossComputer._single_freq_loss(
                    pred_tensor[i], gt_tensor[i], diff_flag,
                    Fs, std, low_cut, high_cut
                )
                ce_list.append(ce_i)
                kl_list.append(kl_i)
            return torch.stack(ce_list).mean(), torch.stack(kl_list).mean()
        else:
            raise ValueError(f"Frequency_loss: pred_tensor.dim()={pred_tensor.dim()} 지원 안 함")

    @staticmethod
    def _single_freq_loss(pred_1d: torch.Tensor,
                          gt_1d: torch.Tensor,
                          diff_flag: bool,
                          Fs: int,
                          std: float,
                          low_cut: float,
                          high_cut: float) -> (torch.Tensor, torch.Tensor):
        """
        1D PPG 신호 하나에 대해 주파수 손실을 계산합니다.

        Args:
            pred_1d:   (T,) GPU float Tensor
            gt_1d:     (T,) GPU float Tensor
            diff_flag: bool, True면 차분된 상태 → 누적 후 Detrend
            Fs:        int, 샘플링 주파수
            std:       float, Gaussian sigma for soft target
        """
        # 1) Detrend
        lam = 100.0
        x = TorchLossComputer._detrend_torch(pred_1d, lam, diff_flag)
        y = TorchLossComputer._detrend_torch(gt_1d, lam, diff_flag)

        # 2) FFT → PSD
        n = x.shape[0]
        x_float = x.to(torch.float32) if x.dtype == torch.float16 else x
        y_float = y.to(torch.float32) if y.dtype == torch.float16 else y
        psd_pred = torch.abs(torch.fft.rfft(x_float, dim=-1)) ** 2
        psd_gt = torch.abs(torch.fft.rfft(y_float, dim=-1)) ** 2

        # 3) freqs
        freqs = torch.fft.rfftfreq(n, d=1.0 / Fs, device=pred_1d.device)  # GPU에서 생성

        # 4) low_cut ~ high_cut 마스크
        mask = (freqs >= low_cut) & (freqs <= high_cut)

        # 5) 정규화
        eps = 1e-8
        psd_pred_norm = psd_pred / (psd_pred.sum() + eps)
        psd_gt_norm = psd_gt / (psd_gt.sum() + eps)

        # 6) soft target (Gaussian around peak)
        masked_gt = psd_gt_norm * mask.float()
        peak_idx = torch.argmax(masked_gt)
        f0 = freqs[peak_idx]
        gauss = torch.exp(-(freqs - f0) ** 2 / (2 * (std ** 2))) * mask.float()
        target = gauss / (gauss.sum() + eps)

        # 7) CE + KL
        ce_loss = - (target[mask] * torch.log(psd_pred_norm[mask] + eps)).sum()
        kl_loss = F.kl_div(psd_pred_norm[mask].log(), target[mask], reduction='sum')

        return ce_loss, kl_loss

    @staticmethod
    def _detrend_torch(signal_1d: torch.Tensor,
                       lam: float,
                       diff_flag: bool) -> torch.Tensor:
        """
        Thin‐plate spline 기반 Detrend(박막 평활) 연산을 GPU에서 수행합니다.

        P = I - inv(I + λ^2 D^T D)
        - diff_flag == True: 차분(normalized) 상태 → 누적합 → Detrend
        - diff_flag == False: 바로 Detrend
        - (T, λ, device)별로 P 행렬을 캐시하여 이후 재사용
        """
        if diff_flag:
            signal_1d = torch.cumsum(signal_1d, dim=0)

        T, device = signal_1d.shape[0], signal_1d.device
        key = (T, lam, device)
        if key not in TorchLossComputer._DETREND_CACHE:
            I = torch.eye(T, device=device, dtype=signal_1d.dtype)
            D = torch.zeros((T - 2, T), device=device, dtype=signal_1d.dtype)
            idx = torch.arange(T - 2, device=device)
            D[idx, idx] = 1.0
            D[idx, idx + 1] = -2.0
            D[idx, idx + 2] = 1.0
            M = I + (lam ** 2) * (D.t() @ D)
            invM = torch.inverse(M)
            TorchLossComputer._DETREND_CACHE[key] = I - invM
        P = TorchLossComputer._DETREND_CACHE[key]
        return P @ signal_1d
