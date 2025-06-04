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
    @staticmethod
    def Frequency_loss(pred_tensor: torch.Tensor,
                       gt_tensor: torch.Tensor,
                       diff_flag: bool,
                       Fs: int,
                       std: float) -> (torch.Tensor, torch.Tensor):
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
            # 단일 시퀀스 처리
            return TorchLossComputer._single_freq_loss(pred_tensor, gt_tensor, diff_flag, Fs, std)
        elif pred_tensor.dim() == 2:
            # 배치 처리 (B, T)
            B = pred_tensor.size(0)
            ce_list = []
            kl_list = []
            for i in range(B):
                ce_i, kl_i = TorchLossComputer._single_freq_loss(
                    pred_tensor[i], gt_tensor[i], diff_flag, Fs, std
                )
                ce_list.append(ce_i)
                kl_list.append(kl_i)
            ce_batch = torch.stack(ce_list).mean()
            kl_batch = torch.stack(kl_list).mean()
            return ce_batch, kl_batch
        else:
            raise ValueError(f"Frequency_loss: pred_tensor.dim()={pred_tensor.dim()} 을 지원하지 않습니다.")

    @staticmethod
    def _single_freq_loss(pred_1d: torch.Tensor,
                          gt_1d: torch.Tensor,
                          diff_flag: bool,
                          Fs: int,
                          std: float) -> (torch.Tensor, torch.Tensor):
        """
        1D PPG 신호 하나에 대해 주파수 손실을 계산합니다.

        Args:
            pred_1d:   (T,) GPU float Tensor
            gt_1d:     (T,) GPU float Tensor
            diff_flag: bool, True면 차분된 상태 → 누적 후 Detrend
            Fs:        int, 샘플링 주파수
            std:       float, Gaussian sigma for soft target
        """
        # 1) Thin‐plate spline 기반 Detrend (GPU)
        lam = 100.0
        x = TorchLossComputer._detrend_torch(pred_1d, lam, diff_flag)  # (T,)
        y = TorchLossComputer._detrend_torch(gt_1d, lam, diff_flag)  # (T,)

        # 2) FFT → PSD 계산
        n = x.shape[0]
        fft_pred = torch.fft.rfft(x)  # shape = (freq_bins,)
        fft_gt = torch.fft.rfft(y)  # shape = (freq_bins,)
        psd_pred = torch.abs(fft_pred) ** 2  # (freq_bins,)
        psd_gt = torch.abs(fft_gt) ** 2  # (freq_bins,)

        # 3) 주파수 벡터 생성 (rfftfreq)
        freqs = torch.fft.rfftfreq(n, d=1.0 / Fs)  # (freq_bins,)

        # 4) 0.75 ~ 3.0 Hz 마스크
        mask = (freqs >= 0.75) & (freqs <= 3.0)  # (freq_bins,)

        # 5) PSD 정규화 (합 = 1)
        eps = 1e-8
        pred_sum = psd_pred.sum() + eps
        gt_sum = psd_gt.sum() + eps
        psd_pred_norm = psd_pred / pred_sum  # (freq_bins,)
        psd_gt_norm = psd_gt / gt_sum  # (freq_bins,)

        # 6) CrossEntropy Loss (마스크 구간만)
        #    - 실질적으로는 - sum( gt * log(pred) )
        ce_loss = - (psd_gt_norm[mask] * torch.log(psd_pred_norm[mask] + eps)).sum()

        # 7) KL‐divergence Loss: KL(pred || gt) = sum( pred * log(pred/gt) )
        #    torch.nn.functional.kl_div 입력: input_log_probs, target_probs
        kl_loss = F.kl_div(
            psd_pred_norm[mask].log(),
            psd_gt_norm[mask],
            reduction='sum'
        )

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
            # 먼저 누적합(cumsum) 수행
            signal_1d = torch.cumsum(signal_1d, dim=0)

        T = signal_1d.shape[0]
        device = signal_1d.device
        key = (T, lam, device)

        if key not in _DETREND_CACHE:
            # 1) 정체 행렬 I
            I = torch.eye(T, device=device)  # (T, T)

            # 2) D 행렬 ((T-2) x T)
            D = torch.zeros((T - 2, T), device=device)
            idx = torch.arange(T - 2, device=device)
            D[idx, idx] = 1.0
            D[idx, idx + 1] = -2.0
            D[idx, idx + 2] = 1.0

            # 3) M = I + λ^2 (D^T D)
            M = I + (lam ** 2) * (D.t() @ D)  # (T, T)

            # 4) invM = inv(M)
            invM = torch.inverse(M)  # (T, T)

            # 5) P = I - invM
            P = I - invM  # (T, T)

            # 캐시에 저장
            _DETREND_CACHE[key] = P
        else:
            P = _DETREND_CACHE[key]

        # 6) P @ signal_1d → Detrended signal 반환
        return P @ signal_1d
