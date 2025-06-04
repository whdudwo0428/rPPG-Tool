# File: neural_methods/model/PhysMamba.py

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from torch import nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_, DropPath
from einops import rearrange

from mamba_ssm.modules.SA_Mamba2_simple import Mamba2Simple as SA_Mamba
from mamba_ssm.modules.CA_Mamba2_simple import Mamba2Simple as CA_Mamba


class CAFusion_Stem(nn.Module):
    def __init__(self, apha=0.5, belta=0.5, dim=24):
        super(CAFusion_Stem, self).__init__()
        # ── 채널 수를 init 시점에 고정할 수 없으므로, forward에서 자동 감지/교정한다 ──

        # “stem11”: 원본 채널 수(C) → dim//2 로 올리기
        # 그렇게 하기 위해서, 여기서는 in_channels=3으로 두되,
        # 실제 forward 도중 입력 채널 수가 3이 아니라면 보기 좋게 “재생성” 로직을 구현한다.
        self.stem11 = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # “stem12”: 차분/병합 결과 채널 수(=4*C) → dim//2 로 줄이기 (여기서는 4*3=12로 두되, forward에서 교정)
        self.stem12 = nn.Sequential(
            nn.Conv2d(12, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.apha = apha
        self.belta = belta

    def forward(self, x):
        # x가 (B, D, C, H, W)가 아니라 (B, D, H, W, C)로 들어올 수 있으므로, 자동 판별 후 교정
        # (통상 C=3이므로, 두번째 차원(인덱스2)이 네 번째 차원(인덱스3)보다 작으면 이미 (B,D,C,H,W))
        if x.dim() == 5 and x.shape[2] >= x.shape[3]:
            # 예: x.shape == (B, D, H, W, C)  → (B, D, C, H, W) 로 permute
            x = x.permute(0, 1, 4, 2, 3).contiguous()

        # 그 후에야 안전하게 shape 추출
        N, D, C, H, W = x.shape

        # ── 차분 병합(diff_cat) 계산 ──
        x1 = torch.cat([x[:, :1], x[:, :1], x[:, :D - 2]], dim=1)
        x2 = torch.cat([x[:, :1], x[:, :D - 1]], dim=1)
        x3 = x
        x4 = torch.cat([x[:, 1:], x[:, D - 1:]], dim=1)
        x5 = torch.cat([x[:, 2:], x[:, D - 1:], x[:, D - 1:]], dim=1)

        diff_cat = torch.cat([
            x2 - x1,
            x3 - x2,
            x4 - x3,
            x5 - x4
        ], dim=2).contiguous()  # → (N, D, 4*C, H, W)

        # ── diff_cat → stem12 로 넘길 텐서 형태로 변환 ──
        #    기존: diff_cat.view(N*D, 12, H, W)  # (12 = 4*3)
        #    이제: 채널 수가 '4*C'인 점을 반영
        in_ch = 4 * C      # 실제 채널 수
        diff_cat = diff_cat.view(N * D, in_ch, H, W)

        # 만약 'in_ch != 12'라면, 새로운 Conv2d 계층을 생성해야 하므로
        if in_ch != 12:
            # 간단히 기존 stem12를 “교체”하여 올바른 입력 채널 수를 받도록 업데이트
            self.stem12 = nn.Sequential(
                nn.Conv2d(in_ch, self.stem12[0].out_channels,
                          kernel_size=self.stem12[0].kernel_size,
                          stride=self.stem12[0].stride,
                          padding=self.stem12[0].padding),
                nn.BatchNorm2d(self.stem12[0].out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ).to(x.device)

        x_diff = self.stem12(diff_cat)  # (N*D, dim//2, H/8, W/8)

        # ── 원본 경로(stem11) 처리를 위한 view ──
        x3_reshaped = x3.contiguous().view(N * D, C, H, W)
        x_path0 = self.stem11(x3_reshaped)  # (N*D, dim//2, H/8, W/8)

        x_path1 = self.apha * x_path0 + self.belta * x_diff
        x_path1 = self.stem21(x_path1)
        x_path2 = self.stem22(x_diff)

        out = self.apha * x_path1 + self.belta * x_path2
        return out  # (N*D, dim, H/8, W/8)


class SAFusion_Stem(nn.Module):
    def __init__(self, apha=0.5, belta=0.5, dim=24):
        super(SAFusion_Stem, self).__init__()
        self.stem11 = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stem12 = nn.Sequential(
            nn.Conv2d(12, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.apha = apha
        self.belta = belta

    def forward(self, x):
        # (B, D, H, W, C) → (B, D, C, H, W) 필요 시 자동 교정
        if x.dim() == 5 and x.shape[2] >= x.shape[3]:
            x = x.permute(0, 1, 4, 2, 3).contiguous()

        N, D, C, H, W = x.shape

        x1 = torch.cat([x[:, :1], x[:, :1], x[:, :D - 2]], dim=1)
        x2 = torch.cat([x[:, :1], x[:, :D - 1]], dim=1)
        x3 = x
        x4 = torch.cat([x[:, 1:], x[:, D - 1:]], dim=1)
        x5 = torch.cat([x[:, 2:], x[:, D - 1:], x[:, D - 1:]], dim=1)

        diff_cat = torch.cat([
            x2 - x1,
            x3 - x2,
            x4 - x3,
            x5 - x4
        ], dim=2).contiguous()  # (N, D, 4*C, H, W)

        in_ch = 4 * C
        diff_cat = diff_cat.view(N * D, in_ch, H, W)

        if in_ch != 12:
            # stem12의 입력 채널 수를 “in_ch”로 교체
            self.stem12 = nn.Sequential(
                nn.Conv2d(in_ch, self.stem12[0].out_channels,
                          kernel_size=self.stem12[0].kernel_size,
                          stride=self.stem12[0].stride,
                          padding=self.stem12[0].padding),
                nn.BatchNorm2d(self.stem12[0].out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ).to(x.device)

        x_diff = self.stem12(diff_cat)  # (N*D, dim//2, H/8, W/8)

        x3_reshaped = x3.contiguous().view(N * D, C, H, W)
        x_path0 = self.stem11(x3_reshaped)  # (N*D, dim//2, H/8, W/8)

        x_path1 = self.apha * x_path0 + self.belta * x_diff
        x_path1 = self.stem21(x_path1)
        x_path2 = self.stem22(x_diff)

        out = self.apha * x_path1 + self.belta * x_path2
        return out  # (N*D, dim, H/8, W/8)


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        # x: (B, C, T, H, W)
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / (xsum + 1e-6) * xshape[3] * xshape[4] * 0.5


class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.scale = 0.02
        self.dim = dim * mlp_ratio

        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        # (B, N, C)
        B, N, C = x.shape
        device = x.device

        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)  # → (B, N, dim*mlp)
        try:
            x_fre = torch.fft.fft(x, dim=1, norm='ortho')
            x_real = F.relu(
                torch.einsum('bnc,cc->bnc', x_fre.real, self.r.to(device))
                - torch.einsum('bnc,cc->bnc', x_fre.imag, self.i.to(device))
                + self.rb.to(device)
            )
            x_imag = F.relu(
                torch.einsum('bnc,cc->bnc', x_fre.imag, self.r.to(device))
                + torch.einsum('bnc,cc->bnc', x_fre.real, self.i.to(device))
                + self.ib.to(device)
            )
            x_fre = torch.stack([x_real, x_imag], dim=-1).float()
            x_fre = torch.view_as_complex(x_fre)
            x = torch.fft.ifft(x_fre, dim=1, norm='ortho').real.to(torch.float32).to(device)
        except RuntimeError:
            x_cpu = x.detach().cpu()
            x_fre_cpu = torch.fft.fft(x_cpu, dim=1, norm='ortho')
            r_cpu, i_cpu = self.r.detach().cpu(), self.i.detach().cpu()
            rb_cpu, ib_cpu = self.rb.detach().cpu(), self.ib.detach().cpu()
            x_real_cpu = F.relu(
                torch.einsum('bnc,cc->bnc', x_fre_cpu.real, r_cpu)
                - torch.einsum('bnc,cc->bnc', x_fre_cpu.imag, i_cpu)
                + rb_cpu
            )
            x_imag_cpu = F.relu(
                torch.einsum('bnc,cc->bnc', x_fre_cpu.imag, r_cpu)
                + torch.einsum('bnc,cc->bnc', x_fre_cpu.real, i_cpu)
                + ib_cpu
            )
            x_fre_cpu = torch.stack([x_real_cpu, x_imag_cpu], dim=-1).float()
            x_fre_cpu = torch.view_as_complex(x_fre_cpu)
            x = torch.fft.ifft(x_fre_cpu, dim=1, norm='ortho').real.to(torch.float32).to(device)

        x = x.transpose(1, 2).contiguous()  # (B, C*mlp, N)
        x = self.fc2(x)                     # (B, C, N)
        x = x.transpose(1, 2).contiguous()  # (B, N, C)
        return x  # (B, N, C)


class CA_MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = CA_Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x, C_SA):
        if C_SA.ndim == 2:
            L_sub, d_state = C_SA.shape
            B_sub = x.size(0)
            C_SA = C_SA.unsqueeze(0).expand(B_sub, -1, -1).contiguous()

        assert C_SA.ndim == 3, f"CA_MambaLayer: C_SA must be 3D, got {C_SA.ndim}D."
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm, C_SA)
        return x_mamba  # (B_sub, L_sub, dim)


class SA_MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = SA_Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        x_norm = self.norm(x)
        x_mamba, C_fast = self.mamba(x_norm)
        return x_mamba, C_fast


class CABlock_CrossMamba(nn.Module):
    def __init__(self, dim, mlp_ratio, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CA_MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, C_SA_list):
        C0 = C_SA_list[0]
        x_o = self.drop_path(self.attn(x, C0))

        B, L_sub, _ = x.size()
        x_path = torch.zeros_like(x, device=x.device)
        for lvl, factor in enumerate([2, 4, 8], start=1):
            tt = L_sub // factor
            for j in range(factor):
                Cj = C_SA_list[lvl][j]
                x_div = self.attn(x[:, j * tt:(j + 1) * tt, :], Cj)
                x_path[:, j * tt:(j + 1) * tt, :] = x_div
            x_o = x_o + self.drop_path(x_path)

        x = x + self.drop_path(self.norm1(x_o))
        x = x + self.drop_path(self.mlp(self.norm2(x).contiguous()))
        return x


class SABlock_CrossMamba(nn.Module):
    def __init__(self, dim, mlp_ratio, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = SA_MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, L_sub, _ = x.size()
        device = x.device

        x_o, C0 = self.attn(x)  # (B, L_sub, dim), (B, L_sub, d_state)
        C_SA_list = [C0.clone().to(device), [], [], []]
        x_o = self.drop_path(x_o)

        # Level 1~3
        tt = L_sub // 2
        x_path = torch.zeros_like(x, device=device)
        for j in range(2):
            x_div, Cj = self.attn(x[:, j * tt:(j + 1) * tt, :])
            C_SA_list[1].append(Cj.clone().to(device))
            x_path[:, j * tt:(j + 1) * tt, :] = x_div
        x_o = x_o + self.drop_path(x_path)

        tt = L_sub // 4
        x_path.zero_()
        for j in range(4):
            x_div, Cj = self.attn(x[:, j * tt:(j + 1) * tt, :])
            C_SA_list[2].append(Cj.clone().to(device))
            x_path[:, j * tt:(j + 1) * tt, :] = x_div
        x_o = x_o + self.drop_path(x_path)

        tt = L_sub // 8
        x_path.zero_()
        for j in range(8):
            x_div, Cj = self.attn(x[:, j * tt:(j + 1) * tt, :])
            C_SA_list[3].append(Cj.clone().to(device))
            x_path[:, j * tt:(j + 1) * tt, :] = x_div
        x_o = x_o + self.drop_path(x_path)

        x = x + self.drop_path(self.norm1(x_o))
        x = x + self.drop_path(self.mlp(self.norm2(x).contiguous()))

        return x, C_SA_list


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p.div_(math.sqrt(n_residuals_per_layer * n_layer))


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class PhysMamba(nn.Module):
    def __init__(
        self,
        depth=24,
        embed_dim_CA=64,
        embed_dim_SA=64,
        mlp_ratio=2,
        drop_rate=0.0,
        drop_path_rate_CA=0.1,
        drop_path_rate_SA=0.1,
        initializer_cfg=None,
        device=None,
        dtype=None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super().__init__()

        # === CA Layers ===
        self.embed_dim_CA = embed_dim_CA
        self.Fusion_Stem_CA = CAFusion_Stem(dim=embed_dim_CA // 4)
        self.attn_mask_CA = Attention_mask()
        self.stem3_CA = nn.Sequential(
            nn.Conv3d(embed_dim_CA // 4, embed_dim_CA, kernel_size=(2, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(embed_dim_CA),
        )
        dpr_CA = [x.item() for x in torch.linspace(0, drop_path_rate_CA, depth)]
        inter_dpr_CA = [0.0] + dpr_CA
        self.blocks_CrossCA = nn.ModuleList([
            CABlock_CrossMamba(dim=embed_dim_CA, mlp_ratio=mlp_ratio, drop_path=inter_dpr_CA[i], norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])

        # === SA Layers ===
        self.embed_dim_SA = embed_dim_SA
        self.Fusion_Stem_SA = SAFusion_Stem(dim=embed_dim_SA // 4)
        self.attn_mask_SA = Attention_mask()
        self.stem3_SA = nn.Sequential(
            nn.Conv3d(embed_dim_SA // 4, embed_dim_SA, kernel_size=(2, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(embed_dim_SA),
        )
        dpr_SA = [x.item() for x in torch.linspace(0, drop_path_rate_SA, depth)]
        inter_dpr_SA = [0.0] + dpr_SA
        self.blocks_CrossSA = nn.ModuleList([
            SABlock_CrossMamba(dim=embed_dim_SA, mlp_ratio=mlp_ratio, drop_path=inter_dpr_SA[i], norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])

        # === Prediction Head ===
        self.ConvBlockLast = nn.Conv1d(embed_dim_SA + embed_dim_CA, 1, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2)

        # 가중치 초기화
        self.apply(segm_init_weights)
        self.apply(
            partial(_init_weights,
                    n_layer=depth,
                    **(initializer_cfg if initializer_cfg is not None else {}))
        )

    def forward(self, x):
        """
        x: (B, D, C, H, W) 또는 (B, D, H, W, C)
        """
        B, D = x.shape[0], x.shape[1]
        device = x.device

        # === SA 경로 전처리 ===
        x_SA = self.Fusion_Stem_SA(x)  # (B*D, embed_dim_SA//4, H/8, W/8)
        # ── 원본 코드에서 아래 두 줄이 모든 배치마다 출력되어 너무 verbose했기 때문에 주석 처리했습니다.
        # print(f"[Debug][forward] x_SA shape before pooling = {tuple(x_SA.shape)}")
        x_SA = x_SA.view(B, D, self.embed_dim_SA // 4, x_SA.shape[-2], x_SA.shape[-1]) \
                 .permute(0, 2, 1, 3, 4)       # (B, dim//4, D, H/8, W/8)
        x_SA = self.stem3_SA(x_SA)            # (B, embed_dim_SA, D/2, H/8, W/8)

        mask_f = torch.sigmoid(x_SA)
        mask_f = self.attn_mask_SA(mask_f)
        x_SA = x_SA * mask_f
        x_SA = torch.mean(x_SA, dim=4)  # (B, embed_dim_SA, D/2, H/8)
        x_SA = torch.mean(x_SA, dim=3)  # (B, embed_dim_SA, D/2)
        x_SA = rearrange(x_SA, 'b c t -> b t c')  # (B, D/2, embed_dim_SA)

        # === CA 경로 전처리 ===
        x_CA = self.Fusion_Stem_CA(x)  # (B*D, embed_dim_CA//4, H/8, W/8)
        # ── 원본 코드에서 아래 두 줄이 모든 배치마다 출력되어 너무 verbose했기 때문에 주석 처리했습니다.
        # print(f"[Debug][forward] x_CA shape before pooling = {tuple(x_CA.shape)}")
        x_CA = x_CA.view(B, D, self.embed_dim_CA // 4, x_CA.shape[-2], x_CA.shape[-1]) \
                 .permute(0, 2, 1, 3, 4)       # (B, dim//4, D, H/8, W/8)
        x_CA = self.stem3_CA(x_CA)            # (B, embed_dim_CA, D/2, H/8, W/8)

        mask_s = torch.sigmoid(x_CA)
        mask_s = self.attn_mask_CA(mask_s)
        x_CA = x_CA * mask_s
        x_CA = torch.mean(x_CA, dim=4)  # (B, embed_dim_CA, D/2, H/8)
        x_CA = torch.mean(x_CA, dim=3)  # (B, embed_dim_CA, D/2)
        x_CA = rearrange(x_CA, 'b c t -> b t c')  # (B, D/2, embed_dim_CA)

        # === Dual-Stream SSD: SA → CA 순차 처리 ===
        C_SA_list_all = []
        num_sa_blocks = len(self.blocks_CrossSA)

        # 1) SA 블록: C_fast(레벨별) 수집
        for idx, sa_blk in enumerate(self.blocks_CrossSA):
            x_SA, C_SA_list = sa_blk(x_SA)
            C_SA_list_all.append(C_SA_list)

        # 2) CA 블록
        for idx, ca_blk in enumerate(self.blocks_CrossCA):
            if idx < len(C_SA_list_all):
                C_SA_list = C_SA_list_all[idx]
                x_CA = ca_blk(x_CA, C_SA_list)  # (B, D/2, embed_dim_CA)

        # === Fusion & Prediction ===
        x_concat = torch.cat((x_SA, x_CA), dim=2)      # (B, D/2, embed_dim_SA+embed_dim_CA)
        rPPG = x_concat.permute(0, 2, 1).contiguous()  # (B, C_total, D/2)
        rPPG = self.upsample(rPPG).contiguous()        # (B, C_total, D)
        rPPG = self.ConvBlockLast(rPPG)                # (B, 1, D)
        rPPG = rPPG.squeeze(1)                         # (B, D)

        return rPPG
