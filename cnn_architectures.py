# cnn_architectures.py
# ---------------------------------------------------------------------------
# Stronger-but-still-lightweight CNN baselines for the LOSO comparison.
#
# Motivation (external reviews, unanimous): the thesis's SimpleEMGCNN (3 plain
# conv blocks, 32-64-128) is a 2015-era design, so "classical beats deep under
# LOSO" is confounded by architectural under-investment. This module adds a
# modern-but-compact alternative: a 1D residual network with squeeze-and-excite
# (SE) channel attention. It keeps the SAME input (9 x 500 envelopes), the SAME
# per-subject normalisation and LOSO protocol, so ONLY the architecture changes.
# It is a fairer deep baseline, not a bid for state of the art.
# ---------------------------------------------------------------------------
from __future__ import annotations
import torch
import torch.nn as nn


class SEBlock1d(nn.Module):
    """Squeeze-and-excitation channel attention (Hu et al., 2018), 1D."""
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        hidden = max(ch // r, 4)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, ch)

    def forward(self, x):                      # x: (N, C, T)
        s = x.mean(dim=-1)                      # global average pool over time -> (N, C)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)             # re-weight channels


class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, k=7, stride=1, use_se=True, p=0.1):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.se = SEBlock1d(cout) if use_se else nn.Identity()
        self.drop = nn.Dropout(p)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                      nn.BatchNorm1d(cout))

    def forward(self, x):
        idn = x if self.down is None else self.down(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.drop(torch.relu(out + idn))


class EMGResNet1D(nn.Module):
    """Compact 1D ResNet with optional SE attention. ~0.3M params at defaults."""
    def __init__(self, in_ch: int, n_classes: int, widths=(32, 64, 128),
                 blocks_per_stage: int = 2, use_se: bool = True, k: int = 7, p: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, widths[0], 9, padding=4, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU(), nn.MaxPool1d(2))
        layers, cin = [], widths[0]
        for si, w in enumerate(widths):
            for bi in range(blocks_per_stage):
                stride = 2 if (bi == 0 and si > 0) else 1
                layers.append(ResBlock1d(cin, w, k=k, stride=stride, use_se=use_se, p=p))
                cin = w
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(cin, n_classes))
        self.feat_dim = cin

    def features(self, x):                     # penultimate embedding (for Deep CORAL)
        x = self.stem(x); x = self.body(x); x = self.pool(x)
        return torch.flatten(x, 1)             # (N, feat_dim)

    def forward(self, x, return_feat=False):
        f = self.features(x)
        logits = self.head(f)
        return (logits, f) if return_feat else logits


def build_model(arch: str, in_ch: int, n_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "simple":
        from train_cnn_loso import SimpleEMGCNN
        return SimpleEMGCNN(in_ch=in_ch, n_classes=n_classes)
    if arch in ("resnet_se", "resnetse", "se"):
        return EMGResNet1D(in_ch, n_classes, use_se=True)
    if arch == "resnet":
        return EMGResNet1D(in_ch, n_classes, use_se=False)
    raise ValueError(f"unknown arch '{arch}' (use simple | resnet | resnet_se)")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
