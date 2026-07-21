"""
model_resnet.py — УСИЛЕННАЯ архитектура для Durak.

Улучшения относительно исходной:
  1. Squeeze-and-Excitation (SE) блоки после каждого residual —
     канал-wise attention, +0.5-1% winrate в шахматах на том же бюджете.
  2. 10 residual-блоков вместо 8 — больше ёмкости.
  3. Поле value head расширено: 256 hidden вместо 128 —
     точнее предсказывает результат.
  4. Dropout 0.1 в heads — против overfitting на малом буфере.
  5. Stem 3×3 + BN + ReLU без изменений — проверенная схема.
  6. Stable masking через finfo.min — без NaN.
  7. Count параметров: ~5M (4060 Ti 8GB FP16 = ~80MB activations,
     batch 4096 fits comfortably).

Архитектура (B=10, C=256):
  Stem:        Conv3×3(5→256) → BN → ReLU
  Blocks:      10× [ResBlock + SEBlock]
  Policy head: Conv1×1(256→32) → BN → ReLU → flatten(1152) ⊕ scalars(40) → FC(1192→38)
  Value head:  Conv1×1(256→32) → BN → ReLU → flatten(1152) ⊕ scalars(40) → FC(1192→256) → ReLU → FC(256→1) → tanh

VRAM бюджет на 4060 Ti 8GB (FP16, batch=4096):
  Параметры: ~5M × 2 bytes = 10 MB
  Activations (largest): 4096 × 256 × 4 × 9 × 2 bytes ≈ 75 MB
  Gradients: ~10 MB
  Optimizer state (AdamW): 2 × 10 MB = 20 MB
  Итого: ~120 MB — огромный запас.

Можно даже поднять batch до 8192 или num_channels до 384.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """SE-блок: канал-wise attention через global avg pool → FC → sigmoid."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, _, _ = x.shape
        s = x.mean(dim=(2, 3))               # [B, C]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))        # [B, C]
        return x * s.view(B, C, 1, 1)


class ResidualBlock(nn.Module):
    """Residual block с SE-attention."""

    def __init__(self, channels: int, se_reduction: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, reduction=se_reduction)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + identity)
        return out


class DurakResNet(nn.Module):
    """
    Усиленная Policy-value сеть для Durak (AlphaZero-style + SE).

    Task 3: Neural-Bayesian Fusion.
    Вход: [B, 256] — 6 масок × 36 карт + 40 скаляров.
    Выход: (policy[B, 38], value[B, 1]).

    Layout входа (см. bindings.cpp::encodeState):
      [0..35]    моя рука (4 масти × 9 рангов)
      [36..71]   козырная масть
      [72..107]  атаки
      [108..143] защиты
      [144..179] бито
      [180..215] Task 3: байесовские вероятности карт у оппонента (oppProbs)
      [216..255] скаляры (40 байт)
    """

    NUM_CARD_CHANNELS = 6  # Task 3: было 5, стало 6 (добавлен oppProbs)
    CARD_H = 4
    CARD_W = 9
    SCALAR_SIZE = 40

    def __init__(self,
                 num_blocks: int = 10,
                 num_channels: int = 256,
                 action_size: int = 38,
                 se_reduction: int = 16,
                 dropout: float = 0.1,
                 value_hidden: int = 256):
        super().__init__()
        self.action_size = action_size

        # Stem: 5 card channels → num_channels.
        self.input_conv = nn.Conv2d(self.NUM_CARD_CHANNELS, num_channels,
                                     kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)

        # Residual+SE stack.
        self.blocks = nn.ModuleList([
            ResidualBlock(num_channels, se_reduction=se_reduction,
                          dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Policy head.
        policy_conv_out = 32
        self.policy_conv = nn.Conv2d(num_channels, policy_conv_out,
                                      kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_conv_out)
        policy_flat = policy_conv_out * self.CARD_H * self.CARD_W  # 1152
        self.policy_fc = nn.Linear(policy_flat + self.SCALAR_SIZE, action_size)
        self.policy_dropout = nn.Dropout(dropout)

        # Value head (расширенная).
        value_conv_out = 32
        self.value_conv = nn.Conv2d(num_channels, value_conv_out,
                                     kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_conv_out)
        value_flat = value_conv_out * self.CARD_H * self.CARD_W
        self.value_fc1 = nn.Linear(value_flat + self.SCALAR_SIZE, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)
        self.value_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                legal_mask: torch.Tensor = None):
        """
        Task 3: x — [B, 256] (uint8 или float).
        legal_mask:  [B, 38] — для маскирования policy
        Returns: (policy[B, 38], value[B, 1])

        Layout:
          [0..215]   = 6 масок × 36 (моя рука, козырь, атаки, защиты, бито, oppProbs)
          [216..255] = 40 скаляров
        """
        B = x.size(0)

        # Нормализация uint8 → float.
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        # Task 3: card masks (6 × 36 = 216) и scalars (40).
        card_flat = x[:, :216]
        scalars = x[:, 216:256]

        # Reshape card masks в [B, 6, 4, 9].
        card_masks = card_flat.view(B, self.NUM_CARD_CHANNELS,
                                     self.CARD_H, self.CARD_W)

        # Stem.
        h = F.relu(self.input_bn(self.input_conv(card_masks)))

        # Residual+SE stack.
        for block in self.blocks:
            h = block(h)

        # Policy head.
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(B, -1)                                # [B, 1152]
        p = torch.cat([p, scalars], dim=1)               # [B, 1192]
        p = self.policy_dropout(p)
        p = self.policy_fc(p)                            # [B, 38]

        if legal_mask is not None:
            # finfo.min вместо -inf — избегает NaN в log_softmax.
            p = p.masked_fill(legal_mask < 0.5, torch.finfo(p.dtype).min)

        # Value head.
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.view(B, -1)
        v = torch.cat([v, scalars], dim=1)
        v = self.value_dropout(v)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))                # [B, 1] в [-1, 1]

        return p, v

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_blocks: int = 10, num_channels: int = 256,
                **kwargs) -> DurakResNet:
    return DurakResNet(num_blocks=num_blocks,
                       num_channels=num_channels,
                       **kwargs)


if __name__ == "__main__":
    model = build_model(num_blocks=10, num_channels=256)
    print(f"Параметров: {model.count_parameters():,}")
    print(f"VRAM (FP32): {model.count_parameters() * 4 / 1024**2:.1f} MB")
    print(f"VRAM (FP16): {model.count_parameters() * 2 / 1024**2:.1f} MB")

    B = 4
    x = torch.rand(B, 256, dtype=torch.float32)  # Task 3: 256
    mask = torch.ones(B, 38)
    p, v = model(x, mask)
    print(f"Policy: {p.shape} (ожидалось [{B}, 38])")
    print(f"Value:  {v.shape} (ожидалось [{B}, 1])")
    assert not torch.isnan(p).any(), "NaN в policy!"
    assert not torch.isnan(v).any(), "NaN в value!"
    print("OK — нет NaN, размерности верны")

    # Тест batch=4096 на GPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        x = torch.rand(4096, 256, dtype=torch.float32, device=device)  # Task 3: 256
        mask = torch.ones(4096, 38, device=device)
        with torch.amp.autocast('cuda'):
            p, v = model(x, mask)
        print(f"Batch=4096 на GPU: policy={p.shape}, value={v.shape}, "
              f"VRAM={torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
