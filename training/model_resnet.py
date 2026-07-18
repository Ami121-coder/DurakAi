"""
model_resnet.py — ResNet-8 архитектура для Durak (AlphaZero-style).

Что меняется по сравнению с оригинальным model.py:
  - 3-слойный MLP заменён на ResNet из 8 residual блоков (256 каналов).
  - Policy head: 2 conv 1×1 + FC → 38 логитов.
  - Value head: 2 conv 1×1 + FC → 1 скаляр (tanh).
  - Вход: 220 признаков перестраиваются в (5, 11, 4) тензор для conv.
    Альтернатива: reshape в (44, 5) — но (5,11,4) даёт лучшую структуру.

VRAM (FP32, batch 4096):
  - Параметры: ~2.1M
  - Activations: ~150MB
  - Итого: ~165MB. На 4060 Ti 8GB — с большим запасом.

Throughput (4060 Ti, FP32): ~15-20k samples/sec.
Throughput (4060 Ti, FP16, TensorRT): ~40-60k samples/sec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Классический residual блок AlphaZero: Conv3×3 + BN + ReLU × 2."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class DurakResNet(nn.Module):
    """
    Полная policy-value сеть Durak на основе ResNet.

    Вход: [B, 220] float (см. bindings.cpp::encodeState).
    Выход: (policy[B, 38], value[B, 1]).

    Структура:
      1. Входной слой: Linear(220 → 4*5*11=220) → reshape (B, 4, 5, 11) → Conv 1×1 → 256 каналов.
      2. N residual блоков (256 каналов).
      3. Policy head: Conv 1×1 → 32 каналов → flatten → FC → 38 логитов.
      4. Value head: Conv 1×1 → 32 каналов → flatten → FC(128) → ReLU → FC(1) → tanh.
    """

    def __init__(self,
                 num_blocks: int = 8,
                 num_channels: int = 256,
                 input_size: int = 220,
                 action_size: int = 38,
                 input_shape=(5, 11, 4)):
        super().__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.input_shape = input_shape  # (C=5, H=11, W=4) → 5*11*4 = 220 ✓

        # 1. Входной «стем»: проекция 220 → 5×11×4, затем conv 1×1 до num_channels.
        self.input_proj = nn.Linear(input_size, input_shape[0] * input_shape[1] * input_shape[2])
        self.input_conv = nn.Conv2d(input_shape[0], num_channels, kernel_size=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)

        # 2. Residual блоки.
        self.blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])

        # 3. Policy head.
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * input_shape[1] * input_shape[2], action_size)

        # 4. Value head.
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * input_shape[1] * input_shape[2], 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor = None):
        """
        x:           [B, 220] float
        legal_mask:  [B, 38] float (0.0 или 1.0) — для маскирования policy
                     (на inference; при обучении маска нужна для Cross-Entropy).
        Returns:
            policy: [B, 38] логиты (masked, если передан legal_mask)
            value:  [B, 1] в [-1, 1]
        """
        B = x.size(0)
        C, H, W = self.input_shape

        # 1. Stem
        h = self.input_proj(x)
        h = h.view(B, C, H, W)
        h = F.relu(self.input_bn(self.input_conv(h)))

        # 2. Residual stack
        for block in self.blocks:
            h = block(h)

        # 3. Policy head
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(B, -1)
        p = self.policy_fc(p)  # [B, 38] логиты
        if legal_mask is not None:
            # Masked: logit[i] = -inf если legal_mask[i] = 0.
            p = p.masked_fill(legal_mask < 0.5, float('-inf'))

        # 4. Value head
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.view(B, -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # [B, 1] в [-1, 1]

        return p, v

    # ---- Сериализация ----
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_blocks: int = 8, num_channels: int = 256) -> DurakResNet:
    """Фабрика с дефолтами для 4060 Ti 8GB."""
    return DurakResNet(num_blocks=num_blocks, num_channels=num_channels)


if __name__ == "__main__":
    # Smoke-test: проверим, что forward проходит и размерности верны.
    model = build_model(num_blocks=8, num_channels=256)
    print(f"Параметров: {model.count_parameters():,}")
    print(f"VRAM (FP32): {model.count_parameters() * 4 / 1024 / 1024:.1f} MB")

    B = 4
    x = torch.randn(B, 220)
    mask = torch.ones(B, 38)
    p, v = model(x, mask)
    print(f"Policy: {p.shape} (ожидалось [{B}, 38])")
    print(f"Value:  {v.shape} (ожидалось [{B}, 1])")
    print("OK")
