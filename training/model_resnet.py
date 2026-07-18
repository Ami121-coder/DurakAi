"""
model_resnet.py — FIX #5: правильная архитектура с семантическим input_shape

КРИТИЧЕСКИЙ БАГ: (5, 11, 4) — фичи не выровнены по каналам.
  Канал 0 содержал: my_hand[0..35] + trump[0..7] — смесь двух масок.
  Conv3×3 учил паттерны между несвязанными фичами. Сходимость медленная.

ФИКС: (5, 4, 9) — 5 семантических каналов × 4 масти × 9 рангов.
  Канал 0: моя рука (бинарная маска 4×9)
  Канал 1: козырная масть (бинарная маска 4×9)
  Канал 2: атаки на столе
  Канал 3: защиты на столе
  Канал 4: бито

  Скалярные фичи (40 байт) подаются отдельной веткой и конкатенируются
  перед policy/value heads.

Дополнительно:
  - Убран input_proj (Linear 220→220) — не нужен, reshape сразу.
  - BatchNorm после каждого conv.
  - masked_fill использует finfo.min вместо -inf (избегает NaN).

ВНИМАНИЕ: новый формат НЕСОВМЕСТИМ со старыми чекпойнтами.
  Требуется переобучение с нуля.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
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
    Policy-value сеть для Durak (AlphaZero-style).

    Вход: [B, 220] — 5 масок × 36 карт + 40 скаляров.
    Выход: (policy[B, 38], value[B, 1]).

    Layout входа (см. bindings.cpp::encodeState):
      [0..35]   моя рука (4 масти × 9 рангов)
      [36..71]  козырная масть
      [72..107] атаки
      [108..143] защиты
      [144..179] бито
      [180..219] скаляры (40 байт)
    """

    NUM_CARD_CHANNELS = 5   # my_hand, trump, atk, def, discard
    CARD_H = 4              # 4 масти
    CARD_W = 9              # 9 рангов (6..A)
    SCALAR_SIZE = 40

    def __init__(self,
                 num_blocks: int = 8,
                 num_channels: int = 256,
                 action_size: int = 38):
        super().__init__()
        self.action_size = action_size

        # Stem: 5 card channels → num_channels
        self.input_conv = nn.Conv2d(self.NUM_CARD_CHANNELS, num_channels,
                                     kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])

        # Policy head: conv → flatten → FC (с конкатенацией скаляров)
        policy_conv_out = 32
        self.policy_conv = nn.Conv2d(num_channels, policy_conv_out, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_conv_out)
        policy_flat = policy_conv_out * self.CARD_H * self.CARD_W  # 32 * 4 * 9 = 1152
        self.policy_fc = nn.Linear(policy_flat + self.SCALAR_SIZE, action_size)

        # Value head: conv → flatten → FC(128) → ReLU → FC(1) → tanh
        value_conv_out = 32
        self.value_conv = nn.Conv2d(num_channels, value_conv_out, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_conv_out)
        value_flat = value_conv_out * self.CARD_H * self.CARD_W
        self.value_fc1 = nn.Linear(value_flat + self.SCALAR_SIZE, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor = None):
        """
        x:           [B, 220] — uint8 или float
        legal_mask:  [B, 38] — для маскирования policy
        Returns: (policy[B, 38], value[B, 1])
        """
        B = x.size(0)

        # Нормализуем: uint8 [0..255] → float [0..1]
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()

        # Разделяем на card masks (5 × 36 = 180) и scalars (40)
        card_flat = x[:, :180]                          # [B, 180]
        scalars = x[:, 180:220]                         # [B, 40]

        # Reshape card masks в [B, 5, 4, 9]
        card_masks = card_flat.view(B, self.NUM_CARD_CHANNELS, self.CARD_H, self.CARD_W)

        # Stem
        h = F.relu(self.input_bn(self.input_conv(card_masks)))

        # Residual stack
        for block in self.blocks:
            h = block(h)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(B, -1)                               # [B, 1152]
        p = torch.cat([p, scalars], dim=1)              # [B, 1192]
        p = self.policy_fc(p)                            # [B, 38]

        if legal_mask is not None:
            # Используем finfo.min вместо -inf — avoids NaN в log_softmax
            p = p.masked_fill(legal_mask < 0.5, torch.finfo(p.dtype).min)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.view(B, -1)
        v = torch.cat([v, scalars], dim=1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))                # [B, 1] в [-1, 1]

        return p, v

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_blocks: int = 8, num_channels: int = 256) -> DurakResNet:
    return DurakResNet(num_blocks=num_blocks, num_channels=num_channels)


if __name__ == "__main__":
    model = build_model(num_blocks=8, num_channels=256)
    print(f"Параметров: {model.count_parameters():,}")
    print(f"VRAM (FP32): {model.count_parameters() * 4 / 1024**2:.1f} MB")

    # Smoke test с float32 input
    B = 4
    x = torch.rand(B, 220, dtype=torch.float32)
    mask = torch.ones(B, 38)
    p, v = model(x, mask)
    print(f"Policy: {p.shape} (ожидалось [{B}, 38])")
    print(f"Value:  {v.shape} (ожидалось [{B}, 1])")

    # Проверим что нет NaN
    assert not torch.isnan(p).any(), "NaN в policy!"
    assert not torch.isnan(v).any(), "NaN в value!"
    print("OK — нет NaN, размерности верны")
