"""
model_transformer.py — альтернативная архитектура: Set-Transformer над 36 картами.

Концепция:
  Durak = множество карт с ролями (моя рука / стол / бито / неизвестно).
  Каждая из 36 карт — это «токен» с rich feature-вектором:
    [in_my_hand, in_opp_known_taken, in_table_attack, in_table_defense,
     in_discard, is_trump_suit, rank_one_hot(9), suit_one_hot(4),
     scalar_features...]  — 25-30 признаков на карту.
  К этому добавляем 1 «глобальный» токен для game-state признаков
  (фаза, ход, deckRemaining и т.д.).

  Дальше — Transformer encoder (4-6 слоёв, 4-8 голов, 256 dim).
  Policy head: линейный слой с 36+2 выходами (по карте + Take + Done).
  Value head: пул + 2-layer MLP.

VRAM (FP32, batch 1024):
  - Параметры: ~3.5M (4 слоя, 256 dim, 8 head)
  - Activations: ~250MB (внимание O(N²) на 37 токенов = 1369, ок)
  - Итого: ~280MB. На 4060 Ti 8GB — ок.

Throughput (4060 Ti, FP32): ~3-5k samples/sec (медленнее ResNet).
Throughput (4060 Ti, FP16): ~8-12k samples/sec.

Когда использовать:
  - Если хочется поэкспериментировать с attention-картами.
  - Если важно интерпретировать какие карты «внимание привлекают».
Когда НЕ использовать:
  - На этапе становления pipeline. ResNet быстрее обучается и обычно даёт
    лучшую силу на том же бюджете обучения.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CardTokenizer(nn.Module):
    """Превращает [B, 220] в [B, 37, D] — 36 карт + 1 глобальный токен."""

    def __init__(self, d_model: int = 256, input_size: int = 220):
        super().__init__()
        self.d_model = d_model
        # 220 = 5 масок × 36 карт + 40 скаляров
        #   masks: [my_hand, trump_suit, atk, def, discard] × 36 = 180
        #   scalars: 40
        # На карту: 5 фичей (по одной маске на карту) + 4 ранговых one-hot (по 9 — нет, это 9)
        # Упростим: на карту приходит 5 масок + 4 флага масти + 9 one-hot ранга = 18 признаков.
        # + 1 «глобальный» токен со всеми 40 скалярами.
        self.card_proj = nn.Linear(18, d_model)
        self.global_proj = nn.Linear(40, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 220]
        Returns: [B, 37, D]
        """
        B = x.size(0)
        # Маски: первые 5 × 36 = 180 значений — это 5 масок по 36.
        masks = x[:, :180].view(B, 5, 36)              # [B, 5, 36]
        scalars = x[:, 180:220]                          # [B, 40]

        # На карту i: 5 масок + 9-ранг one-hot + 4-масть one-hot.
        # Индекс карты 0..35: suit = idx // 9, rank = idx % 9.
        device = x.device
        idx = torch.arange(36, device=device)
        suit = idx // 9  # 0..3
        rank = idx % 9   # 0..8
        suit_oh = F.one_hot(suit, 4).float()   # [36, 4]
        rank_oh = F.one_hot(rank, 9).float()   # [36, 9]

        # masks[:, :, i] → [B, 5] для карты i.
        # Склеим: [B, 36, 5] + [36, 4] + [36, 9] → [B, 36, 18].
        masks_per_card = masks.permute(0, 2, 1)  # [B, 36, 5]
        suit_oh_b = suit_oh.unsqueeze(0).expand(B, -1, -1)  # [B, 36, 4]
        rank_oh_b = rank_oh.unsqueeze(0).expand(B, -1, -1)  # [B, 36, 9]
        card_features = torch.cat([masks_per_card, suit_oh_b, rank_oh_b], dim=-1)  # [B, 36, 18]

        card_tokens = self.card_proj(card_features)  # [B, 36, D]
        global_token = self.global_proj(scalars).unsqueeze(1)  # [B, 1, D]

        return torch.cat([card_tokens, global_token], dim=1)  # [B, 37, D]


class TransformerBlock(nn.Module):
    """Стандартный Transformer encoder layer (Pre-LN вариант)."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: x = x + attn(norm(x));  x = x + ff(norm(x))
        h = self.norm1(x)
        a, _ = self.attn(h, h, h)
        x = x + self.dropout(a)
        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x


class DurakTransformer(nn.Module):
    """
    Set-Transformer для Durak.

    Вход: [B, 220].
    Выход: (policy[B, 38], value[B, 1]).
    """

    def __init__(self,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 action_size: int = 38):
        super().__init__()
        self.tokenizer = CardTokenizer(d_model=d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Policy head: каждый карточный токен → 1 логит (карта) + 2 глобальных (Take, Done).
        self.policy_card = nn.Linear(d_model, 1)
        self.policy_extra = nn.Linear(d_model, 2)

        # Value head: глобальный токен → MLP.
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor = None):
        """
        x: [B, 220]
        legal_mask: [B, 38] — для маскирования policy.
        Returns: (policy[B, 38], value[B, 1])
        """
        tokens = self.tokenizer(x)  # [B, 37, D]
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # Policy: 36 карт (карточные токены) + Take + Done (глобальный токен).
        card_logits = self.policy_card(tokens[:, :36, :]).squeeze(-1)  # [B, 36]
        extra_logits = self.policy_extra(tokens[:, 36:37, :]).squeeze(1)  # [B, 2]
        policy = torch.cat([card_logits, extra_logits], dim=-1)  # [B, 38]

        if legal_mask is not None:
            policy = policy.masked_fill(legal_mask < 0.5, float('-inf'))

        # Value: из глобального токена.
        value = torch.tanh(self.value_head(tokens[:, 36, :]))  # [B, 1]
        return policy, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_transformer(d_model: int = 256, n_heads: int = 8, n_layers: int = 4) -> DurakTransformer:
    return DurakTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers)


if __name__ == "__main__":
    model = build_transformer()
    print(f"Параметров: {model.count_parameters():,}")
    print(f"VRAM (FP32): {model.count_parameters() * 4 / 1024 / 1024:.1f} MB")

    B = 4
    x = torch.randn(B, 220)
    mask = torch.ones(B, 38)
    p, v = model(x, mask)
    print(f"Policy: {p.shape} (ожидалось [{B}, 38])")
    print(f"Value:  {v.shape} (ожидалось [{B}, 1])")
    print("OK")
