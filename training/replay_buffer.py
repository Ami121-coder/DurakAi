"""
replay_buffer.py — кольцевой буфер примеров для AlphaZero-обучения.

Особенности:
  - Хранит состояния как uint8 (220 байт/позиция) — в 4× меньше, чем float32.
  - Policies — float32 (38 × 4 = 152 байта).
  - Values — float32 (4 байта).
  - Итого: 376 байт на позицию.
  - При capacity = 1M позиций: ~376 MB RAM. На 32GB DDR5 — пренебрежимо.

API:
  buf = ReplayBuffer(capacity=1_000_000)
  buf.add(states_np, policies_np, values_np)
  states, policies, values, masks = buf.sample(batch_size=4096)
  buf.save("/path/buffer.pt") / buf.load(...)
"""

import os
import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000,
                 state_size: int = 220,
                 action_size: int = 38,
                 seed: int = 42):
        self.capacity = capacity
        self.state_size = state_size
        self.action_size = action_size
        self.rng = np.random.default_rng(seed)

        # Pre-allocate numpy arrays (лучше, чем list.append — нет фрагментации).
        self.states = np.zeros((capacity, state_size), dtype=np.uint8)
        self.policies = np.zeros((capacity, action_size), dtype=np.float32)
        self.values = np.zeros((capacity, 1), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, action_size), dtype=np.float32)

        self.size = 0
        self.idx = 0  # кольцевой индекс для записи

    def add(self,
            states: np.ndarray,
            policies: np.ndarray,
            values: np.ndarray,
            legal_masks: np.ndarray = None) -> None:
        """Добавить батч примеров. Если буфер полон — перезаписываем старые."""
        n = states.shape[0]
        if n == 0:
            return

        # Если legal_masks не передан — заполним по policies (>0 → legal).
        if legal_masks is None:
            legal_masks = (policies > 0).astype(np.float32)

        for i in range(n):
            self.states[self.idx] = states[i]
            self.policies[self.idx] = policies[i]
            self.values[self.idx] = values[i]
            self.legal_masks[self.idx] = legal_masks[i]
            self.idx = (self.idx + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Сэмплируем случайный батч. С возвращением (для больших эпох)."""
        if self.size == 0:
            raise RuntimeError("ReplayBuffer пуст — не из чего сэмплировать.")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.policies[idx],
            self.values[idx],
            self.legal_masks[idx],
        )

    def __len__(self) -> int:
        return self.size

    def save(self, path: str) -> None:
        """Сохраняем в .pt файл. На 1M позиций ~376 MB."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "states": torch.from_numpy(self.states[:self.size]),
            "policies": torch.from_numpy(self.policies[:self.size]),
            "values": torch.from_numpy(self.values[:self.size]),
            "legal_masks": torch.from_numpy(self.legal_masks[:self.size]),
            "state_size": self.state_size,
            "action_size": self.action_size,
        }, path)
        print(f"[ReplayBuffer] Сохранено {self.size:,} позиций в {path}")

    def load(self, path: str) -> None:
        """Загружаем из .pt файла."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        states = data["states"].numpy()
        n = states.shape[0]
        if n > self.capacity:
            # Возьмём последние capacity элементов.
            states = states[-self.capacity:]
            n = self.capacity
        self.states[:n] = states
        self.policies[:n] = data["policies"].numpy()[:n]
        self.values[:n] = data["values"].numpy()[:n]
        self.legal_masks[:n] = data["legal_masks"].numpy()[:n]
        self.size = n
        self.idx = n % self.capacity
        print(f"[ReplayBuffer] Загружено {n:,} позиций из {path}")


if __name__ == "__main__":
    # Smoke-test.
    buf = ReplayBuffer(capacity=10_000)
    fake_states = np.random.randint(0, 2, size=(100, 220), dtype=np.uint8)
    fake_policies = np.random.dirichlet(np.ones(38), size=100).astype(np.float32)
    fake_values = np.random.uniform(-1, 1, size=(100, 1)).astype(np.float32)
    buf.add(fake_states, fake_policies, fake_values)

    s, p, v, m = buf.sample(batch_size=16)
    print(f"Sampled: states={s.shape}, policies={p.shape}, values={v.shape}, masks={m.shape}")
    print(f"Buffer size: {len(buf)}")
    print("OK")
