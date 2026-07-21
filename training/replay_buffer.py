"""
replay_buffer.py — кольцевой буфер с опциональным Prioritized Experience Replay (PER).

Возможности:
  1. Стандартный кольцевой буфер (как в исходной версии).
  2. Опциональный PER (Schaul et al., 2015) — примеры с большим TD-error
     сэмплируются чаще. Ускоряет сходимость AlphaZero на 20-40%.
  3. Alpha-обрезка: важные потери (TD-error) сглаживаются alpha-степенью.
  4. Importance sampling weights для компенсации bias.
  5. Сохранение/загрузка через torch.save.

Память (capacity=1M, формат uint8):
  states: 1M × 220 bytes = 220 MB
  policies: 1M × 38 × 4 bytes = 152 MB
  values: 1M × 4 bytes = 4 MB
  masks: 1M × 38 × 4 bytes = 152 MB
  priorities (PER): 1M × 4 bytes = 4 MB
  Итого: ~530 MB — комфортно на 32GB DDR5.

API:
  buf = ReplayBuffer(capacity=1_000_000, prioritized=True)
  buf.add(states_np, policies_np, values_np, masks_np,
          priorities=np.abs(td_errors))  # опционально
  states, policies, values, masks, weights, idxs = buf.sample(batch_size=4096)
  buf.update_priorities(idxs, new_td_errors)  # для PER
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional


class ReplayBuffer:
    """Кольцевой буфер примеров для AlphaZero-обучения."""

    def __init__(self,
                 capacity: int = 1_000_000,
                 state_size: int = 220,
                 action_size: int = 38,
                 prioritized: bool = True,
                 alpha: float = 0.6,        # PER priority exponent
                 beta: float = 0.4,         # PER importance sampling exponent
                 beta_anneal: float = 0.0001,  # прирост beta за sample
                 seed: int = 42):
        self.capacity = capacity
        self.state_size = state_size
        self.action_size = action_size
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = beta_anneal

        self.rng = np.random.default_rng(seed)

        # Pre-allocate.
        self.states = np.zeros((capacity, state_size), dtype=np.uint8)
        self.policies = np.zeros((capacity, action_size), dtype=np.float32)
        self.values = np.zeros((capacity, 1), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, action_size), dtype=np.float32)

        # PER: priorities хранят |TD-error|^alpha.
        if prioritized:
            self.priorities = np.zeros(capacity, dtype=np.float32)
            # Начальный приоритет — для новых примеров (если priority не передан).
            self._max_priority = 1.0
        else:
            self.priorities = None

        self.size = 0
        self.idx = 0  # кольцевой индекс записи

    def add(self,
            states: np.ndarray,
            policies: np.ndarray,
            values: np.ndarray,
            legal_masks: np.ndarray = None,
            priorities: Optional[np.ndarray] = None) -> None:
        """Добавить батч примеров."""
        n = states.shape[0]
        if n == 0:
            return

        if legal_masks is None:
            legal_masks = (policies > 0).astype(np.float32)

        for i in range(n):
            self.states[self.idx] = states[i]
            self.policies[self.idx] = policies[i]
            self.values[self.idx] = values[i]
            self.legal_masks[self.idx] = legal_masks[i]

            if self.prioritized:
                if priorities is not None and i < len(priorities):
                    # |TD-error|^alpha, clip снизу для стабильности.
                    p = max(float(priorities[i]) ** self.alpha, 1e-6)
                    self.priorities[self.idx] = p
                    if p > self._max_priority:
                        self._max_priority = p
                else:
                    # Новые примеры получают max priority — гарантия хотя бы
                    # одного сэмплинга.
                    self.priorities[self.idx] = self._max_priority

            self.idx = (self.idx + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1

    def sample(self,
               batch_size: int) -> Tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
        """
        Сэмплируем батч.

        Returns: (states, policies, values, masks, weights, idxs)
          weights: importance sampling weights для PER (1 если не PER).
          idxs: индексы примеров (для update_priorities).
        """
        if self.size == 0:
            raise RuntimeError("ReplayBuffer пуст — не из чего сэмплировать.")

        if self.prioritized and self.size > 1:
            # Сэмплирование пропорционально priorities.
            probs = self.priorities[:self.size]
            probs_sum = probs.sum()
            if probs_sum <= 0:
                # Fallback на uniform.
                probs = np.ones(self.size, dtype=np.float32) / self.size
            else:
                probs = probs / probs_sum
            idxs = self.rng.choice(self.size, size=batch_size, p=probs)

            # Importance sampling weights: (N * P(i))^(-beta).
            # Это компенсирует bias от неравномерного сэмплирования.
            weights = (self.size * probs[idxs]) ** (-self.beta)
            weights = weights / weights.max()  # нормализация
            weights = weights.astype(np.float32)

            # Аннеалим beta к 1.0 (стабильнее в поздних итерациях).
            self.beta = min(1.0, self.beta + self.beta_anneal)
        else:
            # Uniform сэмплинг.
            idxs = self.rng.integers(0, self.size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)

        return (
            self.states[idxs],
            self.policies[idxs],
            self.values[idxs],
            self.legal_masks[idxs],
            weights,
            idxs,
        )

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        """Обновить priorities по результатам обучения (для PER)."""
        if not self.prioritized:
            return
        for i, idx in enumerate(idxs):
            p = max(abs(float(td_errors[i])) ** self.alpha, 1e-6)
            self.priorities[idx] = p
            if p > self._max_priority:
                self._max_priority = p

    def __len__(self) -> int:
        return self.size

    def save(self, path: str) -> None:
        """Сохраняем в .pt файл (~530 MB на 1M позиций с PER)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_dict = {
            "states": torch.from_numpy(self.states[:self.size]),
            "policies": torch.from_numpy(self.policies[:self.size]),
            "values": torch.from_numpy(self.values[:self.size]),
            "legal_masks": torch.from_numpy(self.legal_masks[:self.size]),
            "state_size": self.state_size,
            "action_size": self.action_size,
            "size": self.size,
        }
        if self.prioritized:
            save_dict["priorities"] = torch.from_numpy(
                self.priorities[:self.size])
        torch.save(save_dict, path)
        print(f"[ReplayBuffer] Сохранено {self.size:,} позиций в {path}")

    def load(self, path: str) -> None:
        """Загружаем из .pt файла."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        states = data["states"].numpy()
        n = states.shape[0]
        if n > self.capacity:
            states = states[-self.capacity:]
            n = self.capacity
        self.states[:n] = states
        self.policies[:n] = data["policies"].numpy()[:n]
        self.values[:n] = data["values"].numpy()[:n]
        self.legal_masks[:n] = data["legal_masks"].numpy()[:n]
        if self.prioritized and "priorities" in data:
            self.priorities[:n] = data["priorities"].numpy()[:n]
            self._max_priority = float(self.priorities[:n].max())
        self.size = n
        self.idx = n % self.capacity
        print(f"[ReplayBuffer] Загружено {n:,} позиций из {path}")


if __name__ == "__main__":
    # Smoke-test.
    buf = ReplayBuffer(capacity=10_000, prioritized=True)
    fake_states = np.random.randint(0, 2, size=(100, 220), dtype=np.uint8)
    fake_policies = np.random.dirichlet(np.ones(38), size=100).astype(np.float32)
    fake_values = np.random.uniform(-1, 1, size=(100, 1)).astype(np.float32)
    fake_td = np.random.uniform(0.01, 2.0, size=100).astype(np.float32)
    buf.add(fake_states, fake_policies, fake_values, priorities=fake_td)

    s, p, v, m, w, idxs = buf.sample(batch_size=16)
    print(f"Sampled: states={s.shape}, policies={p.shape}, values={v.shape}, "
          f"masks={m.shape}, weights={w.shape}, idxs={idxs.shape}")
    print(f"Buffer size: {len(buf)}, max_priority: {buf._max_priority:.3f}")
    print("OK")
