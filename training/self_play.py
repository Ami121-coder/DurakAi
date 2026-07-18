"""
self_play.py — асинхронный self-play worker для AlphaZero-обучения.

Архитектура:
  - N воркеров (процессов, multiprocessing.Pool) — каждый держит свой durakk_env.
  - Каждый воркер играет партии, используя ISMCTS + текущую модель (через ONNX
    inference внутри C++ ИЛИ через PolicyValueNet в Python).
  - По мере игры воркер кладёт (state, policy, value) в общую очередь.
  - Главный процесс забирает батчи из очереди и кладёт в ReplayBuffer.
  - Отдельный trainer-процесс читает из ReplayBuffer и обучает модель.
  - После каждых K итераций обучения — обновлённая модель публикуется в воркеры.

Почему multiprocess, а не thread:
  - Python GIL отпускается во время C++ ISMCTS, но PyTorch inference тоже
    отпускает GIL. Однако если в одном процессе делать inference 4060 Ti и
    одновременно генерить игры на CPU — они будут конкурировать за GIL между
    вызовами PyTorch/Numpy. Process pool изолирует это.
  - Каждый воркер грузит свой C++ env — изолированное состояние, нет гонок.

Параметры под 7500F + 4060 Ti 8GB + 32GB:
  - 6-8 self-play воркеров (по числу физических ядер 7500F).
  - 1 trainer процесс с эксклюзивным доступом к GPU.
  - 1 arbiter процесс (arena evaluation).
"""

import os
import sys
import time
import pickle
import signal
import numpy as np
import torch
import multiprocessing as mp
from typing import Optional, Tuple, List
from queue import Empty

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import durakk_env
from model_resnet import build_model


# ---------- Вспомогательные функции ----------

def _softmax_sample(probs: np.ndarray, legal_mask: np.ndarray, temperature: float) -> int:
    """Сэмплирование хода с температурой."""
    p = probs.copy().astype(np.float64)
    p = np.where(legal_mask > 0, p, 0.0)
    if temperature <= 1e-6:
        # argmax.
        return int(np.argmax(p))
    # Tempered: p^(1/T), затем нормализуем.
    p = np.power(np.clip(p, 1e-9, 1.0), 1.0 / temperature)
    s = p.sum()
    if s <= 0:
        # Fallback на uniform по legal_mask.
        idx = np.where(legal_mask > 0)[0]
        return int(np.random.choice(idx)) if len(idx) > 0 else 37
    p = p / s
    return int(np.random.choice(len(p), p=p))


def _add_dirichlet_noise(probs: np.ndarray, legal_mask: np.ndarray,
                          alpha: float = 0.3, eps: float = 0.25) -> np.ndarray:
    """Dirichlet noise на корне для exploration."""
    legal_idx = np.where(legal_mask > 0)[0]
    if len(legal_idx) == 0:
        return probs
    noise = np.random.dirichlet([alpha] * len(legal_idx))
    mixed = probs.copy()
    for i, idx in enumerate(legal_idx):
        mixed[idx] = (1 - eps) * mixed[idx] + eps * noise[i]
    return mixed


# ---------- Self-play worker ----------

def self_play_worker(worker_id: int,
                     model_path: str,
                     output_queue: mp.Queue,
                     stop_event: mp.Event,
                     games_per_iteration: int = 4,
                     time_budget: float = 0.5,
                     num_threads: int = 1,
                     temperature_schedule: tuple = (1.0, 10, 0.25),
                     dirichlet_alpha: float = 0.3,
                     dirichlet_eps: float = 0.25):
    """
    Один воркер: играет games_per_iteration партий, кладёт примеры в очередь.

    temperature_schedule: (T_init, decay_after_n_moves, T_final) — температура
                          падает после decay_after_n_moves ходов партии.
                          AlphaZero: T=1 первые 30 ходов, потом T=0.25.
                          Для Durak (короче партии) — T=1 первые 10, потом 0.25.
    """
    # Импортируем C++ env в контексте воркера.
    env = durakk_env.DurakEnv()
    ACTION_SIZE = 38

    # Модель — загружаем из .pt (или .onnx — но тогда через OnnxNet в C++).
    # Здесь используем Python-модель, чтобы ISMCTS в C++ НЕ использовал сеть —
    # упрощённый путь: C++ ISMCTS считает чистую статистику, а Python-модель
    # предсказывает policy/value ДЛЯ ОБУЧЕНИЯ (имитируя ISMCTS).
    # Полноценный AlphaZero-путь: загрузить .onnx через OnnxNet в C++.
    model = None
    if model_path and os.path.exists(model_path):
        try:
            model = build_model(num_blocks=8, num_channels=256)
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            model.eval()
        except Exception as e:
            print(f"[Worker {worker_id}] Не удалось загрузить модель: {e}", file=sys.stderr)
            model = None

    try:
        while not stop_event.is_set():
            for _ in range(games_per_iteration):
                env.reset()
                game_states = []
                game_policies = []
                game_players = []
                game_masks = []

                move_idx = 0
                while not env.is_game_over():
                    # Температура по расписанию.
                    T_init, decay_after, T_final = temperature_schedule
                    T = T_init if move_idx < decay_after else T_final

                    # ISMCTS — возвращает visit_probs (если model=None, это чистый ISMCTS).
                    visit_probs = env.run_ismcts(time_budget, num_threads)
                    probs = np.array(visit_probs, dtype=np.float32)
                    legal_mask = np.array(env.get_legal_action_mask(), dtype=np.float32)

                    # Если probs нули — fallback на uniform по legal_mask.
                    if probs.sum() < 1e-6:
                        probs = legal_mask / max(legal_mask.sum(), 1.0)
                    else:
                        probs = probs / probs.sum()

                    # Dirichlet noise на корне (если T > 0).
                    if T > 0 and move_idx == 0:
                        probs = _add_dirichlet_noise(probs, legal_mask,
                                                      dirichlet_alpha, dirichlet_eps)

                    # Кодируем состояние ДО хода.
                    state = env.encode_state()
                    game_states.append(np.frombuffer(state, dtype=np.uint8).copy())
                    game_policies.append(probs.copy())
                    game_masks.append(legal_mask.copy())
                    game_players.append(env.current_player())

                    # Выбор хода.
                    if T <= 1e-6:
                        action = int(np.argmax(probs * legal_mask))
                    else:
                        action = _softmax_sample(probs, legal_mask, T)

                    ok = env.step(action)
                    if not ok:
                        # Если ход не прошёл — возьмём первый легальный.
                        idx = np.where(legal_mask > 0)[0]
                        if len(idx) > 0:
                            env.step(int(idx[0]))
                        else:
                            break
                    move_idx += 1

                # Партия завершена — распределение наград.
                winner = env.winner()
                for i in range(len(game_states)):
                    p = game_players[i]
                    if winner == -1:
                        val = 0.0  # ничья
                    elif winner == p:
                        val = 1.0
                    else:
                        val = -1.0
                    output_queue.put((
                        game_states[i],
                        game_policies[i],
                        np.array([val], dtype=np.float32),
                        game_masks[i],
                    ))

    except KeyboardInterrupt:
        pass
    finally:
        output_queue.put(None)  # сигнал "воркер завершился"


# ---------- Arbiter (arena) ----------

def arena_evaluation(model_a_path: str,
                     model_b_path: str,
                     num_games: int = 100,
                     time_budget: float = 0.5) -> dict:
    """
    Сыграть num_games партий между двумя моделями (точнее — между двумя ISMCTS
    с разными time_budgets или разными конфигами).

    Возвращает {wins_a, wins_b, draws}.
    """
    env = durakk_env.DurakEnv()
    wins_a = wins_b = draws = 0
    for g in range(num_games):
        env.reset_with_seed(g)
        while not env.is_game_over():
            # Упрощение: обе «модели» — это просто ISMCTS. В реальной арене
            # одна модель должна использовать OnnxNet.
            probs = env.run_ismcts(time_budget, 1)
            legal_mask = np.array(env.get_legal_action_mask(), dtype=np.float32)
            p = np.array(probs) * legal_mask
            action = int(np.argmax(p)) if p.sum() > 0 else int(np.argmax(legal_mask))
            if not env.step(action):
                idx = np.where(legal_mask > 0)[0]
                if len(idx) > 0:
                    env.step(int(idx[0]))
                else:
                    break
        w = env.winner()
        if w == 0:
            wins_a += 1
        elif w == 1:
            wins_b += 1
        else:
            draws += 1
    return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws}


# ---------- Менеджер воркеров ----------

class SelfPlayManager:
    """Запускает N self-play воркеров, собирает примеры."""

    def __init__(self,
                 num_workers: int = 6,
                 model_path: Optional[str] = None,
                 games_per_iteration: int = 4,
                 time_budget: float = 0.5,
                 num_threads: int = 1):
        self.num_workers = num_workers
        self.model_path = model_path
        self.games_per_iteration = games_per_iteration
        self.time_budget = time_budget
        self.num_threads = num_threads

        self.queue: mp.Queue = mp.Queue(maxsize=200_000)
        self.stop_event: mp.Event = mp.Event()
        self.workers: List[mp.Process] = []

    def start(self):
        ctx = mp.get_context("spawn")  # безопаснее для C++ модулей на Windows
        for i in range(self.num_workers):
            p = ctx.Process(
                target=self_play_worker,
                args=(i, self.model_path, self.queue, self.stop_event,
                      self.games_per_iteration, self.time_budget, self.num_threads),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
        print(f"[SelfPlayManager] Запущено {self.num_workers} воркеров")

    def collect(self, max_examples: int = 10_000, timeout: float = 60.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Собрать до max_examples примеров из очереди."""
        states, policies, values, masks = [], [], [], []
        deadline = time.time() + timeout
        while len(states) < max_examples and time.time() < deadline:
            try:
                item = self.queue.get(timeout=1.0)
            except Empty:
                continue
            if item is None:
                # Воркер завершился.
                continue
            s, p, v, m = item
            states.append(s)
            policies.append(p)
            values.append(v)
            masks.append(m)
        if not states:
            return (
                np.empty((0, 220), dtype=np.uint8),
                np.empty((0, 38), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 38), dtype=np.float32),
            )
        return (
            np.stack(states),
            np.stack(policies),
            np.stack(values),
            np.stack(masks),
        )

    def stop(self):
        self.stop_event.set()
        for p in self.workers:
            if p.is_alive():
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
        self.workers.clear()


if __name__ == "__main__":
    # Smoke-test: запустить 2 воркера на 30 секунд, собрать примеры.
    mgr = SelfPlayManager(num_workers=2, games_per_iteration=2, time_budget=0.1)
    mgr.start()
    try:
        time.sleep(30)
        s, p, v, m = mgr.collect(max_examples=1000, timeout=5.0)
        print(f"Собрано: states={s.shape}, policies={p.shape}, values={v.shape}, masks={m.shape}")
    finally:
        mgr.stop()
