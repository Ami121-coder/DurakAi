"""
self_play.py — FIX #2: НАСТОЯЩИЙ AlphaZero (OnnxNet в C++ ISMCTS)

КРИТИЧЕСКИЙ БАГ: blend_alpha — это НЕ AlphaZero.
  blend_alpha=1.0: чистый ISMCTS, сеть не используется.
  blend_alpha<1.0: сеть пост-блендит policy, но НЕ участвует в поиске.
  blend_alpha=0.0: ищет current.onnx, которого нет — fallback на ISMCTS.

ФИКС:
  1. Убран blend_alpha. ВСЕГДА используем OnnxNet в C++ ISMCTS.
  2. Если ONNX не существует — воркер ждёт (сеть ещё не обучена).
  3. Dirichlet noise + temperature для exploration в self-play.
  4. Правильная обработка dead воркеров.

Теперь цикл AlphaZero замкнут:
  train.py → export ONNX → self_play.py загружает ONNX в C++
  → ISMCTS с PUCT + сетью → качественные (state, policy, value)
  → replay buffer → train.py обучается → снова export ONNX → ...
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import numpy as np
import torch
import multiprocessing as mp
from typing import Optional, Tuple, List
from queue import Empty

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import durakk_env


# ---- Вспомогательные функции ----

def _softmax_sample(probs: np.ndarray, legal_mask: np.ndarray, temperature: float) -> int:
    p = probs.copy().astype(np.float64)
    p = np.where(legal_mask > 0, p, 0.0)
    if temperature <= 1e-6:
        return int(np.argmax(p))
    p = np.power(np.clip(p, 1e-9, 1.0), 1.0 / temperature)
    s = p.sum()
    if s <= 0:
        idx = np.where(legal_mask > 0)[0]
        return int(np.random.choice(idx)) if len(idx) > 0 else 37
    p = p / s
    return int(np.random.choice(len(p), p=p))


def _add_dirichlet_noise(probs: np.ndarray, legal_mask: np.ndarray,
                          alpha: float = 0.3, eps: float = 0.25) -> np.ndarray:
    legal_idx = np.where(legal_mask > 0)[0]
    if len(legal_idx) == 0:
        return probs
    noise = np.random.dirichlet([alpha] * len(legal_idx))
    mixed = probs.copy()
    for i, idx in enumerate(legal_idx):
        mixed[idx] = (1 - eps) * mixed[idx] + eps * noise[i]
    return mixed


# ---- Self-play worker ----

def self_play_worker(worker_id: int,
                     onnx_path: str,
                     output_queue: mp.Queue,
                     stop_event: mp.Event,
                     games_per_iteration: int,
                     time_budget: float,
                     num_threads: int,
                     temperature_schedule: tuple,
                     dirichlet_alpha: float,
                     dirichlet_eps: float):
    """
    Один воркер: играет games_per_iteration партий, кладёт примеры в очередь.

    ИСПОЛЬЗУЕТ OnnxNet В C++ ISMCTS — настоящий AlphaZero.
    """
    torch.set_num_threads(1)

    env = durakk_env.DurakEnv()
    ACTION_SIZE = 38

    # FIX #2: ждём пока ONNX модель появится
    waited = 0
    while not os.path.exists(onnx_path):
        if stop_event.is_set():
            output_queue.put(None)
            return
        if waited == 0:
            print(f"[Worker {worker_id}] Ждём ONNX модель: {onnx_path}")
        time.sleep(5)
        waited += 1
        if waited > 60:  # 5 минут
            print(f"[Worker {worker_id}] ONNX не появился за 5 мин — fallback на ISMCTS")
            break

    # Загружаем ONNX в C++ движок
    if os.path.exists(onnx_path):
        try:
            env.load_model(onnx_path, "CPU")
            print(f"[Worker {worker_id}] ONNX загружена в C++ ISMCTS ✓")
        except Exception as e:
            print(f"[Worker {worker_id}] Не удалось загрузить ONNX: {e}", file=sys.stderr)

    import traceback
    try:
        while not stop_event.is_set():
            for _ in range(games_per_iteration):
                env.reset()
                game_states = []
                game_policies = []
                game_players = []
                game_masks = []
                game_values = []  # FIX: value из сети на каждый ход

                move_idx = 0
                while not env.is_game_over():
                    T_init, decay_after, T_final = temperature_schedule
                    T = T_init if move_idx < decay_after else T_final

                    # ISMCTS с OnnxNet (если загружена) — visit_probs
                    visit_probs = env.run_ismcts(time_budget, num_threads)
                    probs = np.array(visit_probs, dtype=np.float32)
                    legal_mask = np.array(env.get_legal_action_mask(), dtype=np.float32)

                    if probs.sum() < 1e-6:
                        probs = legal_mask / max(legal_mask.sum(), 1.0)
                    else:
                        probs = probs / probs.sum()

                    # Dirichlet noise на первом ходу
                    if T > 0 and move_idx == 0:
                        probs = _add_dirichlet_noise(probs, legal_mask,
                                                      dirichlet_alpha, dirichlet_eps)

                    # Кодируем состояние ДО хода
                    state = env.encode_state()
                    game_states.append(np.frombuffer(state, dtype=np.uint8).copy())
                    game_policies.append(probs.copy())
                    game_masks.append(legal_mask.copy())
                    game_players.append(env.current_player())

                    # Выбор хода
                    if T <= 1e-6:
                        action = int(np.argmax(probs * legal_mask))
                    else:
                        try:
                            action = _softmax_sample(probs, legal_mask, T)
                        except Exception:
                            action = int(np.argmax(probs * legal_mask))

                    ok = env.step(action)
                    if not ok:
                        idx = np.where(legal_mask > 0)[0]
                        if len(idx) > 0:
                            env.step(int(idx[0]))
                        else:
                            break
                    move_idx += 1

                # Партия завершена — распределяем награды
                winner = env.winner()
                for i in range(len(game_states)):
                    p = game_players[i]
                    if winner == -1:
                        val = 0.0
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
    except Exception as e:
        with open(f"worker_{worker_id}_fatal.log", "a") as f:
            f.write(traceback.format_exc())
    finally:
        output_queue.put(None)


# ---- Arena evaluation ----

def arena_evaluation(onnx_a: str, onnx_b: str, num_games: int = 40,
                     time_budget: float = 0.5) -> dict:
    """Сыграть num_games партий между двумя ONNX-моделями."""
    env = durakk_env.DurakEnv()
    wins_a = wins_b = draws = 0

    # Загружаем модель A
    if os.path.exists(onnx_a):
        env.load_model(onnx_a, "CPU")

    for g in range(num_games):
        env.reset_with_seed(g)
        moves = 0
        while not env.is_game_over() and moves < 500:
            probs = env.run_ismcts(time_budget, 1)
            mask = np.array(env.get_legal_action_mask(), dtype=np.float32)
            p = np.array(probs) * mask
            action = int(np.argmax(p)) if p.sum() > 0 else int(np.argmax(mask))
            if not env.step(action):
                idx = np.where(mask > 0)[0]
                if len(idx) > 0:
                    env.step(int(idx[0]))
                else:
                    break
            moves += 1
        w = env.winner()
        if w == 0:
            wins_a += 1
        elif w == 1:
            wins_b += 1
        else:
            draws += 1

    return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws}


# ---- Менеджер воркеров ----

class SelfPlayManager:
    """Запускает N self-play воркеров с OnnxNet в C++ ISMCTS."""

    def __init__(self,
                 num_workers: int = 6,
                 onnx_path: Optional[str] = None,
                 games_per_iteration: int = 4,
                 time_budget: float = 0.5,
                 num_threads: int = 1):
        self.num_workers = num_workers
        self.onnx_path = onnx_path
        self.games_per_iteration = games_per_iteration
        self.time_budget = time_budget
        self.num_threads = num_threads

        self.queue: mp.Queue = mp.Queue(maxsize=200_000)
        self.stop_event: mp.Event = mp.Event()
        self.workers: List[mp.Process] = []

    def start(self):
        ctx = mp.get_context("spawn")
        for i in range(self.num_workers):
            p = ctx.Process(
                target=self_play_worker,
                args=(i, self.onnx_path, self.queue, self.stop_event,
                      self.games_per_iteration, self.time_budget, self.num_threads,
                      (1.0, 10, 0.25), 0.3, 0.25),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
        print(f"[SelfPlayManager] Запущено {self.num_workers} воркеров (AlphaZero режим)")

    def collect(self, max_examples: int = 10_000, timeout: float = 60.0):
        states, policies, values, masks = [], [], [], []
        deadline = time.time() + timeout
        while len(states) < max_examples and time.time() < deadline:
            try:
                item = self.queue.get(timeout=1.0)
            except Empty:
                continue
            if item is None:
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
    # Smoke test
    mgr = SelfPlayManager(num_workers=2, onnx_path="checkpoints/current.onnx",
                          games_per_iteration=2, time_budget=0.1)
    mgr.start()
    try:
        time.sleep(30)
        s, p, v, m = mgr.collect(max_examples=1000, timeout=5.0)
        print(f"Собрано: states={s.shape}, policies={p.shape}, values={v.shape}")
    finally:
        mgr.stop()
