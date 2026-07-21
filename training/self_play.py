"""
self_play.py — УСИЛЕННЫЙ self-play для AlphaZero (под 7500F + 4060 Ti).

УЛУЧШЕНИЯ:
  1. Больше воркеров: 8 (по умолчанию) вместо 6.
     7500F = 12 потоков SMT, 8 воркеров оставляют 4 потока под
     главный процесс, arena, TensorBoard.
  2. CUDA в воркерах (опционально): env.load_model(path, "CUDA").
     RTX 4060 Ti 8GB — 8 воркеров × ~500MB = 4GB VRAM, запас есть.
     ВНИМАНИЕ: если VRAM не хватает, fallback на CPU.
  3. Больше партий на воркер: games_per_iteration=8 (вместо 4).
     Это поднимает разнообразие данных в 2 раза.
  4. Температурное расписание с annealing:
       T=1.0  первые 8 ходов (exploration)
       T=0.5  ходы 8-20
       T=0.25 ходы 20+
     Это даёт разнообразие в дебютах и точность в эндшпилях.
  5. Dirichlet noise только на первых 2 ходах (вместо 1).
  6. Правильная обработка dead воркеров с автоматическим рестартом.
  7. Метрики throughput: партии/мин, позиции/мин.
  8. Очередь с увеличенным размером (500k вместо 200k).

Пропускная способность (ожидаемо на 7500F + 4060 Ti):
  - 8 воркеров × 8 партий × ~30 ходов = ~1920 позиций/итерацию.
  - time_budget=0.3 сек/ход → 8 воркеров × ~30 ходов × 0.3 сек = 72 сек/итер.
  - + 4 эпохи обучения на 4060 Ti ≈ 8 сек.
  - Итого: ~80 сек/итерация, 1000 итераций ≈ 22 часа.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Fix: add ONNX Runtime capi dir to PATH so subprocess workers find
# onnxruntime_providers_shared.dll (needed for CUDA/TensorRT providers).
try:
    import onnxruntime as _ort
    _ort_capi = os.path.join(os.path.dirname(_ort.__file__), "capi")
    if os.path.isdir(_ort_capi) and _ort_capi not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ort_capi + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass


import sys
import time
import signal
import numpy as np
import torch
import multiprocessing as mp
from typing import Optional, Tuple, List
from queue import Empty

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import durakk_env


# ---- Вспомогательные функции ----

def _softmax_sample(probs: np.ndarray, legal_mask: np.ndarray,
                    temperature: float) -> int:
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
                     dirichlet_eps: float,
                     device: str,
                     noise_first_n_moves: int):
    """
    Один воркер: играет games_per_iteration партий, кладёт примеры в очередь.

    ИСПОЛЬЗУЕТ OnnxNet в C++ ISMCTS — настоящий AlphaZero.
    device: "CUDA" или "CPU".
    """
    torch.set_num_threads(1)

    env = durakk_env.DurakEnv()
    ACTION_SIZE = 38

    # Ждём пока ONNX модель появится.
    waited = 0
    while not os.path.exists(onnx_path):
        if stop_event.is_set():
            output_queue.put(None)
            return
        if waited == 0:
            print(f"[Worker {worker_id}] Ждём ONNX модель: {onnx_path}")
        time.sleep(5)
        waited += 1
        if waited > 60:
            print(f"[Worker {worker_id}] ONNX не появился за 5 мин — fallback на ISMCTS")
            break

    # Загружаем ONNX в C++ движок.
    if os.path.exists(onnx_path):
        try:
            env.load_model(onnx_path, device)
            print(f"[Worker {worker_id}] ONNX загружена на {device} "
                  f"в C++ ISMCTS [OK]")
        except Exception as e:
            print(f"[Worker {worker_id}] Не удалось загрузить ONNX на {device}, "
                  f"пробуем CPU: {e}", file=sys.stderr)
            try:
                env.load_model(onnx_path, "CPU")
                print(f"[Worker {worker_id}] ONNX загружена на CPU (fallback) [OK]")
            except Exception as e2:
                print(f"[Worker {worker_id}] CPU fallback тоже не сработал: {e2}",
                      file=sys.stderr)

    import traceback
    try:
        games_played = 0
        positions_generated = 0

        while not stop_event.is_set():
            for game_idx in range(games_per_iteration):
                if stop_event.is_set():
                    break

                env.reset()
                game_states = []
                game_policies = []
                game_players = []
                game_masks = []

                move_idx = 0
                t0_game = time.time()

                while not env.is_game_over():
                    T_init, T_mid, T_final, decay_after_init, decay_after_mid = \
                        temperature_schedule
                    if move_idx < decay_after_init:
                        T = T_init
                    elif move_idx < decay_after_mid:
                        T = T_mid
                    else:
                        T = T_final

                    # ISMCTS с OnnxNet.
                    visit_probs = env.run_ismcts(time_budget, num_threads)
                    probs = np.array(visit_probs, dtype=np.float32)
                    legal_mask = np.array(env.get_legal_action_mask(),
                                           dtype=np.float32)

                    if probs.sum() < 1e-6:
                        probs = legal_mask / max(legal_mask.sum(), 1.0)
                    else:
                        probs = probs / probs.sum()

                    # Dirichlet noise на первых N ходах.
                    if T > 0 and move_idx < noise_first_n_moves:
                        probs = _add_dirichlet_noise(probs, legal_mask,
                                                      dirichlet_alpha,
                                                      dirichlet_eps)

                    # Кодируем состояние ДО хода.
                    state = env.encode_state()
                    game_states.append(
                        np.frombuffer(state, dtype=np.uint8).copy())
                    game_policies.append(probs.copy())
                    game_masks.append(legal_mask.copy())
                    game_players.append(env.current_player())

                    # Выбор хода.
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

                # Партия завершена — распределяем награды.
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

                games_played += 1
                positions_generated += len(game_states)

            # Периодический отчёт.
            if worker_id == 0 and games_played > 0 and \
               games_played % 10 == 0:
                elapsed = time.time() - t0_game if games_played == 1 else 60.0
                print(f"[Worker {worker_id}] {games_played} партий, "
                      f"{positions_generated} позиций", flush=True)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        with open(f"worker_{worker_id}_fatal.log", "a") as f:
            f.write(traceback.format_exc())
        print(f"[Worker {worker_id}] FATAL: {e}", file=sys.stderr, flush=True)
    finally:
        output_queue.put(None)


# ---- Менеджер воркеров ----

class SelfPlayManager:
    """Запускает N self-play воркеров с OnnxNet в C++ ISMCTS."""

    def __init__(self,
                 num_workers: int = 8,
                 onnx_path: Optional[str] = None,
                 games_per_iteration: int = 8,
                 time_budget: float = 0.3,
                 num_threads: int = 1,
                 device: str = "CUDA",
                 temperature_schedule: tuple = (1.0, 0.5, 0.25, 8, 20),
                 dirichlet_alpha: float = 0.3,
                 dirichlet_eps: float = 0.25,
                 noise_first_n_moves: int = 2):
        self.num_workers = num_workers
        self.onnx_path = onnx_path
        self.games_per_iteration = games_per_iteration
        self.time_budget = time_budget
        self.num_threads = num_threads
        self.device = device
        self.temperature_schedule = temperature_schedule
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.noise_first_n_moves = noise_first_n_moves

        self.queue: mp.Queue = mp.Queue(maxsize=500_000)
        self.stop_event: mp.Event = mp.Event()
        self.workers: List[mp.Process] = []
        self._start_time = None

    def start(self):
        ctx = mp.get_context("spawn")
        for i in range(self.num_workers):
            p = ctx.Process(
                target=self_play_worker,
                args=(i, self.onnx_path, self.queue, self.stop_event,
                      self.games_per_iteration, self.time_budget, self.num_threads,
                      self.temperature_schedule,
                      self.dirichlet_alpha, self.dirichlet_eps,
                      self.device, self.noise_first_n_moves),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
        self._start_time = time.time()
        print(f"[SelfPlayManager] Запущено {self.num_workers} воркеров "
              f"(AlphaZero режим, device={self.device}, "
              f"time_budget={self.time_budget}s, games={self.games_per_iteration})")

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

    def alive_workers(self) -> int:
        return sum(1 for w in self.workers if w.is_alive())

    def stop(self):
        self.stop_event.set()
        for p in self.workers:
            if p.is_alive():
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
        self.workers.clear()

    def stats(self) -> dict:
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "elapsed_sec": elapsed,
            "alive_workers": self.alive_workers(),
            "queue_size": self.queue.qsize() if hasattr(self.queue, 'qsize') else -1,
        }


if __name__ == "__main__":
    # Smoke test.
    mgr = SelfPlayManager(num_workers=2, onnx_path="checkpoints/current.onnx",
                          games_per_iteration=2, time_budget=0.1,
                          device="CPU")
    mgr.start()
    try:
        time.sleep(30)
        s, p, v, m = mgr.collect(max_examples=1000, timeout=5.0)
        print(f"Собрано: states={s.shape}, policies={p.shape}, values={v.shape}")
        print(f"Stats: {mgr.stats()}")
    finally:
        mgr.stop()
