"""
vector_env.py.patched — оставляем как fallback для синхронной генерации,
но теперь с правильным API: ACTION_SIZE=38, temperature, Dirichlet noise.

Использовать ТОЛЬКО если multiprocessing-based self_play.py не подходит
(например, для отладки). В боевом режиме — self_play.py.
"""

import os
import sys
import numpy as np
import concurrent.futures

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import durakk_env


ACTION_SIZE = 38  # 36 карт + Take + Done


class VectorizedSelfPlay:
    """Синхронный self-play через ThreadPoolExecutor (старый интерфейс)."""

    def __init__(self, num_envs: int = 12):
        self.num_envs = num_envs

    def generate_games(self,
                       num_games_per_env: int = 2,
                       time_budget: float = 0.1,
                       temperature_init: float = 1.0,
                       temperature_final: float = 0.25,
                       temperature_decay_after: int = 10,
                       dirichlet_alpha: float = 0.3,
                       dirichlet_eps: float = 0.25):
        """
        Возвращает:
            states:   np.ndarray (N, 220) uint8
            policies: np.ndarray (N, 38)  float32
            values:   np.ndarray (N, 1)   float32
            masks:    np.ndarray (N, 38)  float32
        """
        all_states, all_policies, all_values, all_masks = [], [], [], []

        def worker():
            env = durakk_env.DurakEnv()
            w_s, w_p, w_v, w_m = [], [], [], []

            for _ in range(num_games_per_env):
                env.reset()
                game_states, game_policies, game_players, game_masks = [], [], [], []
                move_idx = 0

                while not env.is_game_over():
                    # Температура по расписанию.
                    T = temperature_init if move_idx < temperature_decay_after else temperature_final

                    # ISMCTS.
                    visit_probs = env.run_ismcts(time_budget, 1)
                    probs = np.array(visit_probs, dtype=np.float32)
                    mask = np.array(env.get_legal_action_mask(), dtype=np.float32)

                    # Normalize.
                    if probs.sum() < 1e-6:
                        probs = mask / max(mask.sum(), 1.0)
                    else:
                        probs = probs / probs.sum()

                    # Dirichlet noise на первом ходу.
                    if T > 0 and move_idx == 0:
                        legal_idx = np.where(mask > 0)[0]
                        if len(legal_idx) > 0:
                            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_idx))
                            for i, idx in enumerate(legal_idx):
                                probs[idx] = (1 - dirichlet_eps) * probs[idx] + dirichlet_eps * noise[i]

                    # Кодируем состояние ДО хода.
                    state = env.encode_state()
                    game_states.append(np.frombuffer(state, dtype=np.uint8).copy())
                    game_policies.append(probs.copy())
                    game_masks.append(mask.copy())
                    game_players.append(env.current_player())

                    # Выбор хода.
                    if T <= 1e-6:
                        action = int(np.argmax(probs * mask))
                    else:
                        p = probs * mask
                        s = p.sum()
                        if s > 0:
                            p = p / s
                            action = int(np.random.choice(ACTION_SIZE, p=p))
                        else:
                            idx = np.where(mask > 0)[0]
                            action = int(idx[0]) if len(idx) > 0 else 37

                    ok = env.step(action)
                    if not ok:
                        idx = np.where(mask > 0)[0]
                        if len(idx) > 0:
                            env.step(int(idx[0]))
                        else:
                            break
                    move_idx += 1

                # Партия завершена.
                winner = env.winner()
                for i in range(len(game_states)):
                    p = game_players[i]
                    if winner == -1:
                        val = 0.0
                    elif winner == p:
                        val = 1.0
                    else:
                        val = -1.0
                    w_s.append(game_states[i])
                    w_p.append(game_policies[i])
                    w_v.append([val])
                    w_m.append(game_masks[i])
            return w_s, w_p, w_v, w_m

        print(f"[VectorizedSelfPlay] {self.num_envs} потоков × {num_games_per_env} партий "
              f"(бюджет {time_budget}s/ход)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            futures = [executor.submit(worker) for _ in range(self.num_envs)]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    s, p, v, m = fut.result()
                    all_states.extend(s)
                    all_policies.extend(p)
                    all_values.extend(v)
                    all_masks.extend(m)
                except Exception as e:
                    print(f"Ошибка в потоке: {e}")

        if not all_states:
            return (np.empty((0, 256), dtype=np.uint8),  # Task 3: 220 → 256
                    np.empty((0, ACTION_SIZE), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32),
                    np.empty((0, ACTION_SIZE), dtype=np.float32))

        return (
            np.stack(all_states),
            np.stack(all_policies),
            np.array(all_values, dtype=np.float32),
            np.stack(all_masks),
        )
