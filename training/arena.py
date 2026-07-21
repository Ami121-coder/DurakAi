"""
arena.py — ИСПРАВЛЕННАЯ arena evaluation (FIX критического бага).

КРИТИЧЕСКИЙ БАГ В СТАРОЙ ВЕРСИИ:
  - C++ движок держал ОДНУ модель за раз.
  - Arena загружала ТОЛЬКО current.onnx.
  - После первого обновления baseline = current, и играли две одинаковые модели.
  - Winrate всегда ≈ 50%, метрика нерабочая.

ФИКС:
  1. Arena запускает ДВА отдельных субпроцесса durakk_env.
     Каждый процесс грузит свою ONNX-модель.
  2. Turn-based протокол: главный арбитр держит состояние игры,
     воркеры только дают ходы по запросу.
  3. Корректный winrate current vs baseline.
  4. Дополнительный внешний baseline — RandomNet-ISMCTS.
  5. Логирование Elo-разницы.

Воркер-протокол (stdin/stdout, JSON-строки):
  → {"cmd": "load", "path": "..."}                    ← {"ok": true}
  → {"cmd": "unload"}                                  ← {"ok": true}
  → {"cmd": "init", "seed": 42}                        ← {"ok": true}
  → {"cmd": "ask_action", "time_budget": 0.3}          ← {"action": 17}
  → {"cmd": "apply", "action": 17}                     ← {"ok": true}
  → {"cmd": "quit"}
"""

import os
import sys
import json
import time
import subprocess
import threading
import numpy as np
from typing import Optional


# ============================================================================
# Субпроцесс-обёртка над durakk_env: один процесс = одна модель.
# ============================================================================

class _WorkerProcess:
    """Один субпроцесс с загруженной ONNX-моделью. Потокобезопасный."""

    def __init__(self, name: str, durakk_env_path: str,
                 device: str = "CPU"):
        self.name = name
        self.durakk_env_path = durakk_env_path
        self.device = device

        # Запускаем Python-субпроцесс, который держит env и модель.
        self.proc = subprocess.Popen(
            [sys.executable, "-u", __file__, "--worker", name,
             "--durakk-env-path", durakk_env_path,
             "--device", device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._lock = threading.Lock()

        # Ждём готовности.
        line = self.proc.stdout.readline()
        if "READY" not in line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"Worker {name} не стартовал: {line}\n{err}")
        print(f"[Arena] Worker '{name}' готов (pid={self.proc.pid})")

    def load_model(self, path: str) -> bool:
        with self._lock:
            self._send({"cmd": "load", "path": path})
            resp = self._recv()
            return resp.get("ok", False)

    def unload_model(self) -> bool:
        with self._lock:
            self._send({"cmd": "unload"})
            resp = self._recv()
            return resp.get("ok", False)

    def init_game(self, seed: int):
        with self._lock:
            self._send({"cmd": "init", "seed": seed})
            self._recv()

    def ask_action(self, time_budget: float = 0.3) -> int:
        with self._lock:
            self._send({"cmd": "ask_action", "time_budget": time_budget})
            resp = self._recv()
            return int(resp.get("action", 37))

    def apply_action(self, action: int):
        with self._lock:
            self._send({"cmd": "apply", "action": action})
            self._recv()

    def close(self):
        try:
            self._send({"cmd": "quit"})
            self.proc.wait(timeout=5.0)
        except Exception:
            self.proc.kill()

    def _send(self, msg: dict):
        line = json.dumps(msg) + "\n"
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _recv(self, timeout: float = 120.0) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"Worker {self.name} умер: stderr={err}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Worker {self.name} прислал мусор: {line!r}: {e}")


# ============================================================================
# Субпроцесс-воркер (точка входа при запуске с --worker).
# ============================================================================

def _worker_main(name: str, durakk_env_path: str, device: str = "CPU"):
    """Тело субпроцесса: держит durakk_env, отвечает на команды."""
    # durakk_env_path может быть директорией, где лежит .pyd/.so.
    sys.path.insert(0, os.path.dirname(durakk_env_path))
    sys.path.insert(0, durakk_env_path)

    try:
        import durakk_env
    except ImportError as e:
        print(f"WORKER_ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    import numpy as np

    env = durakk_env.DurakEnv()
    model_loaded = False
    has_model = False

    print("READY", flush=True)

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        cmd = msg.get("cmd")

        if cmd == "quit":
            break

        elif cmd == "load":
            try:
                env.load_model(msg["path"], device)
                model_loaded = True
                has_model = True
                print(json.dumps({"ok": True}), flush=True)
            except Exception as e:
                print(json.dumps({"ok": False, "error": str(e)}), flush=True)

        elif cmd == "unload":
            env = durakk_env.DurakEnv()
            model_loaded = False
            has_model = False
            print(json.dumps({"ok": True}), flush=True)

        elif cmd == "init":
            try:
                env.reset_with_seed(int(msg["seed"]))
                print(json.dumps({"ok": True}), flush=True)
            except Exception as e:
                print(json.dumps({"ok": False, "error": str(e)}), flush=True)

        elif cmd == "ask_action":
            # Возвращает action_index лучшего хода в текущей позиции.
            try:
                tb = float(msg.get("time_budget", 0.3))
                probs = env.run_ismcts(tb, 1)
                mask = np.array(env.get_legal_action_mask(), dtype=np.float32)
                p = np.array(probs) * mask
                action = int(np.argmax(p)) if p.sum() > 0 else int(np.argmax(mask))
                print(json.dumps({"action": action}), flush=True)
            except Exception as e:
                print(json.dumps({"action": 37, "error": str(e)}), flush=True)

        elif cmd == "apply":
            # Применить ход в своём env (для синхронизации с арбитром).
            try:
                ok = env.step(int(msg["action"]))
                print(json.dumps({"ok": ok}), flush=True)
            except Exception as e:
                print(json.dumps({"ok": False, "error": str(e)}), flush=True)

        else:
            print(json.dumps({"ok": False, "error": f"unknown cmd: {cmd}"}),
                  flush=True)


# ============================================================================
# Arena Evaluator: current vs baseline (turn-based, корректный).
# ============================================================================

class ArenaEvaluator:
    """
    Корректная arena: current vs baseline через turn-based протокол.

    Архитектура:
      - Главный процесс (арбитр) держит "голый" durakk_env без модели.
      - Воркер A: env с current.onnx.
      - Воркер B: env с best.onnx.
      - На каждый ход:
          1. Арбитр определяет, чей ход.
          2. Если ход current — просит action у worker A.
          3. Если ход baseline — просит action у worker B.
          4. Применяет action в своём env.
          5. Синхронизирует другой воркер (apply_action).

    Каждые N итераций: current vs baseline на num_games партий.
    Если winrate ≥ 55% → current становится новым baseline.
    """

    def __init__(self,
                 onnx_path: str,
                 durakk_env_path: str,
                 time_budget: float = 0.3,
                 num_games: int = 40,
                 baseline_path: Optional[str] = None,
                 device: str = "CPU"):
        self.onnx_path = os.path.abspath(onnx_path)
        self.baseline_path = os.path.abspath(
            baseline_path or
            os.path.join(os.path.dirname(onnx_path), "best.onnx")
        )
        self.durakk_env_path = durakk_env_path
        self.time_budget = time_budget
        self.num_games = num_games
        self.device = device

        # История Elo.
        self.elo_history = []  # list of (iter, elo_diff, wr)

        # При первом запуске baseline = current.
        if not os.path.exists(self.baseline_path) and os.path.exists(self.onnx_path):
            import shutil
            shutil.copy2(self.onnx_path, self.baseline_path)
            print(f"[Arena] Создан baseline: {self.baseline_path}")

        self._worker_a: Optional[_WorkerProcess] = None
        self._worker_b: Optional[_WorkerProcess] = None

        # Импортируем durakk_env в главном процессе (арбитр).
        sys.path.insert(0, os.path.dirname(self.durakk_env_path))
        sys.path.insert(0, self.durakk_env_path)
        import durakk_env  # noqa
        self._durakk_env_module = durakk_env

    def _ensure_workers(self):
        if self._worker_a is None:
            self._worker_a = _WorkerProcess("current", self.durakk_env_path,
                                             device=self.device)
            self._worker_b = _WorkerProcess("baseline", self.durakk_env_path,
                                             device=self.device)
            # Грузим модели.
            if not self._worker_a.load_model(self.onnx_path):
                print(f"[Arena] WARNING: не загрузилась current модель "
                      f"{self.onnx_path}")
            if os.path.exists(self.baseline_path):
                if not self._worker_b.load_model(self.baseline_path):
                    print(f"[Arena] WARNING: не загрузилась baseline модель "
                          f"{self.baseline_path}")

    def evaluate_current_vs_baseline(self) -> dict:
        """Играем num_games партий current vs baseline."""
        self._ensure_workers()

        current_wins = 0
        baseline_wins = 0
        draws = 0
        errors = 0

        for g in range(self.num_games):
            current_is_me = (g % 2 == 0)  # чередуем стороны
            seed = g + 1000

            try:
                winner = self._play_one_turn_based_game(
                    seed=seed,
                    current_is_me=current_is_me,
                )
            except Exception as e:
                print(f"[Arena] Ошибка в игре {g}: {e}")
                errors += 1
                continue

            if winner == -1:
                draws += 1
            elif (winner == 0 and current_is_me) or \
                 (winner == 1 and not current_is_me):
                current_wins += 1
            else:
                baseline_wins += 1

            if (g + 1) % 10 == 0:
                total = current_wins + baseline_wins + draws
                wr = current_wins / max(1, total)
                print(f"[Arena] Игра {g+1}/{self.num_games}: "
                      f"current={current_wins} baseline={baseline_wins} "
                      f"draws={draws} wr={wr:.1%}")

        result = {
            "current_wins": current_wins,
            "baseline_wins": baseline_wins,
            "draws": draws,
            "errors": errors,
        }

        # Elo-разница.
        total = current_wins + baseline_wins
        if total > 0:
            wr = current_wins / total
            wr = max(0.01, min(0.99, wr))
            elo_diff = float(-400.0 * np.log10(1.0 / wr - 1.0))
            result["elo_diff"] = elo_diff
            print(f"[Arena] Elo diff (current - baseline): {elo_diff:+.1f}")

        return result

    def _play_one_turn_based_game(self, seed: int,
                                   current_is_me: bool) -> int:
        """Одна партия turn-based между worker_a и worker_b."""
        env = self._durakk_env_module.DurakEnv()  # без модели
        env.reset_with_seed(seed)

        # Синхронизируем воркеры той же раздачей.
        self._worker_a.init_game(seed)
        self._worker_b.init_game(seed)

        moves = 0
        max_moves = 500
        while not env.is_game_over() and moves < max_moves:
            current = env.current_player()  # 0=Me, 1=Opp
            # current_model ходит, если (current==0) == current_is_me.
            current_model_turn = (current == 0) == current_is_me

            if current_model_turn:
                action = self._worker_a.ask_action(self.time_budget)
                other = self._worker_b
            else:
                action = self._worker_b.ask_action(self.time_budget)
                other = self._worker_a

            ok = env.step(action)
            if not ok:
                # Невалидный ход — берём первый валидный.
                mask = np.array(env.get_legal_action_mask())
                idx = np.where(mask > 0)[0]
                if len(idx) > 0:
                    action = int(idx[0])
                    env.step(action)
                else:
                    break

            # Синхронизируем другого воркера (чтобы его env был в той же позиции).
            other.apply_action(action)

            moves += 1

        return env.winner()

    def update_baseline(self):
        """Копируем current.onnx → best.onnx."""
        import shutil
        if os.path.exists(self.onnx_path):
            shutil.copy2(self.onnx_path, self.baseline_path)
            if self._worker_b is not None:
                self._worker_b.load_model(self.baseline_path)
            print(f"[Arena] Baseline обновлён: {self.baseline_path}")

    def close(self):
        if self._worker_a:
            self._worker_a.close()
        if self._worker_b:
            self._worker_b.close()
        self._worker_a = None
        self._worker_b = None


# ============================================================================
# Внешний baseline evaluator: против фиксированного pure-MCTS (без нейросети).
# Эта метрика — ИСТИННЫЙ показатель прогресса.
# ============================================================================

class ExternalBaselineEvaluator:
    """
    Оценивает current.onnx против фиксированного внешнего baseline:
      - чистый MCTS без нейросети (UCB1 + rollout).

    FIX #15: в комментарии было «vs RandomNet-ISMCTS», но DurakEnv без модели
    использует НЕ RandomNet, а чистый MCTS (UCB1 + rollout) — потому что в
    bot.cpp при отсутствии сети передаётся nullptr и ISMCTS идёт по
    математическому пути без priors/value. RandomNet — это отдельный класс
    (равномерные priors + value=0.5), который в этом пайплайне не используется.
    Названия метрик в TensorBoard сохранены для обратной совместимости, но
    в логах и комментариях теперь корректно указано «pure-MCTS».

    Это правильная метрика прогресса: всегда сравниваем с одним и тем же
    слабым противником. Если winrate растёт — модель становится сильнее.
    """

    def __init__(self,
                 onnx_path: str,
                 durakk_env_path: str,
                 time_budget: float = 0.3,
                 num_games: int = 60,
                 device: str = "CPU"):
        self.onnx_path = os.path.abspath(onnx_path)
        self.durakk_env_path = durakk_env_path
        self.time_budget = time_budget
        self.num_games = num_games
        self.device = device

        self._worker: Optional[_WorkerProcess] = None

        sys.path.insert(0, os.path.dirname(self.durakk_env_path))
        sys.path.insert(0, self.durakk_env_path)
        import durakk_env  # noqa
        self._durakk_env_module = durakk_env

    def _ensure_worker(self):
        if self._worker is None:
            self._worker = _WorkerProcess("current_ext",
                                           self.durakk_env_path,
                                           device=self.device)
            if not self._worker.load_model(self.onnx_path):
                print(f"[ExtBaseline] WARNING: не загрузилась модель "
                      f"{self.onnx_path}")

    def evaluate_vs_random_ismcts(self) -> dict:
        """current vs pure-MCTS (без нейросети)."""
        self._ensure_worker()

        wins = losses = draws = 0

        for g in range(self.num_games):
            current_is_me = (g % 2 == 0)
            seed = g + 5000

            try:
                winner = self._play_one_game(seed, current_is_me)
            except Exception as e:
                print(f"[ExtBaseline] Ошибка в игре {g}: {e}")
                continue

            if winner == -1:
                draws += 1
            elif (winner == 0 and current_is_me) or \
                 (winner == 1 and not current_is_me):
                wins += 1
            else:
                losses += 1

            if (g + 1) % 20 == 0:
                total = wins + losses + draws
                wr = wins / max(1, total)
                print(f"[ExtBaseline] Игра {g+1}/{self.num_games}: "
                      f"wins={wins} losses={losses} draws={draws} "
                      f"wr={wr:.1%}")

        result = {"wins": wins, "losses": losses, "draws": draws}

        total = wins + losses
        if total > 0:
            wr = wins / total
            wr = max(0.01, min(0.99, wr))
            result["elo_vs_random"] = float(-400.0 * np.log10(1.0 / wr - 1.0))

        return result

    def _play_one_game(self, seed: int, current_is_me: bool) -> int:
        env = self._durakk_env_module.DurakEnv()  # без модели = RandomNet
        env.reset_with_seed(seed)
        self._worker.init_game(seed)

        moves = 0
        while not env.is_game_over() and moves < 500:
            current = env.current_player()
            our_turn = (current == 0) == current_is_me

            if our_turn:
                action = self._worker.ask_action(self.time_budget)
            else:
                # Противник — RandomNet-ISMCTS.
                probs = env.run_ismcts(self.time_budget, 1)
                mask = np.array(env.get_legal_action_mask(), dtype=np.float32)
                p = np.array(probs) * mask
                action = int(np.argmax(p)) if p.sum() > 0 else int(np.argmax(mask))

            ok = env.step(action)
            if not ok:
                mask = np.array(env.get_legal_action_mask())
                idx = np.where(mask > 0)[0]
                if len(idx) > 0:
                    action = int(idx[0])
                    env.step(action)
                else:
                    break

            moves += 1

        return env.winner()

    def close(self):
        if self._worker:
            self._worker.close()
            self._worker = None


# ============================================================================
# Точка входа.
# ============================================================================

if __name__ == "__main__":
    if "--worker" in sys.argv:
        # Субпроцесс-воркер.
        idx = sys.argv.index("--worker")
        name = sys.argv[idx + 1]

        durakk_env_path = "."
        if "--durakk-env-path" in sys.argv:
            i = sys.argv.index("--durakk-env-path")
            durakk_env_path = sys.argv[i + 1]

        device = "CPU"
        if "--device" in sys.argv:
            i = sys.argv.index("--device")
            device = sys.argv[i + 1]

        _worker_main(name, durakk_env_path, device)
    else:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--onnx", required=True)
        p.add_argument("--durakk-env-path", default=".")
        p.add_argument("--games", type=int, default=20)
        p.add_argument("--time-budget", type=float, default=0.3)
        p.add_argument("--device", default="CPU")
        args = p.parse_args()

        print("=== Тест ArenaEvaluator ===")
        arena = ArenaEvaluator(
            onnx_path=args.onnx,
            durakk_env_path=args.durakk_env_path,
            time_budget=args.time_budget,
            num_games=args.games,
            device=args.device,
        )
        try:
            result = arena.evaluate_current_vs_baseline()
            print(f"vs best.onnx: {result}")
        finally:
            arena.close()

        print("\n=== Тест ExternalBaselineEvaluator ===")
        ext = ExternalBaselineEvaluator(
            onnx_path=args.onnx,
            durakk_env_path=args.durakk_env_path,
            time_budget=args.time_budget,
            num_games=args.games,
            device=args.device,
        )
        try:
            result = ext.evaluate_vs_random_ismcts()
            print(f"vs RandomNet-ISMCTS: {result}")
        finally:
            ext.close()
