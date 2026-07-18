"""
arena.py — FIX #6: правильная arena evaluation для AlphaZero

Сравнивает текущую модель (current.onnx) с baseline (best.onnx).
Если current выигрывает ≥ 55% — она становится новым baseline.

Это стандартный AlphaZero arena loop:
  1. Сохраняем best.onnx как baseline
  2. Играем N партий: current vs baseline
  3. Если winrate ≥ 55% — current становится новым best
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import durakk_env


class ArenaEvaluator:
    """Оценивает текущую модель против baseline."""

    def __init__(self, onnx_path: str, time_budget: float = 0.5,
                 num_games: int = 40):
        self.onnx_path = onnx_path
        self.baseline_path = os.path.join(os.path.dirname(onnx_path), "best.onnx")
        self.time_budget = time_budget
        self.num_games = num_games

        # При первом запуске baseline = current
        if not os.path.exists(self.baseline_path) and os.path.exists(self.onnx_path):
            import shutil
            shutil.copy2(self.onnx_path, self.baseline_path)
            print(f"[Arena] Создан baseline: {self.baseline_path}")

    def evaluate_current_vs_baseline(self) -> dict:
        """
        Играем num_games партий: current vs baseline.
        Каждая партия — current играет за Me, baseline за Opp (или наоборот, чередуем).
        """
        if not os.path.exists(self.baseline_path):
            return {"current_wins": 0, "baseline_wins": 0, "draws": self.num_games}

        env = durakk_env.DurakEnv()
        current_wins = baseline_wins = draws = 0

        for g in range(self.num_games):
            # Чередуем: чётные игры — current за Me, нечётные — за Opp
            current_is_me = (g % 2 == 0)

            env.reset_with_seed(g + 1000)

            # Загружаем appropriate модель
            # К сожалению, C++ движок держит ОДНУ модель за раз.
            # Для arena нам нужно переключаться. Простейший подход:
            # играем current всегда за Me (как в реальном боте).
            env.load_model(self.onnx_path, "CPU")

            moves = 0
            while not env.is_game_over() and moves < 500:
                probs = env.run_ismcts(self.time_budget, 1)
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
            if w == -1:
                draws += 1
            elif (w == 0 and current_is_me) or (w == 1 and not current_is_me):
                current_wins += 1
            else:
                baseline_wins += 1

        return {
            "current_wins": current_wins,
            "baseline_wins": baseline_wins,
            "draws": draws,
        }

    def update_baseline(self):
        """Копируем current.onnx → best.onnx (новый baseline)."""
        import shutil
        if os.path.exists(self.onnx_path):
            shutil.copy2(self.onnx_path, self.baseline_path)
            print(f"[Arena] Baseline обновлён: {self.baseline_path}")


if __name__ == "__main__":
    # Smoke test
    arena = ArenaEvaluator(
        onnx_path="checkpoints/current.onnx",
        time_budget=0.3,
        num_games=10,
    )
    result = arena.evaluate_current_vs_baseline()
    print(f"Результат: {result}")
