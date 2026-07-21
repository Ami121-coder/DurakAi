"""
arena_math.py — Лёгкая arena для замера силы МАТЕМАТИЧЕСКОГО бота (без нейросети).

Запускает durakk_env напрямую (без ONNX-воркеров), играет N партий между двумя
конфигурациями, различающимися только time_budget. Это даёт:
  • baseline-кривую силы от времени на ход;
  • способ замерить улучшение от каждого этапа усиления (правим C++, пересобираем,
    прогоняем arena vs старый baseline).

Без внешних baseline-моделей: обе стороны — текущая C++ сборка. Различаем их
только time_budget: "кандидат" (больше времени) vs "baseline" (меньше).

Использование:
    python arena_math.py --games 40 --cand_time 0.5 --base_time 0.1
    python arena_math.py --games 100 --cand_time 0.3 --base_time 0.3  # обе стороны равны → ≈50%

Метрики:
    winrate, draws, Elo diff, среднее число ходов, время/партия.
"""

import os
import sys
import time
import argparse
import numpy as np

# durakk_env лежит в корне проекта и в training/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durakk_env


def softmax_argmax_action(env, time_budget: float) -> int:
    """Выбор хода: ISMCTS → argmax(visit_probs * legal_mask)."""
    probs = env.run_ismcts(time_budget, 1)
    mask = np.array(env.get_legal_action_mask(), dtype=np.float32)
    p = np.array(probs) * mask
    if p.sum() > 0:
        return int(np.argmax(p))
    return int(np.argmax(mask)) if mask.sum() > 0 else 37


def play_one_game(seed: int, cand_is_me: bool,
                  cand_time: float, base_time: float,
                  max_moves: int = 500) -> int:
    """
    Игра между кандидатом (cand_time) и baseline (base_time).
    cand_is_me=True → кандидат играет за Me (сторона 0).
    Возвращает winner: 0/1 или -1 (ничья/таймаут).
    """
    env = durakk_env.DurakEnv()
    env.reset_with_seed(seed)

    for _ in range(max_moves):
        if env.is_game_over():
            break
        current = env.current_player()
        our_turn = (current == 0) == cand_is_me
        tb = cand_time if our_turn else base_time

        action = softmax_argmax_action(env, tb)
        ok = env.step(action)
        if not ok:
            mask = np.array(env.get_legal_action_mask())
            idx = np.where(mask > 0)[0]
            if len(idx) > 0:
                env.step(int(idx[0]))
            else:
                break

    return env.winner()


def run_arena(games: int, cand_time: float, base_time: float,
              seed_offset: int = 0, verbose: bool = True):
    wins = losses = draws = errors = 0
    total_moves = 0
    t0 = time.time()

    for g in range(games):
        cand_is_me = (g % 2 == 0)  # чередуем стороны
        seed = g + 10000 + seed_offset

        try:
            winner = play_one_game(seed, cand_is_me, cand_time, base_time)
        except Exception as e:
            if verbose:
                print(f"  game {g}: ОШИБКА: {e}")
            errors += 1
            continue

        if winner == -1:
            draws += 1
        elif (winner == 0 and cand_is_me) or (winner == 1 and not cand_is_me):
            wins += 1
        else:
            losses += 1

        if verbose and (g + 1) % max(1, games // 10) == 0:
            total = wins + losses + draws
            wr = wins / max(1, total)
            print(f"  game {g+1}/{games}: W={wins} L={losses} D={draws} "
                  f"wr={wr:.1%} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    total = wins + losses + draws
    wr = wins / max(1, total)

    # Elo: clamp wr в [0.01, 0.99] чтобы избежать log(0).
    wr_clamped = max(0.01, min(0.99, wins / max(1, wins + losses)))
    elo_diff = float(-400.0 * np.log10(1.0 / wr_clamped - 1.0))

    result = {
        "games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "errors": errors,
        "winrate": wr,
        "elo_diff": elo_diff,
        "time_sec": elapsed,
        "sec_per_game": elapsed / max(1, total),
        "cand_time_budget": cand_time,
        "base_time_budget": base_time,
    }
    return result


def main():
    p = argparse.ArgumentParser(
        description="Arena для чисто математического бота (без нейросети).")
    p.add_argument("--games", type=int, default=40,
                   help="Число партий (рекомендуется ≥ 100 для низкой дисперсии).")
    p.add_argument("--cand_time", type=float, default=0.3,
                   help="time_budget кандидата (сек/ход).")
    p.add_argument("--base_time", type=float, default=0.1,
                   help="time_budget baseline (сек/ход).")
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    print(f"Arena: кандидат (time={args.cand_time}s) vs baseline (time={args.base_time}s)")
    print(f"Партий: {args.games}, движок БЕЗ нейросети (чистая математика).")
    print()

    result = run_arena(
        games=args.games,
        cand_time=args.cand_time,
        base_time=args.base_time,
        seed_offset=args.seed_offset,
        verbose=not args.quiet,
    )

    print()
    print("=" * 50)
    print(f"Партий сыграно:  {result['games']}")
    print(f"Победы:          {result['wins']}")
    print(f"Поражения:       {result['losses']}")
    print(f"Ничьи:           {result['draws']}")
    print(f"Ошибки:          {result['errors']}")
    print(f"Winrate:         {result['winrate']:.1%}")
    print(f"Elo diff:        {result['elo_diff']:+.1f}")
    print(f"Время:           {result['time_sec']:.0f}s "
          f"({result['sec_per_game']:.1f}s/партия)")
    print("=" * 50)

    return result


if __name__ == "__main__":
    main()
