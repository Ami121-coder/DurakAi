"""
build_endgame_db.py — построение Endgame Database (Task 7)

Запускает buildEndgameDB() из C++ (через durakk_env), сохраняет результат
в бинарный файл engine/data/endgame_db.bin.

Размер БД для maxTotal=4:
  - C(36,1)×C(35,1) + C(36,2)×C(34,2) + C(36,3)×C(33,3) ≈ 1.4M пар
  - × 4 trump × 2 attacker = ~11M позиций
  - × 4 байта = ~44 MB
  - Время построения: ~5-15 минут (50ms на позицию)

Использование:
  cd training
  python build_endgame_db.py --max-total 4 --output ../engine/data/endgame_db.bin

После построения бот может загрузить БД:
  bot.loadEndgameDB("engine/data/endgame_db.bin")
"""

import os
import sys
import time
import argparse

# Добавляем корень проекта в sys.path (там лежит durakk_env.so / .pyd).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

try:
    import durakk_env
except ImportError as e:
    print(f"Ошибка: не удалось импортировать durakk_env. {e}")
    print("Сначала соберите движок: npm run build:engine")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Построение Endgame Database (Task 7)")
    parser.add_argument("--max-total", type=int, default=4,
                        help="Максимальное суммарное число карт на руках (≤6, рекомендую 4)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(ROOT, "engine", "data", "endgame_db.bin"),
                        help="Путь к выходному файлу БД")
    args = parser.parse_args()

    if args.max_total < 2 or args.max_total > 6:
        print(f"max-total должен быть в [2, 6], получено {args.max_total}")
        sys.exit(1)

    print(f"Построение Endgame Database (max-total={args.max_total})")
    print(f"Выходной файл: {args.output}")
    print(f"Оценка времени: ~{args.max_total ** 3} минут")
    print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    t0 = time.time()
    if not hasattr(durakk_env, "build_endgame_db"):
        print("ОШИБКА: в durakk_env нет функции build_endgame_db.")
        print("Убедитесь что движок собран с Task 7 (endgame_db.cpp в CMakeLists).")
        sys.exit(1)

    n_records = durakk_env.build_endgame_db(args.output, args.max_total)
    elapsed = time.time() - t0

    print(f"\nГотово за {elapsed:.1f} сек ({elapsed/60:.1f} мин)")
    print(f"Записей в БД: {n_records:,}")
    print(f"Размер файла: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")
    print(f"Путь: {args.output}")
    print()
    print("Для использования в боте:")
    print(f'  bot.loadEndgameDB("{args.output}")')


if __name__ == "__main__":
    main()
