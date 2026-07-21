#!/bin/bash
# ============================================================================
# run_training.sh — launcher с оптимальными параметрами под:
#   CPU: Ryzen 5 7500F (6 ядер / 12 потоков Zen 4)
#   RAM: 32GB DDR5
#   GPU: RTX 4060 Ti 8GB
#
# Использование:
#   cd C:\path\to\DurakAi\training
#   bash run_training.sh             # стандартный запуск
#   bash run_training.sh --resume checkpoints/current.pt   # продолжить
#   bash run_training.sh --quick     # быстрый smoke-test (5 итераций)
# ============================================================================

set -e

# Переходим в каталог training/ (где лежит этот скрипт).
cd "$(dirname "$0")"

# Параметры по умолчанию (оптимальные для 7500F + 4060 Ti 8GB).
ITERATIONS=1000
BATCH_SIZE=4096
EPOCHS=4
WORKERS=8
GAMES_PER_ITER=8
TIME_BUDGET=0.3
NUM_BLOCKS=10
NUM_CHANNELS=256
FP16_FLAG="--fp16"

# Обработка аргументов.
RESUME=""
QUICK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK=1
            ITERATIONS=5
            WORKERS=2
            GAMES_PER_ITER=2
            TIME_BUDGET=0.05
            shift
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --iterations)
            ITERATIONS=$2
            shift 2
            ;;
        --workers)
            WORKERS=$2
            shift 2
            ;;
        --cpu)
            # Принудительный CPU-режим (если CUDA недоступна).
            FP16_FLAG="--no_fp16 --sp_device CPU"
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo " Durak AlphaZero Training"
echo "============================================================"
echo " Hardware target: Ryzen 7500F + 32GB DDR5 + RTX 4060 Ti 8GB"
echo " Iterations:      $ITERATIONS"
echo " Batch size:      $BATCH_SIZE"
echo " Workers:         $WORKERS"
echo " Games/iter:      $GAMES_PER_ITER"
echo " Time budget:     $TIME_BUDGET sec/move"
echo " Model:           ResNet-SE $NUM_BLOCKS blocks × $NUM_CHANNELS ch"
echo " FP16:            enabled"
echo "============================================================"

# Проверим, что durakk_env собран.
if ! python -c "import durakk_env" 2>/dev/null; then
    echo "ERROR: durakk_env не найден. Соберите движок:"
    echo "  cd /path/to/DurakAi && npm run build:engine:clean"
    exit 1
fi

# Проверим CUDA.
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; vram = torch.cuda.get_device_properties(0).total_memory / 1024**3; print(f'VRAM: {vram:.1f} GB')"
else
    echo "WARNING: CUDA недоступна — будет CPU-режим (очень медленно)."
    FP16_FLAG="--no_fp16 --sp_device CPU"
fi

# Создаём директории.
mkdir -p checkpoints
mkdir -p runs

# Запуск TensorBoard в фоне (опционально).
if command -v tensorboard &>/dev/null; then
    echo "Запускаю TensorBoard на http://localhost:6006 ..."
    tensorboard --logdir=runs --port=6006 &
    TB_PID=$!
    echo "TensorBoard PID: $TB_PID"
    trap "kill $TB_PID 2>/dev/null" EXIT
fi

# Главный запуск.
echo ""
echo "Запускаем обучение..."
python train.py \
    --iterations $ITERATIONS \
    --batch_size $BATCH_SIZE \
    --epochs_per_iter $EPOCHS \
    --num_workers $WORKERS \
    --games_per_iter $GAMES_PER_ITER \
    --time_budget $TIME_BUDGET \
    --num_blocks $NUM_BLOCKS \
    --num_channels $NUM_CHANNELS \
    --buffer_size 1000000 \
    --prioritized \
    --arena_every 20 \
    --arena_games 40 \
    --ext_baseline_every 20 \
    --ext_baseline_games 60 \
    --ext_baseline_time 0.2 \
    --save_every 10 \
    --lr 2e-3 \
    --lr_min 1e-5 \
    --warmup_iters 50 \
    --grad_clip 1.0 \
    --weight_decay 1e-4 \
    --durakk_env_path .. \
    $FP16_FLAG \
    $RESUME

echo ""
echo "============================================================"
echo " Обучение завершено."
echo " Лучшие чекпойнты в: checkpoints/"
echo "   - best.pt          (по winrate vs предыдущего baseline)"
echo "   - best_vs_random.pt (по Elo vs RandomNet-ISMCTS)"
echo "   - final.pt          (последняя итерация)"
echo " Логи: checkpoints/metrics.jsonl"
echo " TensorBoard: runs/durak_az"
echo "============================================================"
