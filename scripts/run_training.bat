@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
REM ============================================================
REM run_training.bat
REM Hardware: Ryzen 7500F + 32GB DDR5 + RTX 4060 Ti 8GB
REM ============================================================

cd /d "%~dp0.."
cd training

set ITERATIONS=1000
set BATCH_SIZE=4096
set EPOCHS=4
set WORKERS=8
set GAMES_PER_ITER=8
set TIME_BUDGET=0.3
set NUM_BLOCKS=10
set NUM_CHANNELS=256
set FP16_FLAG=--fp16
set SP_DEVICE=CUDA
set RESUME=

:parse
if "%~1"=="" goto start
if "%~1"=="--quick" (
    set ITERATIONS=5
    set WORKERS=2
    set GAMES_PER_ITER=2
    set TIME_BUDGET=0.05
    set FP16_FLAG=--no_fp16
    set SP_DEVICE=CPU
    shift
    goto parse
)
if "%~1"=="--resume" (
    set RESUME=--resume %~2
    shift
    shift
    goto parse
)
if "%~1"=="--cpu" (
    set FP16_FLAG=--no_fp16
    set SP_DEVICE=CPU
    shift
    goto parse
)
echo Unknown arg: %~1
exit /b 1

:start
echo ============================================================
echo  Durak AlphaZero Training
echo  GPU: RTX 4060 Ti 8GB
echo  Iterations: %ITERATIONS%
echo  Workers: %WORKERS%  Games/iter: %GAMES_PER_ITER%
echo  Model: ResNet-SE %NUM_BLOCKS% blocks x %NUM_CHANNELS% ch
echo ============================================================

python -c "import durakk_env" >nul 2>&1
if errorlevel 1 (
    echo ERROR: durakk_env not found. Build engine first.
    exit /b 1
)

if not exist checkpoints mkdir checkpoints
if not exist runs mkdir runs
if not exist durakk_env.pyd copy /Y ..\durakk_env.pyd durakk_env.pyd >nul

echo Starting training...
python train.py ^
    --iterations %ITERATIONS% ^
    --batch_size %BATCH_SIZE% ^
    --epochs_per_iter %EPOCHS% ^
    --num_workers %WORKERS% ^
    --games_per_iter %GAMES_PER_ITER% ^
    --time_budget %TIME_BUDGET% ^
    --num_blocks %NUM_BLOCKS% ^
    --num_channels %NUM_CHANNELS% ^
    --buffer_size 1000000 ^
    --prioritized ^
    --arena_every 20 ^
    --arena_games 40 ^
    --ext_baseline_every 20 ^
    --ext_baseline_games 60 ^
    --ext_baseline_time 0.2 ^
    --save_every 10 ^
    --lr 2e-3 ^
    --lr_min 1e-5 ^
    --warmup_iters 50 ^
    --grad_clip 1.0 ^
    --weight_decay 1e-4 ^
    --durakk_env_path . ^
    --sp_device %SP_DEVICE% ^
    %FP16_FLAG% ^
    %RESUME%

echo Training complete.
echo Checkpoints: training\checkpoints\
echo TensorBoard: training\runs\
