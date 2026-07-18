# Обучение нейросети для Durak Online — инструкция

## 0. Что в этом каталоге

Все файлы в `patches/` — это **исправленные или новые** версии файлов из проекта `durakk`. Применяются поверх существующего кода.

| Файл | Назначение | Применение |
|---|---|---|
| `bindings.cpp.patched` | Исправлены 5 критических багов pybind11 | скопировать поверх `engine/src/bindings.cpp` |
| `ismcts.h.patched` | PUCT, Dirichlet, virtual losses, PolicyValueNet* | скопировать поверх `engine/src/ismcts.h` |
| `ismcts.cpp.patched` | Реализация PUCT + leaf eval через сеть | скопировать поверх `engine/src/ismcts.cpp` |
| `policy_value_net.h.patched` | Добавлен `OnnxNet` класс | скопировать поверх `engine/src/nnet/policy_value_net.h` |
| `onnx_net.cpp` | Реализация `OnnxNet` через ONNX Runtime | положить в `engine/src/nnet/onnx_net.cpp` |
| `CMakeLists.txt.patched` | ONNX Runtime + AVX-512 | скопировать поверх `engine/CMakeLists.txt` |
| `model_resnet.py` | ResNet-8 архитектура | положить в `training/model_resnet.py` |
| `model_transformer.py` | Альтернативная Transformer архитектура | положить в `training/model_transformer.py` |
| `replay_buffer.py` | Кольцевой буфер 1M позиций | положить в `training/replay_buffer.py` |
| `self_play.py` | Async self-play воркеры | положить в `training/self_play.py` |
| `train.py.patched` | Полный AlphaZero-цикл | скопировать поверх `training/train.py` |
| `vector_env.py.patched` | Совместимый fallback | скопировать поверх `training/vector_env.py` |
| `export_onnx.py` | Экспорт .pt → .onnx | положить в `training/export_onnx.py` |

## 1. Применение патчей (Windows)

```powershell
cd C:\path\to\durakk

# Бэкап оригиналов
copy engine\src\bindings.cpp engine\src\bindings.cpp.orig
copy engine\src\ismcts.h engine\src\ismcts.h.orig
copy engine\src\ismcts.cpp engine\src\ismcts.cpp.orig
copy engine\src\nnet\policy_value_net.h engine\src\nnet\policy_value_net.h.orig
copy engine\CMakeLists.txt engine\CMakeLists.txt.orig
copy training\train.py training\train.py.orig
copy training\vector_env.py training\vector_env.py.orig

# Применяем патчи (предполагаем, что patches/ — содержимое этого каталога)
copy patches\bindings.cpp.patched engine\src\bindings.cpp
copy patches\ismcts.h.patched engine\src\ismcts.h
copy patches\ismcts.cpp.patched engine\src\ismcts.cpp
copy patches\policy_value_net.h.patched engine\src\nnet\policy_value_net.h
copy patches\onnx_net.cpp engine\src\nnet\onnx_net.cpp
copy patches\CMakeLists.txt.patched engine\CMakeLists.txt
copy patches\train.py.patched training\train.py
copy patches\vector_env.py.patched training\vector_env.py

# Новые файлы
copy patches\model_resnet.py training\model_resnet.py
copy patches\model_transformer.py training\model_transformer.py
copy patches\replay_buffer.py training\replay_buffer.py
copy patches\self_play.py training\self_play.py
copy patches\export_onnx.py training\export_onnx.py
```

## 2. Установка ONNX Runtime (для C++ OnnxNet)

### Вариант A: vcpkg (рекомендуется на Windows)
```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install onnxruntime-cuda:x64-windows
# В CMake:
cmake -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ..
```

### Вариант B: системная установка (Linux)
```bash
# Ubuntu/Debian
sudo apt install onnxruntime-dev

# Или из бинарников Microsoft:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1.tgz
tar xzf onnxruntime-linux-x64-gpu-1.17.1.tgz
sudo cp -r onnxruntime-linux-x64-gpu-1.17.1/include/* /usr/local/include/
sudo cp onnxruntime-linux-x64-gpu-1.17.1/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig
```

## 3. Установка Python-зависимостей

```bash
cd training
pip install -r requirements.txt
# Дополнительно:
pip install onnx onnxruntime  # для экспорта и теста
pip install onnxruntime-gpu   # для GPU inference в Python (опционально)
```

Обновлённый `requirements.txt` (рекомендуется):
```
torch>=2.0
numpy>=1.24
tensorboard>=2.13
tqdm>=4.65
onnx>=1.14
onnxruntime>=1.16
onnxruntime-gpu>=1.16  # опционально, для GPU inference в Python
```

## 4. Сборка движка

```bash
npm run build:engine:clean
```

Если ONNX Runtime не найден — CMake выдаст warning, и `OnnxNet` будет заглушкой
(fallback на `RandomNet`). Это ок для первого запуска.

## 5. Sanity test (без обучения)

```python
# training/sanity_test.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import durakk_env

env = durakk_env.DurakEnv()
env.reset()
print(f"Initial: deckRemaining={env.get_stats()['deckRemaining']}, "
      f"tableLen={env.get_stats()['tableLen']}, "
      f"isGameOver={env.is_game_over()}")

moves = 0
while not env.is_game_over() and moves < 200:
    probs = env.run_ismcts(0.05, 1)
    mask = list(env.get_legal_action_mask())
    import numpy as np
    p = np.array(probs) * np.array(mask)
    action = int(np.argmax(p))
    env.step(action)
    moves += 1

print(f"Game over after {moves} moves. Winner={env.winner()}")
```

Должна доиграть до конца без ошибок. Если `is_game_over()` возвращает `True`
сразу после `reset()` — патч `bindings.cpp.patched` не применён.

## 6. Запуск обучения

```bash
cd training

# Полный AlphaZero-цикл, 1000 итераций.
python train.py --iterations 1000 \
                --batch_size 4096 \
                --epochs_per_iter 4 \
                --num_workers 6 \
                --time_budget 0.5 \
                --fp16

# TensorBoard в другом терминале:
tensorboard --logdir=runs
```

Ожидаемая скорость на 7500F + 4060 Ti:
- Self-play: ~6-10 партий/мин на воркер × 6 воркеров = 36-60 партий/мин
- За итерацию (4 партии × 6 воркеров = 24 партии): ~30-60 секунд
- Training: 4 эпохи × 4096 батч × FP16 = ~3-5 секунд на 4060 Ti
- Итого: ~40-70 секунд/итерация
- 1000 итераций: ~12-20 часов

## 7. Экспорт в ONNX

```bash
python export_onnx.py --checkpoint checkpoints/model_100.pt --output model.onnx
```

Положите `model.onnx` в корень проекта. `Bot` в `main.cpp` нужно обновить,
чтобы он создавал `OnnxNet("model.onnx", "CUDA", 0)` вместо `RandomNet`.

## 8. Интеграция в C++ бота

В `engine/src/bot.cpp` (после патча или вручную):

```cpp
Bot::Bot() {
    // Сначала пробуем ONNX.
    try {
        net_ = std::make_unique<OnnxNet>("model.onnx", "CUDA", 0);
        if (!net_->isReady()) {
            std::fprintf(stderr, "[Bot] OnnxNet не готов, fallback на RandomNet.\n");
            net_ = std::make_unique<RandomNet>();
        }
    } catch (...) {
        net_ = std::make_unique<RandomNet>();
    }
}
```

И в `Bot::decide` передавай `net_.get()` в `runIsmcts`:

```cpp
IsmctsResult r = runIsmcts(root, k, lim, nullptr, Player::Me, net_.get());
```

## 9. Метрики приёмки

| Метрика | Цель | Как измерить |
|---|---|---|
| Sanity test | 100 партий доигрывают до конца без ошибок | `python sanity_test.py` |
| Buffer fill rate | ≥ 500k позиций за 100 итераций | TensorBoard: `Buffer/size` |
| Policy loss | < 1.5 после 200 итераций | TensorBoard: `Loss/Policy` |
| Value loss | < 0.3 после 200 итераций | TensorBoard: `Loss/Value` |
| Winrate vs baseline (RandomNet-ISMCTS) | ≥ 60% после 200 итераций, ≥ 70% после 500 | Arena evaluation |
| Latency в C++ (Strength::Normal) | ≤ 2 сек/ход | UI stopwatch |
| Endgame solved | 100% на пустой колоде до 12 карт | Тестовые позиции |

## 10. Откат (если что-то сломалось)

```bash
# Восстановить оригиналы:
mv engine\src\bindings.cpp.orig engine\src\bindings.cpp
# ... и т.д.
npm run build:engine:clean
```

## 11. Known issues / TODO

- `onnx_net.cpp`: requires `onnxruntime_cxx_api.h` — устанавливается вместе с onnxruntime-dev.
- `self_play.py`: на Windows multiprocessing использует `spawn` — первый запуск будет медленным (каждый воркер импортирует torch заново).
- `train.py`: если GPU < 8GB VRAM — уменьшите `--batch_size` до 2048 или `--num_channels` до 128.
- AVX-512: на старых CPU (pre-Zen 4) уберите `-mavx512f -mavx512bw` и `/arch:AVX512`.
- TTable в эндшпиле: добавьте eviction policy (например, LRU на 1M записей) — пока не сделано.
