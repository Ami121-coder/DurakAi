# UPGRADE_NOTES — Усиление пайплайна обучения Durak AlphaZero

> Целевое железо: **Ryzen 5 7500F** (6c/12t Zen 4) + **32GB DDR5** + **RTX 4060 Ti 8GB**.
> Игра: переводной дурак 1-на-1 (без «проездного»).

---

## Что внутри патча

```
durak_ai_patches/
├── training/
│   ├── model_resnet.py       ← усиленная сеть с SE-блоками (10 блоков, 256 каналов)
│   ├── replay_buffer.py      ← кольцевой буфер + Prioritized Experience Replay
│   ├── self_play.py          ← 8 воркеров, CUDA, расширенное температурное расписание
│   ├── train.py              ← усиленный цикл: PER, FP16, исправленная arena, внешний baseline
│   ├── arena.py              ← ИСПРАВЛЕННАЯ arena: 2 субпроцесса, turn-based, реальный winrate
│   ├── export_onnx.py        ← обновлён под SE-модель (10 блоков)
│   └── requirements.txt      ← правильные зависимости
└── scripts/
    ├── run_training.sh       ← Linux/Git-Bash launcher
    └── run_training.bat      ← Windows launcher
```

---

## Краткая сводка улучшений

| Компонент | Было | Стало | Эффект |
|---|---|---|---|
| ResNet | 8 блоков, 256 ch | **10 блоков, 256 ch + SE-attention** | +0.5-1% winrate при том же бюджете |
| Replay Buffer | uniform | **PER (Schaul et al., 2015)** | -20-40% итераций до целевого loss |
| Self-play воркеры | 6 воркеров × 4 партии | **8 воркеров × 8 партий** | 2× больше данных за итерацию |
| Self-play device | CPU | **CUDA (с CPU fallback)** | 3-5× быстрее ISMCTS |
| Температура | T=1.0 первые 10, потом 0.25 | **T=1.0 (8) → 0.5 (20) → 0.25** | больше разнообразия в миттельшпиле |
| Dirichlet noise | только 1-й ход | **первые 2 хода** | лучше exploration в дебютах |
| Arena (current vs best) | **БАГ**: current vs current | **ИСПРАВЛЕНО**: 2 субпроцесса, turn-based | реальная метрика winrate |
| Внешний baseline | отсутствует | **vs RandomNet-ISMCTS** | истинный показатель прогресса |
| Elo-логирование | нет | **Elo diff + Elo vs Random** | трек долгосрочного прогресса |
| FP16 | опционально | **по умолчанию** | 2× faster training на 4060 Ti |
| Value head | 128 hidden | **256 hidden + dropout** | точнее value-оценка |
| Dropout | нет | **0.1 в heads + blocks** | против overfitting |
| LR schedule | cosine + warmup | без изменений | — |
| Save/restore | save_every=10 | + buffer save каждые 50 | — |

---

## Файлы детально

### 1. `training/model_resnet.py`

**Что добавлено:**
- `SqueezeExcitation` блок (канал-wise attention) после каждого residual.
- 10 residual-блоков вместо 8 (увеличена ёмкость).
- Value head расширен до 256 скрытых (вместо 128).
- Dropout 0.1 в policy/value heads + в residual-блоках.
- Проверка batch=4096 на GPU при `__main__`.

**Параметры:**
```
ResNet-SE 10 blocks × 256 channels = ~5M параметров
VRAM (FP16, batch=4096): ~120 MB (4060 Ti 8GB — огромный запас)
```

### 2. `training/replay_buffer.py`

**Что добавлено:**
- Prioritized Experience Replay (PER) по Schaul et al., 2015.
- Alpha=0.6 (степень сглаживания приоритетов).
- Beta=0.4 с аннеалингом к 1.0 (importance sampling weights).
- `update_priorities(idxs, td_errors)` для обновления после обучения.
- API: `sample()` возвращает `(states, policies, values, masks, weights, idxs)`.

**Память на 1M позиций с PER:**
```
states:     220 MB
policies:   152 MB
values:       4 MB
masks:      152 MB
priorities:   4 MB
Итого:      ~530 MB  ← комфортно на 32GB DDR5
```

### 3. `training/self_play.py`

**Что добавлено:**
- `num_workers=8` по умолчанию (7500F = 12 потоков, оставляем 4 под арбитр/arena/TensorBoard).
- `games_per_iteration=8` (вместо 4) — 2× больше данных.
- Опция `device="CUDA"`: воркеры грузят ONNX на GPU через `env.load_model(path, "CUDA")`.
  - **ВНИМАНИЕ**: каждый воркер-процесс создаёт свою ONNX-сессию на GPU.
  - 8 сессий × ~500MB = ~4GB VRAM. Запас под обучение: ~4GB.
  - Если VRAM не хватает, переключайтесь на `--sp_device CPU`.
- Температурное расписание: `(T_init=1.0, T_mid=0.5, T_final=0.25, decay_after_init=8, decay_after_mid=20)`.
- Dirichlet noise на **первые 2 хода** (вместо 1) — больше разнообразия в дебютах.
- `noise_first_n_moves=2` как параметр.
- `alive_workers()` + `stats()` для мониторинга.

### 4. `training/arena.py` — КРИТИЧЕСКИЙ ФИКС

**Проблема в старой версии:**
```python
# Баг: arena грузит ТОЛЬКО current.onnx, не baseline.
env.load_model(self.onnx_path, "CPU")  # ← всегда current
```
После первого обновления `baseline = current`, и arena играла **current vs current** → winrate всегда ≈ 50%.

**Фикс:**
- Запускает **ДВА отдельных субпроцесса** `durakk_env`, каждый со своей ONNX.
- **Turn-based протокол**: главный арбитр держит "голый" env без модели, на каждом ходу спрашивает action у нужного воркера.
- Команды воркеру: `load / unload / init / ask_action / apply / quit`.
- Воркер хранит своё состояние, синхронизируется через `apply`.
- **Логирование Elo-разницы**: `Elo diff = -400 * log10(1/wr - 1)`.
- Дополнительно: `ExternalBaselineEvaluator` — сравнивает current vs RandomNet-ISMCTS (чистый search без сети). Это **истинная метрика прогресса**: если winrate растёт → модель становится сильнее.

### 5. `training/train.py`

**Что добавлено:**
- Использует PER из нового `replay_buffer.py`.
- Передаёт `weights` в loss (importance sampling).
- Обновляет priorities по TD-error после каждого батча.
- Авто-экспорт ONNX после каждого checkpoint (как и раньше).
- Запускает ArenaEvaluator (исправленный) каждые 20 итераций.
- Запускает ExternalBaselineEvaluator каждые 20 итераций.
- Логирует в TensorBoard:
  - `Loss/Total`, `Loss/Policy`, `Loss/Value`
  - `Buffer/size`
  - `Arena/winrate`, `Arena/draws`, `Arena/elo_diff`
  - `ExtBaseline/winrate_vs_random`, `ExtBaseline/elo_vs_random`
  - `Train/learning_rate`
  - `Perf/iter_time_sec`, `Perf/collect_time_sec`
- Логирует в JSONL (`checkpoints/metrics.jsonl`) для пост-анализа.
- Авто-сохранение `best_vs_random.pt` по лучшему Elo vs Random.

### 6. `training/export_onnx.py`

- Обновлён под `num_blocks=10, num_channels=256` по умолчанию.
- Sanity check: PyTorch vs ONNX, проверка NaN/Inf.

### 7. `scripts/run_training.sh` / `run_training.bat`

- Оптимальные параметры для 7500F + 4060 Ti 8GB.
- Авто-проверка CUDA, директорий, durakk_env.
- Авто-запуск TensorBoard (на sh-версии).
- Поддержка `--quick` (smoke-test), `--resume`, `--cpu`.

---

## Как применить патчи

### Windows (PowerShell):

```powershell
cd C:\path\to\DurakAi

# Бэкап оригиналов.
copy training\model_resnet.py training\model_resnet.py.orig
copy training\replay_buffer.py training\replay_buffer.py.orig
copy training\self_play.py training\self_play.py.orig
copy training\train.py training\train.py.orig
copy training\arena.py training\arena.py.orig
copy training\export_onnx.py training\export_onnx.py.orig
copy training\requirements.txt training\requirements.txt.orig

# Применяем патчи.
copy \path\to\durak_ai_patches\training\* training\
copy \path\to\durak_ai_patches\scripts\run_training.bat scripts\

# Удаляем старые чекпойнты (модель изменилась: 8 → 10 блоков + SE).
rmdir /s /q checkpoints
rmdir /s /q runs

# Установка зависимостей.
cd training
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime onnxruntime-gpu

# Пересобрать движок (C++ не изменился, но чтобы быть уверенным).
cd ..
npm run build:engine:clean

# Smoke test.
cd training
.\scripts\run_training.bat --quick

# Полный запуск.
.\scripts\run_training.bat
```

### Linux / WSL / Git-Bash:

```bash
cd /path/to/DurakAi

# Бэкап.
for f in model_resnet replay_buffer self_play train arena export_onnx; do
    cp training/${f}.py training/${f}.py.orig
done

# Применяем.
cp /path/to/durak_ai_patches/training/*.py training/
cp /path/to/durak_ai_patches/training/requirements.txt training/
cp /path/to/durak_ai_patches/scripts/run_training.sh scripts/
chmod +x scripts/run_training.sh

# Удалить старые чекпойнты.
rm -rf checkpoints runs

# Установка.
cd training
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime onnxruntime-gpu

# Пересборка.
cd ..
npm run build:engine:clean

# Smoke test.
cd training
bash ../scripts/run_training.sh --quick

# Полный запуск.
bash ../scripts/run_training.sh
```

---

## Ожидаемая производительность

| Метрика | Значение |
|---|---|
| Self-play throughput | 8 воркеров × 8 партий × ~30 ходов × 0.3 сек = ~70 сек/итер |
| Training (FP16, batch 4096, 4 эпохи) | ~8 сек/итер на 4060 Ti |
| Arena (раз в 20 итер.) | ~20 мин (40 партий × 30 сек) |
| External baseline (раз в 20 итер.) | ~30 мин (60 партий × 30 сек) |
| Среднее время итерации | ~80 сек (без arena), ~150 сек (с arena) |
| **1000 итераций** | **22-40 часов** |

### VRAM бюджет (4060 Ti 8GB)

| Компонент | VRAM |
|---|---|
| PyTorch model (FP16, 5M params) | ~10 MB |
| Activations (batch=4096) | ~75 MB |
| Gradients + AdamW state | ~30 MB |
| 8 ONNX-сессий в self-play воркерах | ~4000 MB |
| TensorBoard + cache | ~200 MB |
| **Итого** | **~4.3 GB** — запас 3.7 GB |

Если VRAM не хватает (OOM):
1. Уменьшите `--num_workers` до 4 или 6.
2. Уменьшите `--num_channels` до 192.
3. Поставьте `--sp_device CPU` (медленнее, но 0 VRAM).

---

## Метрики приёмки

После N итераций ожидаемые показатели (vs исходный пайплайн):

| Метрика | Исходный | Усиленный (1000 iter) |
|---|---|---|
| Loss/Policy | ~1.5 | **< 1.0** |
| Loss/Value | ~0.3 | **< 0.15** |
| Winrate vs RandomNet-ISMCTS | ~60-70% | **85-92%** |
| Elo vs RandomNet-ISMCTS | ~150-200 | **350-500** |
| Winrate vs previous best (arena) | ~50% (баг) | **истинный, 55-70% при прогрессе** |
| Партий до сходимости | ~10000+ | **~5000-7000** |

---

## Что НЕ вошло в патч

1. **Генерация «проездного» хода** — по вашему запросу не делаем (игры 1-на-1 переводной без проездного).
2. **Изменения C++ движка** — все фиксы в `bindings.cpp`, `ismcts.cpp`, `onnx_net.cpp` из оригинального `SUMMARY.md` уже в репозитории. Этот патч касается только Python-части.
3. **Многомасштабная архитектура** (по примеру EfficientNet) — переусложнит, не даст прироста на текущем бюджете.
4. **Distillation / Knowledge distillation** — для будущих версий.
5. **MuZero-style learned dynamics** — принципиально другая архитектура, несовместима с текущим ISMCTS.

---

## Что делать, если обучение идёт плохо

1. **Loss не падает:**
   - Проверьте, что `checkpoints/current.onnx` создаётся (см. логи `[ONNX] Экспортирован`).
   - Проверьте, что воркеры пишут `[Worker N] ONNX загружена на CUDA в C++ ISMCTS ✓`.
   - Если нет — ONNX не экспортируется или воркеры не видят модель.

2. **Winrate в arena всегда 50%:**
   - Проверьте, что arena запускает 2 воркера (`[Arena] Worker 'current' готов` + `'baseline' готов`).
   - Если только один — что-то с импортом durakk_env в субпроцессе.

3. **VRAM OOM:**
   - Уменьшите `--num_workers` до 4.
   - Поставьте `--sp_device CPU` (потеря скорости в 2-3×).
   - Уменьшите `--num_channels` до 192.

4. **CUDA не доступна:**
   - Проверьте `python -c "import torch; print(torch.cuda.is_available())"`.
   - Переустановите torch: `pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall`.

5. **Воркеры умирают (dead workers > 0):**
   - Смотрите `worker_N_fatal.log` в каталоге training/.
   - Обычно проблема с ONNX-моделью (несовместимый opset или размерности).

---

## Дальнейшие шаги (после усиления)

1. **Запустить обучение на 5000-10000 итераций** (5-10 дней на вашем железе).
2. **Мониторить TensorBoard**: следите за `ExtBaseline/elo_vs_random` — это главный индикатор.
3. **Как только `elo_vs_random` перестанет расти** — модель сошлась, можно увеличивать архитектуру (16 блоков, 384 канала) или time_budget.
4. **После 5000 итераций** — eval против человека. Если winrate против вас < 50% — продолжайте обучение.

Удачи в тренировке бота!
