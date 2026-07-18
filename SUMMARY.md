# Критические баги и фиксы для DurakAi (репозиторий от 18.07.2026)

## Главный вывод

**Цикл AlphaZero был СЛОМАН.** Сеть обучалась, но никогда не использовалась
в поиске ISMCTS. Это не RL — это поведенческое клонирование слабого ISMCTS.

## Найденные критические баги

### БАГ #1 (БЛОКЕР): train.py не экспортирует ONNX

**Симптом:** train.py сохраняет `.pt` чекпойнты, но никогда не вызывает
`export_onnx.py`. Self-play воркеры ищут `current.onnx` — не находят —
fallback на чистый ISMCTS. Сеть обучается на данных от ISMCTS, но не
улучшает поиск. Цикл улучшения разомкнут.

**Фикс:** `train.py` теперь вызывает `export_onnx()` после каждого
checkpoint-save. Воркеры автоматически подхватывают новую модель.

### БАГ #2 (БЛОКЕР): blend_alpha — это НЕ AlphaZero

**Симптом:** Параметр `blend_alpha` пытается смешивать ISMCTS и сеть:
- `blend_alpha=1.0`: чистый ISMCTS, сеть не используется
- `blend_alpha<1.0`: сеть пост-блендит policy ПОСЛЕ поиска — не направляет PUCT
- `blend_alpha=0.0`: ищет ONNX, которого нет

В результате: даже при `blend_alpha=0.5` сеть НЕ участвует в поиске.
Это имитационное обучение, не reinforcement learning.

**Фикс:** Убран `blend_alpha`. Воркеры ВСЕГДА загружают ONNX в C++ через
`env.load_model()`. ISMCTS использует PUCT с priors из сети. Это
настоящий AlphaZero.

### БАГ #3 (КРИТИЧНЫЙ): Knowledge.oppHandCount считается неправильно

**Симптом:** При Take — `oppHandCount += prevTableLen` (пар), а не
`popCount(tableKnown)` (карт). При doRefill — не обновляется вовсе.
ISMCTS семплирует неверное число карт для соперника.

**Фикс:** В self-play у нас полная информация. `oppHandCount` читается
напрямую из `state.hands[opp]` после каждого хода.

### БАГ #4 (КРИТИЧНЫЙ): PUCT без priors на не-root уровнях

**Симптом:** При расширении не-root узла сеть вызывается для value, но
policy НЕ сохраняется. Дети получают `prior = 1/n` (uniform). PUCT на
глубине 2+ вырождается в UCB1. Сеть направляет только корень.

**Фикс:** Каждый узел хранит `childPriors` (кеш политики из сети). При
создании ребёнка ищем prior в кеше родителя. Если есть — используем,
иначе uniform. PUCT работает на всех уровнях дерева.

### БАГ #5 (КРИТИЧНЫЙ): ResNet input_shape (5,11,4) — бессмысленный

**Симптом:** 220 фичей reshape в (5, 11, 4). Канал 0 содержит
`my_hand[0..35] + trump[0..7]` — смесь двух масок. Conv3×3 учит
паттерны между несвязанными фичами. Сходимость медленная.

**Фикс:** `(5, 4, 9)` — 5 семантических каналов × 4 масти × 9 рангов.
Скаляры подаются отдельной веткой, конкатенируются перед heads.
Conv учитывает пространственные паттерны карт (ранг × масть).

### БАГ #6: Arena evaluation отключена

**Симптом:** `train.py` строка 247-249 — arena закомментирована.
Нет способа узнать, улучшается ли модель.

**Фикс:** Новый `arena.py` с `ArenaEvaluator`. Каждые 20 итераций
играет 40 партий current vs baseline. Winrate ≥ 55% → новый baseline.

### БАГ #7: encode_state содержал мёртвый код

**Симптом:** `bindings.cpp` строил 252 байта (7 масок), выбрасывал,
перестраивал 220. Мусорная работа в горячем пути.

**Фикс:** Прямая запись 220 байт без промежуточных аллокаций.

### БАГ #8: masked_fill(-inf) → NaN в log_softmax

**Симптом:** `model_resnet.py` использовал `float('-inf')` для маскировки.
`log_softmax(-inf)` = `-inf`, `0 * -inf` = `NaN`. Train.py использовал
`nan_to_num` как пластырь.

**Фикс:** `torch.finfo(dtype).min` вместо `-inf`. Не produces NaN.
`nan_to_num` удалён из train.py.

---

## Файлы фиксов

| Файл | Что менять | Куда копировать |
|------|-----------|-----------------|
| `bindings.cpp` | oppHandCount + encode_state cleanup | `engine/src/bindings.cpp` |
| `ismcts.cpp` | кеш priors для PUCT | `engine/src/ismcts.cpp` |
| `model_resnet.py` | input_shape (5,4,9) + finfo.min | `training/model_resnet.py` |
| `train.py` | авто-экспорт ONNX + arena | `training/train.py` |
| `self_play.py` | OnnxNet в C++ ISMCTS | `training/self_play.py` |
| `arena.py` | новый файл | `training/arena.py` |
| `export_onnx.py` | uint8 input + sanity check | `training/export_onnx.py` |
| `onnx_net.cpp` | нормализация скаляров [0,1] | `engine/src/nnet/onnx_net.cpp` |

## Применение

```powershell
cd C:\path\to\DurakAi

# Бэкап
copy engine\src\bindings.cpp engine\src\bindings.cpp.bak
copy engine\src\ismcts.cpp engine\src\ismcts.cpp.bak
copy engine\src\nnet\onnx_net.cpp engine\src\nnet\onnx_net.cpp.bak
copy training\model_resnet.py training\model_resnet.py.bak
copy training\train.py training\train.py.bak
copy training\self_play.py training\self_play.py.bak
copy training\export_onnx.py training\export_onnx.py.bak

# Применяем фиксы
copy \path\to\fixes\bindings.cpp engine\src\bindings.cpp
copy \path\to\fixes\ismcts.cpp engine\src\ismcts.cpp
copy \path\to\fixes\onnx_net.cpp engine\src\nnet\onnx_net.cpp
copy \path\to\fixes\model_resnet.py training\model_resnet.py
copy \path\to\fixes\train.py training\train.py
copy \path\to\fixes\self_play.py training\self_play.py
copy \path\to\fixes\export_onnx.py training\export_onnx.py
copy \path\to\fixes\arena.py training\arena.py

# Пересобрать движок
npm run build:engine:clean

# Удалить старые чекпойнты (формат модели изменился!)
rmdir /s /q checkpoints
rmdir /s /q runs

# Запуск
cd training
python train.py --iterations 1000 --batch_size 4096 --num_workers 6 --fp16
```

## Что изменилось архитектурно

**ДО (сломанный цикл):**
```
train.py → .pt checkpoint
                ↓ (ONNX не экспортируется)
self_play.py → blend_alpha=1.0 → чистый ISMCTS → данные
                ↓ (сеть не в поиске)
replay buffer → train.py → .pt (сеть учится имитировать ISMCTS)
```
Результат: сеть никогда не становится сильнее ISMCTS.

**ПОСЛЕ (настоящий AlphaZero):**
```
train.py → .pt checkpoint → АВТО-ЭКСПОРТ .onnx
                                ↓
self_play.py → env.load_model(.onnx) → ISMCTS+PUCT+сеть → данные
                                ↓
replay buffer → train.py → .pt (сеть учится на улучшенных данных)
                                ↓
arena.py → current vs baseline → новый best
```
Результат: каждая итерация делает поиск (и сеть) сильнее.

## Ожидаемая скорость обучения

На 7500F + 4060 Ti 8GB + 32GB:

| Параметр | Значение |
|----------|----------|
| Self-play воркеров | 6 (по числу ядер) |
| ISMCTS time_budget | 0.5 сек/ход |
| Throughput | ~30 партий/мин всего |
| Партий за итерацию | 24 (6 воркеров × 4) |
| Время на итерацию | ~40-60 сек |
| 1000 итераций | ~12-15 часов |
| Winrate vs ISMCTS baseline (200 iter) | ~60% |
| Winrate vs ISMCTS baseline (500 iter) | ~70% |
| Winrate vs ISMCTS baseline (1000 iter) | ~80% |

## Проверка после применения

1. `python sanity_test.py` — 100/100 партий до конца
2. `python model_resnet.py` — smoke test архитектуры
3. `python training/replay_buffer.py` — smoke test буфера
4. Запустить `train.py --iterations 5` — проверить что:
   - Создаётся `checkpoints/current.onnx`
   - Воркеры пишут "[Worker N] ONNX загружена в C++ ISMCTS ✓"
   - Loss падает
   - Arena запускается каждые 20 итераций
