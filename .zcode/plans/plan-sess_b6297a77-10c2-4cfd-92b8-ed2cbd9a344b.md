# План: Вариант 4 (Гибрид) + 5 математических усилений, поэтапно

## Контекст и архитектурные принципы

**Цель:** чисто математический бот (без ML), гибрид ISMCTS + эндшпиль-minimax (уже есть в `bot.cpp`/`endgame.cpp`), усиленный 5 методами. Каждый этап — самостоятельный вин с arena-замером.

**Принципы:**
- Не ломать существующий гибрид (ISMCTS при колоде / minimax в эндшпиле).
- Каждый этап компилируется и работает независимо.
- Нейросеть остаётся **опциональной** (грузится если есть ONNX, иначе чистая математика).
- Всё на CPU — GPU простаивает (это нормально для математики).
- После каждого этапа — arena-замер (40–100 партий vs baseline). Winrate ≥ 52% → улучшение реально, оставляем.

---

## Этап 0: Подготовка инфраструктуры (1 день)

Прежде чем улучшать, надо иметь способ **измерять силу** и **корректный baseline-режим**.

### 0.1. Режим «без сети» по умолчанию в `bot.cpp`
**Проблема:** сейчас `Bot()` всегда грузит ONNX, при неудаче — `RandomNet`. Но `RandomNet` даёт uniform policy + value≈0.5, что **деградирует PUCT** (uniform priors = бесполезно, value=0.5 = бесполезно).

**Фикс:** в `Bot::decide` передавать `net_.get()` только если сеть реально загружена (OnnxNet с готовой моделью). Иначе `nullptr` → ISMCTS использует UCB1 + rollout (правильный математический путь).

- Файл: `engine/src/bot.cpp:57-71` (конструктор), `bot.cpp:112`.
- Добавить флаг `bool hasRealNet_` — true только при успешной загрузке ONNX.
- В `bot.cpp:112`: `PolicyValueNet* net = hasRealNet_ ? net_.get() : nullptr;`

### 0.2. Фикс эндшпиль-minimax (корректная рука соперника)
**Проблема:** `bot.cpp:82` ставит `oppHand = k.unknownPool()` — все неизвестные карты (может быть 20 при реальных 3). Минимакс играет с нереалистичной рукой.

**Фикс:** ограничить до `oppHandCount` карт. Пока без belief — равновероятный sample (как ISMCTS `determine`), либо (когда неизвестных карт мало, ≤ oppHandCount) — все они у соперника.

- Файл: `engine/src/bot.cpp:79-89`.
- Использовать `sampleSubset(k.unknownPool(), k.oppHandCount)` (нужна экспортированная функция или инлайн). При `deckRemaining==0` неизвестные карты = точно у соперника + оппонент, делим по oppHandCount.

### 0.3. C++ arena-утилита для замеров
Добавить отдельный режим в движок (новый `--arena` CLI-аргумент или команда протокола) для bot-vs-bot на N партий.

- Простой вариант: Python-скрипт через `durakk_env` (без ONNX) играет N партий математического бота vs математического бота с разными настройками.
- Минимально: зафиксировать **baseline-силу** текущего гибрида (vs RandomNet-ISMCTS из `ExternalBaselineEvaluator`), записать в `BENCHMARK.md`.

**Результат этапа 0:** работающий «чисто математический» гибрид + измеренный baseline.

---

## Этап 1: Smart Playout (🥈 #2) — 1–2 дня

Усилить `defaultPolicy` в `rules_fast.cpp:198`. Сейчас: 70% дешёвый ход / 30% random. Не учитывает ключевые факторы.

**Что добавить в `defaultPolicy`:**

1. **Защита — правильный выбор карты для побития:**
   - Бей младшей достаточной некозырной.
   - Козырь — только если атаки ≥2 (не тратить на 1 карту) или если иначе Take теряет много.
   - Если у соперника мало карт и можно «пережить» кон Take'ом — предпочитай Take вместо сливания козыря.

2. **Атака — умное подкидывание:**
   - Не подкидывай козырные, если соперник их побьёт козырем (равноценный обмен в его пользу).
   - Подкидывай парные ранги (есть вторая такая же в руке) — сломает защиту/создаст перевод.
   - Атакуй старшими некозырными, если у соперника мало козырей по belief (но belief — этап 5; пока без него).

3. **Перевод (transfer):**
   - Выгоден, если у соперника `oppHandCount < pairsHeadroom` (он не сможет отбить).
   - Не переводи козырным, если оставишь себя без козырей.

4. **Take vs Defend — ситуативно:**
   - Если на столе ≤2 атаки и есть дешёвая защита — защищайся.
   - Если атак ≥4 и защита дорогая — бери (меньше потерь, чем слив карт).

**Реализация:** переписать `defaultPolicy` с системой весов `moveWeight(s, m, k)` → softmax-выбор с температурой. Учесть `s.pairsHeadroom()`, `s.undefendedCount()`, `s.deckRemaining` (близость эндшпиля).

- Файл: `engine/src/rules_fast.cpp:187-229` (`cardWeight`, `defaultPolicy`).

**Замер:** arena vs baseline. Ожидаемый прирост: +3–6% winrate.

---

## Этап 2: Hand-crafted evaluation function (🥉 #3) — 2 дня

Новая функция статической оценки позиции (без rollout). Используется как «лёгкий лист» в ISMCTS и в quiescence.

**Факторы оценки (с точки зрения viewpoint):**

| Фактор | Вес | Логика |
|---|---|---|
| Размер моей руки vs руки соперника | высокий | меньше карт у меня = лучше |
| Количество козырей у меня | высокий | козыри = сила, особенно в эндшпиле |
| Козыри у соперника (из belief/unknown) | средний | больше = хуже |
| Парные карты у меня (для перевода) | средний | 2× один ранг = гибкость |
| «Мёртвые» карты (старшие некозырные без прикрытия) | средний | застрянут в конце |
| `deckRemaining` (близость эндшпиля) | контекст | в эндшпиле козыри ценнее |
| Незащищённые атаки соперника на столе | высокий | rival take'нет = хорошо для меня |

**Архитектура:**
- Новый файл `engine/src/eval.h` + `eval.cpp` с `float evaluatePosition(const MatchState& s, const Knowledge& k, Player viewpoint)`.
- Возвращает значение в [-1, 1] (или [0, 1]) — совместимое с ISMCTS value-масштабом.
- Используется в ISMCTS: при `maxRolloutDepth > 0` — rollout N ходов через smart playout, затем `evaluatePosition` (вместо rollout до конца).
- Это снижает дисперсию и ускоряет оценку листа.

- Файлы: новый `engine/src/eval.{h,cpp}`, изменения в `engine/src/ismcts.cpp` (лист-оценка), `engine/CMakeLists.txt`.

**Замер:** arena vs Этап-1. Ожидаемый прирост: +4–8% winrate.

---

## Этап 3: Кеш детерминизаций + TT для ISMCTS (#5) — 2 дня

**Проблема:** сейчас каждая playout заново делает `determine()` (семплирует руку соперника) и строит дерево с нуля. Между детерминизациями с похожим распределением карты совпадают — можно кешировать.

**Что сделать:**

1. **Transposition table для ISMCTS-узлов:**
   - Хеш позиции (Zobrist, уже есть в `ttable.cpp` для эндшпиля) → оценка + visits.
   - При заходе в узел проверяем TT: если есть запись с достаточной visits — используем агрегированную оценку вместо нового rollout.
   - Осторожно: детерминизации разные, поэтому TT хранит **агрегат по всем детерминизациям**, не конкретную.

2. **Кеш детерминизаций:**
   - Для одного `decide`-запроса кешировать `oppHand`-сэмплы (10–50 штук) и переиспользовать между плейаутами.
   - Снижает стоимость `determine()`.

3. **Осторожность с mutex:** текущий tree-parallel с mutex (`ismcts.cpp` `Tree::mtx`) + virtual loss. TT добавит ещё contention. Профилировать; если лок становится bottleneck → переключиться на root-parallelisation (этап 4/5).

- Файлы: `engine/src/ismcts.cpp` (структура Tree, функция worker), `engine/src/ttable.h` (обобщить для ISMCTS).

**Замер:** сравнить не только winrate, но и **playouts/sec** (должно вырасти на 30–50%). Winrate может слегка упасть из-за кеш-шумов — балансировать размером кеша.

---

## Этап 4: Selective Deepening / Quiescence (#4) — 2 дня

**Идея:** в «нестабильных» позициях (идёт активный кон, возможен перевод/подкидывание) расширять поиск глубже; в спокойных — мельче.

**Эвристики «нестабильности»:**
- На столе есть непобитые атаки (`undefendedCount() > 0`).
- Возможен перевод (`transferEnabled && topUndefendedAttackRank`).
- Защищающийся под угрозой Take (мало карт в руке, много атак).
- `pairsHeadroom() > 0` и атакующий может подкинуть.

**Реализация:**
- В ISMCTS при расширении узла: если позиция нестабильна → не обрезать rollout, а продолжать expand до стабилизации (quiescence).
- В rollout (smart playout): в нестабильных позициях играть до конца кона, не до конца партии, затем `evaluatePosition`.
- Лимит глубины quiescence (например, +6 ходов) против бесконечных подкидываний.

- Файлы: `engine/src/ismcts.cpp` (логика select/expand), `engine/src/rules_fast.cpp` (rollout с quiescence).

**Замер:** arena vs Этап-3. Ожидаемый прирост: +2–4% winrate (особенно в миттельшпиле).

---

## Этап 5: Belief State Tracking (🥇 #1) — 3–4 дня

**Самое ценное усиление, но самое сложное** (требует расширения JSON-протокола и UI).

### 5.1. Проблема: движок stateless
Движок (`main.cpp`) не помнит историю. Для belief tracking по ходам соперника нужна последовательность наблюдений.

### 5.2. Расширение JSON-протокола
Добавить опциональное поле `oppMoves` в `setState` — массив ходов соперника, которые UI наблюдал с начала партии:
```json
{"cmd":"setState", ..., "oppMoves":[{"action":"defend","rank":"8","suit":"clubs","target":{...}}, ...]}
```
- Файл: `engine/src/protocol.cpp` (`parseState`), `engine/src/protocol.h`, `src/renderer.js` (сбор истории), `electron/engine-bridge.js`.

### 5.3. Класс `BeliefState`
Новый файл `engine/src/belief.h` + `belief.cpp`:
```cpp
class BeliefState {
    float prob_[36] = {0};  // вероятность, что карта у соперника
public:
    void initFromKnowledge(const Knowledge& k);  // uniform по unknownPool
    void updateFromOppMove(Move m, const MatchState& s);  // Bayes-обновление
    float prob(int cardIdx) const;
    void normalize();
};
```

**Bayes-обновления (ключевые правила):**
- Соперник **защитил** карту X → он имел X (нулевое обновление, и так в tableKnown).
- Соперник **не защитил** (Take) атаку, которую мог побить козырем → снижаем prob его козырей.
- Соперник **перевёл** рангом X → у него была карта ранга X (видим в tableKnown).
- Соперник **подкинул** X → у него была X.
- После `deckRemaining == 0`: все неизвестные = точно у соперника → prob=1 для unknownPool.
- Нормализация: сумма prob по unknownPool = oppHandCount.

### 5.4. Weighted sampling в ISMCTS
- Новая функция `sampleSubsetWeighted(CardMask pool, int k, const BeliefState& belief)` — взвешенный Fisher-Yates.
- В `determine()` (ismcts.cpp:75-88) использовать weighted sampling вместо uniform.
- В эндшпиль-minimax: вместо равновероятного sample использовать top-oppHandCount карт по belief (если неизвестных больше, чем нужно).

- Файлы: новый `engine/src/belief.{h,cpp}`, `engine/src/ismcts.cpp` (`determine`, `sampleSubset`), `engine/src/bot.cpp` (передача belief в ISMCTS и minimax), `engine/src/bindings.cpp` (self-play тоже использует belief для консистентности данных).

### 5.5. UI-интеграция
- `src/renderer.js` — собирать массив `oppMoves` из наблюдаемых ходов, передавать в `setState`.
- Для backward compatibility: если `oppMoves` отсутствует → uniform (текущее поведение).

**Замер:** arena vs Этап-4. Ожидаемый прирост: **+10–20% winrate** (самое большое усиление).

---

## Финальная проверка и итоговый замер

- Полный arena: усиленный гибрид vs baseline (Этап 0), 200–500 партий.
- Запись результатов в `BENCHMARK.md`: winrate, playouts/sec, среднее время хода по этапам.
- Ожидаемый **итоговый потолок**: сильный любитель / топ-15% онлайн-платформ.

---

## Сводка файлов по этапам

| Этап | Новые файлы | Изменяемые файлы |
|---|---|---|
| 0 | — | `bot.cpp`, `engine/src/bot.cpp`, `BENCHMARK.md` (новый) |
| 1 | — | `engine/src/rules_fast.cpp` |
| 2 | `engine/src/eval.{h,cpp}` | `engine/src/ismcts.cpp`, `engine/CMakeLists.txt` |
| 3 | — | `engine/src/ismcts.cpp`, `engine/src/ttable.h` |
| 4 | — | `engine/src/ismcts.cpp`, `engine/src/rules_fast.cpp` |
| 5 | `engine/src/belief.{h,cpp}` | `engine/src/ismcts.cpp`, `engine/src/bot.cpp`, `engine/src/protocol.{h,cpp}`, `engine/src/bindings.cpp`, `src/renderer.js`, `electron/engine-bridge.js` |

## Пересборка после каждого этапа
```bash
npm run build:engine:clean
```
После изменений в C++ движке — обязательная пересборка. Python (`training/`) не трогается (кроме опционального arena для замеров).

## Риски и mitigation
- **Mutex contention** на этапе 3 (TT + tree-parallel) → профилировать, при необходимости перейти на root-parallelisation.
- **Belief noise** на этапе 5 → conservative updates, нормализация, fallback на uniform при аномалиях.
- **Arena-дисперсия** (дурак = неполная инфа) → минимум 100 партий на замер, доверительные интервалы.
- **Совместимость с ONNX**: если сеть позже загрузится — она переопределяет `nullptr` в bot.cpp, всё работает.

## Что НЕ входит (по вашему выбору «только математика»)
- Лёгкая NN-оценка (A1), root-parallelisation как отдельное усиление, opening book, endgame tablebase. Их можно добавить позже отдельными этапами.