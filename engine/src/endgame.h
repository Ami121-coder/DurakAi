#pragma once

#include "match.h"
#include "move.h"
#include "knowledge.h"
#include "ttable.h"

#include <atomic>
#include <chrono>
#include <vector>

namespace durakk {

// Уровень силы эндшпильного поиска (таймаут в секундах).
struct EndgameLimits {
    double timeBudgetSec = 2.0;   // мягкий лимит; поиск прерывается по таймеру
    int    maxDepth      = 60;    // защита от бесконечных циклов
};

// Результат эндшпильного поиска.
struct EndgameResult {
    Move   move;
    bool   solved = false;        // true, если найден форсированный результат (мат/ничья)
    int    score = 0;             // +1 = наш форсированный выигрыш, -1 = проигрышь, 0 = ничья/не решено
    int    depthReached = 0;
    long   nodes = 0;             // число рассмотренных узлов
    double timeMs = 0.0;
};

// Найти лучший ход в эндшпиле (колода пуста, perfect information).
// viewpoint — за кого максимизируем (0=бот, 1=соперник); обычно 0.
EndgameResult bestEndgameMove(MatchState s, Player viewpoint,
                              const EndgameLimits& lim,
                              std::atomic<bool>* stopFlag = nullptr);

// Task 5: Sampled Endgame — для случая когда в колоде осталось мало карт,
// но не 0. Сэмплируем N детерминизаций руки оппонента (по байесовской
// матрице Knowledge.oppProbs), для каждой запускаем точный endgame α-β,
// усредняем winrate по всем сэмплам.
//
// В отличие от ISMCTS, это даёт ТОЧНЫЙ результат для каждой детерминизации
// (а не rollout-оценку). Когда deck <= 6, ISMCTS тратит бюджет на ветки
// с высокой вариативностью из-за случайного добора, тогда как sampled
// minimax за то же время находит более надёжное решение.
//
// Параметры:
//   s        — текущее состояние (колода НЕ пуста).
//   k        — Knowledge с oppProbs.
//   viewpoint — за кого максимизируем.
//   lim      — бюджет времени.
//   nSamples — число детерминизаций (адаптивно: budget / time_per_sample).
//   stopFlag — флаг прерывания.
EndgameResult bestSampledEndgameMove(const MatchState& s, const Knowledge& k,
                                     Player viewpoint,
                                     const EndgameLimits& lim,
                                     int nSamples,
                                     std::atomic<bool>* stopFlag = nullptr);

} // namespace durakk
