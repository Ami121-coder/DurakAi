#pragma once

#include "match.h"
#include "move.h"
#include "ttable.h"

#include <atomic>
#include <chrono>

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
    int    score = 0;             // +1 = наш форсированный выигрыш, -1 = проигрыш, 0 = ничья/не решено
    int    depthReached = 0;
    long   nodes = 0;             // число рассмотренных узлов
    double timeMs = 0.0;
};

// Найти лучший ход в эндшпиле (колода пуста, perfect information).
// viewpoint — за кого максимизируем (0=бот, 1=соперник); обычно 0.
EndgameResult bestEndgameMove(MatchState s, Player viewpoint,
                              const EndgameLimits& lim,
                              std::atomic<bool>* stopFlag = nullptr);

} // namespace durakk
