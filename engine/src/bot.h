#pragma once

#include "game_state.h"
#include "knowledge.h"
#include "match.h"
#include "move.h"
#include "nnet/policy_value_net.h"

#include <memory>

namespace durakk {

// Уровень силы
enum class Strength { Fast, Normal, Deep };

// Параметры силы, общие для ISMCTS и эндшпиля.
struct SearchSettings {
    Strength strength = Strength::Normal;
    int numThreads = 8;
};

// Статистика последнего поиска (для отображения в UI).
struct DecisionStats {
    std::string mode;       // "ISMCTS" | "Endgame" | "Fallback"
    long   playouts = 0;
    double winrate = 0.0;   // ISMCTS
    int    depthReached = 0;// эндшпиль
    bool   solved = false;  // эндшпиль: найден форсированный результат
    double timeMs = 0.0;
};

// Бот-советник: маршрутизирует между ISMCTS (фаза с колодой) и минимаксом
// (эндшпиль, когда колода пуста), используя Knowledge (абсолютную память).
class Bot {
public:
    Bot();

    // Полный путь: строит MatchState + Knowledge из GameState, запускает
    // соответствующий решатель, возвращает ход и статистику.
    // `strength` — таймаут/число потоков.
    Move decide(const GameState& s, const Knowledge& k,
                const SearchSettings& settings,
                DecisionStats* statsOut = nullptr);

    // Доступ к сети (чтобы main.cpp мог подменить RandomNet на OnnxNet позже).
    PolicyValueNet* net() { return net_.get(); }

private:
    std::unique_ptr<PolicyValueNet> net_;
};

} // namespace durakk
