#include "game_state.h"

namespace durakk {

int GameState::countRankOnTable(Rank r) const {
    int n = 0;
    for (const auto& p : table) {
        if (p.attack.rank == r) ++n;
        if (p.defended && p.defense.rank == r) ++n;
    }
    return n;
}

int GameState::undefendedAttacksCount() const {
    int n = 0;
    for (const auto& p : table) {
        if (!p.defended) ++n;
    }
    return n;
}

bool GameState::topUndefendedAttackRank(Rank& out) const {
    // Идём с конца: последняя добавленная непобитая атака — «верхняя».
    for (auto it = table.rbegin(); it != table.rend(); ++it) {
        if (!it->defended) {
            out = it->attack.rank;
            return true;
        }
    }
    return false;
}

} // namespace durakk
