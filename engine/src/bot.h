#pragma once
#include "game_state.h"
#include <vector>

struct Move {
    Card card;
    enum MoveType { ATTACK, DEFEND, TAKE, BITO, TRANSFER } type = ATTACK;
};

class Bot {
public:
    Move decide(const GameState& state);
};
