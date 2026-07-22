#pragma once
#include "card.h"
#include <vector>
#include <string>

enum class Phase { ATTACK, DEFEND };

struct TablePair {
    Card attack;
    Card defense;  // rank = -1 если не побита
};

struct GameState {
    // Руки
    std::vector<Card> hands[2];  // [0] = бот, [1] = соперник
    int oppCardCount = 0;        // если рука соперника неизвестна

    // Стол
    std::vector<TablePair> table;

    // Колода
    int deckCount = 0;
    int trumpSuit = 0;  // 0=clubs, 1=diamonds, 2=hearts, 3=spades

    // Фаза
    Phase phase = Phase::ATTACK;
    int attacker = 0;   // 0 или 1

    // Параметры правил
    int pairsLimit = 6;     // 6 обычно, 5 для первого кона
    bool firstTrick = false; // первый кон партии
    bool transferEnabled = true;
    bool flashEnabled = false;
    bool flashUsedThisTrick = false;

    // Сброс (бито) — для Bayesian
    std::vector<Card> discard;
};
