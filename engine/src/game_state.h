#pragma once

#include "card.h"
#include "deck.h"
#include "move.h"

#include <vector>
#include <cstdint>

namespace durakk {

// Чей ход / на чьей стороне бот в текущем сигнале состояния.
enum class Side : uint8_t {
    Me   = 0,  // бот-советник
    Opp  = 1,  // соперник
};

// Фаза кона (отбоя). Определяет набор допустимых действий.
enum class Phase : uint8_t {
    Attack,   // атакующий кладёт карту / подкидывает
    Defense,  // защищающийся бьёт / переводит / берёт
    Done,     // кон завершён (бито или взял) — ждём добора и нового хода
};

// Полное состояние игры на момент запроса к движку.
//
// Замечание: движок — советник, а не полный игровой сервер. Входное состояние
// формируется UI вручную: мы знаем свою руку точно, руку соперника — только
// количеством, стол и козырь — видим полностью.
struct GameState {
    // --- Настройки стола ---
    DeckSize deckSize = DeckSize::Cards36;
    bool transferEnabled = true;   // переводной режим (по ТЗ включён)
    bool flashEnabled = false;     // «проездной» (опция стола, по умолчанию выкл.)
    int firstTrickLimit = 5;       // лимит пар в первом коне партии

    // --- Поле игры ---
    Deck deck;
    std::vector<Card> myHand;          // рука бота (точно)
    int oppHandCount = 0;              // сколько карт у соперника (точно не знаем какие)
    std::vector<TablePair> table;      // текущие пары на столе в порядке добавления

    // --- Кто что делает ---
    Side attacker = Side::Me;          // кто атакует в этом коне
    Side turn = Side::Me;              // чей сейчас ход
    Phase phase = Phase::Attack;       // текущая фаза

    // --- Состояния правил ---
    bool firstTrick = true;            // идёт ли первый кон партии (влияет на лимит)
    bool flashUsedThisTrick = false;   // использован ли «проездной» в этом коне

    // --- Утилиты ---
    // Сколько карт атакующего ранга на столе (атаки + защиты).
    int countRankOnTable(Rank r) const;

    // Сколько пар на столе (атака+защита).
    int pairsCount() const { return static_cast<int>(table.size()); }

    // Сколько атакующих карт на столе (равно pairsCount, т.к. пара = атака + мб защита).
    int attackCardsCount() const { return pairsCount(); }

    // Количество непобитых атакующих карт (для защиты/перевода).
    int undefendedAttacksCount() const;

    // Ранг верхней непобитой атаки — то, чем переводят.
    // Возвращает hasValue=false, если все побиты или стол пуст.
    bool topUndefendedAttackRank(Rank& out) const;

    // Рука «активной стороны»: своей мы знаем, чужой — нет (но можем спросить размер).
    Side opposite(Side s) const { return s == Side::Me ? Side::Opp : Side::Me; }
};

} // namespace durakk
