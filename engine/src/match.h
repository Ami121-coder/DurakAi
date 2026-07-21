#pragma once

#include "bitboard.h"
#include "card.h"
#include "move.h"

#include <cstdint>
#include <array>

namespace durakk {

// Сторона в симуляции: 0 = бот (мы), 1 = соперник.
// В playout ИИ обе руки полностью известны — это детерминизация.
enum class Player : uint8_t { Me = 0, Opp = 1 };

inline Player other(Player p) { return p == Player::Me ? Player::Opp : Player::Me; }
inline int   toIdx(Player p)  { return static_cast<int>(p); }

// Фаза кона в симуляторе (упрощённая, для машины состояний applyMove).
enum class MatchPhase : uint8_t {
    Attack,   // атакующий кладёт/подкидывает
    Defense,  // защищающийся бьёт/переводит/берёт
    Pursue,   // защитник сказал «беру» — атакующий может докинуть «вдогонку»
              // (рангов со стола, в рамках лимита по руке будущего взявшего)
              // перед тем как карты уйдут в руку защитнику
    Done,     // кон завершён, состояние валидно для перехода к новому кону
    GameOver, // партия закончена
};

// Пара на столе: атакующая карта и (опц.) побившая её защита.
struct Pair {
    Card attack;
    Card defense;       // имеет смысл только если defended=true
    bool defended = false;
};

// Единый движок-симулятор партии. Используется:
//   • UI (через toMatchState из GameState) — для валидации и предсказаний;
//   • ISMCTS — детерминизация и rollout до конца;
//   • минимаксом эндшпиля — перебор ходов;
//   • (позже) pybind11 — для self-play нейросети.
//
// Инвариант: hands[i] и deck — непересекающиеся маски; discard накапливается.
struct MatchState {
    CardMask hands[2] = { 0, 0 };   // [0]=Me, [1]=Opp
    CardMask deck = 0;              // в симуляции порядок колоды не важен, только множество
    CardMask discard = 0;
    Suit trump = Suit::Spades;

    std::array<Pair, 6> table;      // текущие пары на столе
    int tableLen = 0;

    int pairsLimit = 6;             // макс пар в коне (5 в первом, иначе 6)
    Player attacker = Player::Me;   // кто атакует
    Player turn     = Player::Me;   // чей активный ход
    MatchPhase phase = MatchPhase::Attack;

    bool firstTrick = true;
    bool transferEnabled = true;
    bool flashEnabled = false;

    int deckRemaining = 0;          // счётчик (для правил добора); может быть > popCount(deck)
                                    // если колода «абстрактна» (порядок скрыт)

    // ---------- Утилиты ----------

    Player defender() const { return other(attacker); }

    // Сколько непобитых атак сейчас на столе.
    int undefendedCount() const;

    // Ранг верхней непобитой атаки (для перевода). false если все побиты / стол пуст.
    bool topUndefendedAttackRank(Rank& out) const;

    // Сколько карт ранга r задействовано на столе (атаки + защиты).
    int countRankOnTable(Rank r) const;

    // Хедрум: сколько ещё пар можно добавить (по лимиту и по руке защищающегося).
    int pairsHeadroom() const;

    // Состояние партии завершено?
    bool isGameOver() const;

    // Кто выиграл: 0/1/-1 (не закончено/ничья). Если isGameOver и у кого-то 0 карт.
    int winner() const;

    // Кладёт все карты стола в discard и очищает стол.
    void clearTableToDiscard();
    // Кладёт все карты стола в руку игрока и очищает стол.
    void takeTableToHand(Player p);
};

// Применить ход к состоянию (мутация). Возвращает false, если ход недопустим
// (тогда состояние не меняется). Реализует ВСЕ переходы:
//   Attack/Toss → добавить пару; защита/перевод — depending on phase.
//   Defend → побить конкретную непобитую атаку.
//   Transfer → положить карту того же ранга, сменить атакующего/защищающегося.
//   Take → забрать стол, ход остаётся атакующему (фаза Attack для вдогонку).
//   Done → «бито» (если все побито) или «больше не подкидываю».
// applyMove автоматически проводит конец кона, добор и переход хода, когда
// ход Done/PASS завершает кон.
bool applyMove(MatchState& s, const Move& m);

} // namespace durakk
