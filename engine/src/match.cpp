#include "match.h"
#include "rules.h" // canBeat

#include <algorithm>

namespace durakk {

// ============================ Утилиты состояния ============================

int MatchState::undefendedCount() const {
    int n = 0;
    for (int i = 0; i < tableLen; ++i)
        if (!table[i].defended) ++n;
    return n;
}

bool MatchState::topUndefendedAttackRank(Rank& out) const {
    for (int i = tableLen - 1; i >= 0; --i) {
        if (!table[i].defended) { out = table[i].attack.rank; return true; }
    }
    return false;
}

int MatchState::countRankOnTable(Rank r) const {
    int n = 0;
    for (int i = 0; i < tableLen; ++i) {
        if (table[i].attack.rank == r) ++n;
        if (table[i].defended && table[i].defense.rank == r) ++n;
    }
    return n;
}

int MatchState::pairsHeadroom() const {
    // FIX #7, #18: используем pairsLimit (пробрасывается из GameState.firstTrickLimit
    // для первого кона через toMatchState) вместо хардкода 5/6.
    int byRules = firstTrick ? std::min(pairsLimit, 5) : pairsLimit;
    int defHand = popCount(hands[toIdx(defender())]);
    return std::min(byRules, defHand) - tableLen;
}

bool MatchState::isGameOver() const {
    if (phase == MatchPhase::GameOver) return true;
    if (deckRemaining > 0) return false; // пока есть колода — играем
    return handSize(hands[0]) == 0 || handSize(hands[1]) == 0;
}

int MatchState::winner() const {
    if (!isGameOver()) return -1;
    bool me0 = handSize(hands[0]) == 0;
    bool op0 = handSize(hands[1]) == 0;
    if (me0 && op0) return -1; // ничья (теоретически)
    if (me0) return 0;
    if (op0) return 1;
    return -1;
}

void MatchState::clearTableToDiscard() {
    for (int i = 0; i < tableLen; ++i) {
        discard |= cardBit(table[i].attack);
        if (table[i].defended) discard |= cardBit(table[i].defense);
    }
    tableLen = 0;
    for (auto& p : table) p = Pair{};
}

void MatchState::takeTableToHand(Player p) {
    CardMask& h = hands[toIdx(p)];
    for (int i = 0; i < tableLen; ++i) {
        h |= cardBit(table[i].attack);
        if (table[i].defended) h |= cardBit(table[i].defense);
    }
    tableLen = 0;
    for (auto& p2 : table) p2 = Pair{};
}

// ============================ applyMove ============================
//
// Соглашение о доборе (по правилам, см. docs/rules.md §8):
//   • После «бито» или «взял» оба добирают до 6 из колоды.
//   • Порядок: первым — тот, кто АТАКОВАЛ в этом коне; затем — кто защищался.
//   • При переводе первым добирает тот, кто совершил последнюю атаку
//     (чей перевод был отбит). В нашей модели ролей это эквивалентно:
//     первым добирает БЫВШИЙ атакующий = нынешний защищающийся.
//     Чтобы не вести отдельную память, мы передаём порядок в refill явно.

bool applyMove(MatchState& s, const Move& m) {
    if (s.phase == MatchPhase::GameOver) return false;

    Player me = s.turn;

    auto doRefill = [&s](Player firstDrawer, Player secondDrawer) {
        if (s.deckRemaining <= 0) return;
        auto draw = [&](Player p) {
            if (s.deckRemaining <= 0 || s.deck == 0) return;
            CardMask bit = s.deck & (~s.deck + 1); // младший установленный бит
            s.deck ^= bit;
            s.hands[toIdx(p)] |= bit;
            --s.deckRemaining;
        };
        while (s.deckRemaining > 0 && handSize(s.hands[toIdx(firstDrawer)]) < 6) draw(firstDrawer);
        while (s.deckRemaining > 0 && handSize(s.hands[toIdx(secondDrawer)]) < 6) draw(secondDrawer);
    };

    switch (m.action) {

    // ---------- Атака / подкидывание / вдогонку ----------
    case Action::Attack:
    case Action::Toss: {
        // Подкидывание «вдогонку» разрешено в фазе Pursue (после «беру»),
        // обычное подкидывание — в фазе Attack.
        if (s.phase != MatchPhase::Attack && s.phase != MatchPhase::Pursue) return false;
        if (s.turn != s.attacker) return false;
        if (!maskHas(s.hands[toIdx(me)], m.card)) return false;

        if (s.tableLen > 0) {
            if (s.countRankOnTable(m.card.rank) == 0) return false;
            // Лимит по руке защитника (будущего взявшего в Pursue).
            int defHand = popCount(s.hands[toIdx(s.defender())]);
            int byRules = s.firstTrick ? 5 : s.pairsLimit;
            int maxPairs = std::min(byRules, defHand);
            if (s.tableLen >= maxPairs) return false;
        }

        s.hands[toIdx(me)] = maskRemove(s.hands[toIdx(me)], m.card);
        s.table[s.tableLen++] = Pair{ m.card, Card{}, false };

        if (s.phase == MatchPhase::Pursue) {
            // В Pursue фаза не меняется — атакующий может подкидывать дальше
            // или завершить через Pass.
            s.turn = s.attacker;
        } else {
            s.turn = s.defender();
            s.phase = MatchPhase::Defense;
        }
        return true;
    }

    // ---------- Защита ----------
    case Action::Defend: {
        if (s.phase != MatchPhase::Defense) return false;
        if (s.turn != s.defender()) return false;
        if (!maskHas(s.hands[toIdx(me)], m.card)) return false;
        if (s.undefendedCount() == 0) return false;

        int idx = -1;
        for (int i = 0; i < s.tableLen; ++i) {
            if (s.table[i].defended) continue;
            if (m.hasTarget && s.table[i].attack != m.target) continue;
            if (canBeat(s.table[i].attack, m.card, s.trump)) { idx = i; break; }
        }
        if (idx < 0) return false;

        s.hands[toIdx(me)] = maskRemove(s.hands[toIdx(me)], m.card);
        s.table[idx].defense = m.card;
        s.table[idx].defended = true;

        if (s.undefendedCount() == 0) {
            s.turn = s.attacker;
            s.phase = MatchPhase::Attack;
        }
        return true;
    }

    // ---------- Перевод ----------
    case Action::Transfer: {
        if (!s.transferEnabled) return false;
        if (s.phase != MatchPhase::Defense) return false;
        if (s.turn != s.defender()) return false;
        if (!maskHas(s.hands[toIdx(me)], m.card)) return false;

        Rank topRank;
        if (!s.topUndefendedAttackRank(topRank)) return false;
        if (m.card.rank != topRank) return false;

        int newDefHand = popCount(s.hands[toIdx(s.attacker)]);
        int cardsAfter = s.undefendedCount() + 1;
        if (newDefHand < cardsAfter) return false;
        // FIX #8: лимит пар по руке НОВОГО защитника (бывшего атакующего).
        int byRules = s.firstTrick ? std::min(s.pairsLimit, 5) : s.pairsLimit;
        int maxPairs = std::min(byRules, newDefHand);
        if (s.tableLen >= maxPairs) return false;

        s.hands[toIdx(me)] = maskRemove(s.hands[toIdx(me)], m.card);
        s.table[s.tableLen++] = Pair{ m.card, Card{}, false };

        // Смена ролей: бывший защищающийся (me) — теперь атакующий.
        s.attacker = me;
        // После перевода отбивается бывший атакующий → он становится защищающимся.
        s.turn = s.defender();
        s.phase = MatchPhase::Defense;
        return true;
    }

    // ---------- Взять ----------
    case Action::Take: {
        if (s.phase != MatchPhase::Defense) return false;
        if (s.turn != s.defender()) return false;
        if (s.tableLen == 0) return false;

        // FIX #5: правильная последовательность «Беру»:
        //   1. Защитник объявляет «беру», но НЕ забирает карты сразу.
        //   2. Фаза переходит в Pursue — атакующий может подкинуть «вдогонку»
        //      (ранги со стола, лимит по руке защитника).
        //   3. Завершение кона происходит через Action::Pass из фазы Pursue.
        // Раньше стол сразу уходил в руку защитника, что нарушало правило
        // о подкидывании «вдогонку» (см. docs/rules.md §7).
        s.phase = MatchPhase::Pursue;
        s.turn = s.attacker;  // ход переходит к атакующему для подкидывания
        return true;
    }

    // ---------- Бито / Пас ----------
    case Action::Done:
    case Action::Pass: {
        if (s.tableLen == 0) return false; // нечего завершать

        // В фазе Pursue (после «беру») Pass/Done завершает кон со взятием.
        if (s.phase == MatchPhase::Pursue) {
            // Take-сценарий: защитник забирает весь стол.
            Player taker = s.defender();
            Player attackSide = s.attacker;
            s.takeTableToHand(taker);
            s.firstTrick = false;
            s.phase = MatchPhase::Attack;
            s.turn = attackSide;
            s.attacker = attackSide;
            // Добор: первым берёт атакующий, затем — взявший.
            doRefill(attackSide, taker);
            if (s.isGameOver()) s.phase = MatchPhase::GameOver;
            return true;
        }

        // Обычный «бито»: только если все атаки побиты.
        if (m.action == Action::Done && s.undefendedCount() > 0) return false;
        // Pass без «бито» — только если есть побитые пары и нет непобитых атак.
        if (m.action == Action::Pass && s.undefendedCount() > 0) return false;

        Player exAttacker = s.attacker;     // кто атаковал в этом коне (добирает первым)
        Player exDefender = s.defender();   // кто защищался
        Player nextAttacker = exDefender;   // после «бито» ход переходит к защищавшемуся

        s.clearTableToDiscard();
        s.firstTrick = false;

        // Добор: первым — экс-атакующий, затем — экс-защитник.
        doRefill(exAttacker, exDefender);

        s.attacker = nextAttacker;
        s.turn = nextAttacker;
        s.phase = MatchPhase::Attack;

        if (s.isGameOver()) s.phase = MatchPhase::GameOver;
        return true;
    }

    } // switch

    return false;
}

} // namespace durakk
