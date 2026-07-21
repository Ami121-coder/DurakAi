#include "rules.h"

#include <algorithm>

namespace durakk {

// ============================ Базовые правила ============================

int maxPairsThisTrick(const GameState& s, Side defender) {
    // Лимит по правилам: 6 пар (или 5 в первом коне партии).
    int byRules = s.firstTrick ? s.firstTrickLimit : 6;
    // ...но не больше, чем карт на руке защищающегося.
    int defenderHand = (defender == Side::Me)
        ? static_cast<int>(s.myHand.size())
        : s.oppHandCount;
    return std::min(byRules, defenderHand);
}

bool canBeat(Card attacker, Card defender, Suit trump) {
    return beats(attacker, defender, trump);
}

// Проверка, есть ли карта в руке (без учёта дублей — нужен точный экземпляр).
static bool handHas(const std::vector<Card>& hand, Card c) {
    return std::find(hand.begin(), hand.end(), c) != hand.end();
}

// Сколько ещё пар можно добавить на стол, не превысив лимит.
static int pairsHeadroom(const GameState& s) {
    Side defender = s.opposite(s.attacker);
    return maxPairsThisTrick(s, defender) - s.pairsCount();
}

// ============================ Атака ============================

ValidationResult validateAttack(const GameState& s, const Move& m) {
    if (s.phase != Phase::Attack)
        return {false, "сейчас не фаза атаки"};
    if (s.turn != s.attacker)
        return {false, "ходит не атакующий"};
    if (!handHas(s.myHand, m.card))
        return {false, "карты нет в руке"};

    // Если стол пуст — атаковать можно любой картой (это первая карта кона).
    if (s.table.empty()) {
        return {true, ""};
    }

    // Иначе атакующий может лишь «подкинуть»: ранг должен быть уже на столе.
    if (s.countRankOnTable(m.card.rank) == 0)
        return {false, "подкидывать можно только ранги, уже есть на столе"};

    if (pairsHeadroom(s) <= 0)
        return {false, "превышен лимит пар в коне"};

    return {true, ""};
}

// ============================ Защита ============================

ValidationResult validateDefend(const GameState& s, const Move& m) {
    if (s.phase != Phase::Defense)
        return {false, "сейчас не фаза защиты"};
    if (s.turn != s.opposite(s.attacker))
        return {false, "ходит не защищающийся"};
    if (!handHas(s.myHand, m.card))
        return {false, "карты нет в руке"};

    // Должна быть непобитая атака, которую бьём.
    if (s.undefendedAttacksCount() == 0)
        return {false, "неотчего защищаться (все атаки побиты)"};

    // target указан — проверим конкретную карту; иначе любая непобитая подойдёт.
    auto beatsTarget = [&](Card attackCard) {
        if (m.hasTarget && m.target != attackCard) return false;
        return canBeat(attackCard, m.card, s.deck.trump);
    };

    for (const auto& p : s.table) {
        if (!p.defended && beatsTarget(p.attack))
            return {true, ""};
    }
    return {false, "этой картой нельзя побить ни одну непобитую атаку"};
}

// ============================ Перевод ============================

ValidationResult validateTransfer(const GameState& s, const Move& m) {
    if (!s.transferEnabled)
        return {false, "режим стола не переводной"};
    if (s.phase != Phase::Defense)
        return {false, "перевод возможен только в фазе защиты"};
    if (s.turn != s.opposite(s.attacker))
        return {false, "ходит не защищающийся"};
    if (!handHas(s.myHand, m.card))
        return {false, "карты нет в руке"};

    // Переводить можно, только если на столе есть непобитая атака того же ранга.
    Rank topRank;
    if (!s.topUndefendedAttackRank(topRank))
        return {false, "нет непобитой атаки для перевода"};
    if (m.card.rank != topRank)
        return {false, "переводить можно картой того же ранга, что верхняя атака"};

    // ОГРАНИЧЕНИЕ: нельзя перевести, если у соперника карт меньше,
    // чем получится на столе после перевода (текущие непобитые + 1 новая).
    Side newDefender = s.attacker; // после перевода отбивается бывший атакующий
    int newDefenderHand = (newDefender == Side::Me)
        ? static_cast<int>(s.myHand.size())
        : s.oppHandCount;
    int cardsAfterTransfer = s.undefendedAttacksCount() + 1;
    if (newDefenderHand < cardsAfterTransfer)
        return {false, "у соперника меньше карт, чем потребуется отбить после перевода"};

    // FIX #8: лимит пар проверяем по руке НОВОГО защитника, а не текущего.
    // Раньше pairsHeadroom() считал по s.opposite(s.attacker) = текущему
    // защитнику (тому, кто переводит). Логически неверно: после перевода
    // отбиваться будет бывший атакующий, лимит должен быть по его руке.
    int byRules = s.firstTrick ? s.firstTrickLimit : 6;
    int maxPairs = std::min(byRules, newDefenderHand);
    if (s.pairsCount() >= maxPairs)
        return {false, "превышен лимит пар в коне после перевода"};

    return {true, ""};
}

// ============================ Подкидывание ============================

ValidationResult validateToss(const GameState& s, const Move& m) {
    if (s.phase != Phase::Attack)
        return {false, "подкидывать можно только в фазе атаки"};
    if (s.turn != s.attacker)
        return {false, "подкидывает атакующий"};
    if (!handHas(s.myHand, m.card))
        return {false, "карты нет в руке"};
    if (s.table.empty())
        return {false, "нечего подкидывать (стол пуст)"};

    if (s.countRankOnTable(m.card.rank) == 0)
        return {false, "подкидывать можно только ранги, уже есть на столе"};
    if (pairsHeadroom(s) <= 0)
        return {false, "превышен лимит пар в коне"};

    return {true, ""};
}

// ============================ Взять (Беру) ============================

ValidationResult validateTake(const GameState& s, const Move& /*m*/) {
    if (s.phase != Phase::Defense)
        return {false, "взять можно только в фазе защиты"};
    if (s.turn != s.opposite(s.attacker))
        return {false, "берёт защищающийся"};
    if (s.table.empty())
        return {false, "нечего брать (стол пуст)"};
    return {true, ""};
}

// ============================ Бито (конец кона) ============================

ValidationResult validateDone(const GameState& s, const Move& /*m*/) {
    if (s.table.empty())
        return {false, "кон ещё не начался"};
    // «Бито» объявляет атакующий в фазе Attack, когда все атаки побиты
    // и он добровольно завершает кон.
    if (s.phase != Phase::Attack)
        return {false, "бито может объявить только атакующий в фазе атаки"};
    if (s.turn != s.attacker)
        return {false, "бито объявляет атакующий"};
    if (s.undefendedAttacksCount() > 0)
        return {false, "есть непобитые атаки — защищающийся должен отбиться или взять"};
    return {true, ""};
}

// ============================ Pass (отказ от подкидывания) ============================

ValidationResult validatePass(const GameState& s, const Move& /*m*/) {
    // Pass — отказ от дальнейшего подкидывания. Легален ТОЛЬКО в фазе Attack
    // со стороны атакующего, когда на столе есть побитые пары (либо все побито,
    // что эквивалентно Done). Раньше возвращали {true, ""} без проверок —
    // это позволяло «завершить кон» через Pass даже в защите с непобитыми
    // атаками, что applyMove превращал в сброс стола как «бито».
    if (s.phase != Phase::Attack)
        return {false, "pass возможен только в фазе атаки"};
    if (s.turn != s.attacker)
        return {false, "pass делает атакующий"};
    if (s.table.empty())
        return {false, "нечего завершать (стол пуст)"};
    return {true, ""};
}

// ============================ Диспетчер ============================

ValidationResult validateMove(const GameState& s, const Move& m) {
    switch (m.action) {
        case Action::Attack:   return validateAttack(s, m);
        case Action::Defend:   return validateDefend(s, m);
        case Action::Transfer: return validateTransfer(s, m);
        case Action::Toss:     return validateToss(s, m);
        case Action::Take:     return validateTake(s, m);
        case Action::Done:     return validateDone(s, m);
        case Action::Pass:     return validatePass(s, m);
    }
    return {false, "неизвестное действие"};
}

// ============================ Генерация легальных ходов ============================

std::vector<Card> legalAttackCards(const GameState& s) {
    std::vector<Card> out;
    if (s.phase != Phase::Attack || s.turn != s.attacker) return out;

    // Первая карта кона — любая; иначе ранг должен быть на столе.
    const bool emptyTable = s.table.empty();
    for (const Card& c : s.myHand) {
        if (emptyTable) {
            out.push_back(c);
        } else if (s.countRankOnTable(c.rank) > 0 && pairsHeadroom(s) > 0) {
            out.push_back(c);
        }
    }
    return out;
}

std::vector<DefendOption> legalDefendCards(const GameState& s) {
    std::vector<DefendOption> out;
    if (s.phase != Phase::Defense || s.turn != s.opposite(s.attacker)) return out;

    // Для каждой непобитой атаки — все карты руки, которые её бьют.
    for (const auto& p : s.table) {
        if (p.defended) continue;
        for (const Card& c : s.myHand) {
            if (canBeat(p.attack, c, s.deck.trump)) {
                out.push_back({c, p.attack});
            }
        }
    }
    return out;
}

std::vector<Card> legalTransferCards(const GameState& s) {
    std::vector<Card> out;
    if (!s.transferEnabled) return out;
    if (s.phase != Phase::Defense || s.turn != s.opposite(s.attacker)) return out;

    Rank topRank;
    if (!s.topUndefendedAttackRank(topRank)) return out;

    // Лимит по руке нового защищающегося (бывшего атакующего).
    Side newDefender = s.attacker;
    int newDefenderHand = (newDefender == Side::Me)
        ? static_cast<int>(s.myHand.size())
        : s.oppHandCount;
    int cardsAfterTransfer = s.undefendedAttacksCount() + 1;
    if (newDefenderHand < cardsAfterTransfer) return out;
    // FIX #8: лимит пар по руке НОВОГО защитника.
    int byRules = s.firstTrick ? s.firstTrickLimit : 6;
    int maxPairs = std::min(byRules, newDefenderHand);
    if (s.pairsCount() >= maxPairs) return out;

    for (const Card& c : s.myHand) {
        if (c.rank == topRank) out.push_back(c);
    }
    return out;
}

std::vector<Card> legalTossCards(const GameState& s) {
    // По сути то же, что подкидывание в фазе Attack — генерация одинакова.
    return legalAttackCards(s);
}

bool canDeclareDone(const GameState& s) {
    if (s.table.empty()) return false;
    // Можно завершить, если все атаки побиты (нечего подкидывать обязательно).
    return s.undefendedAttacksCount() == 0;
}

} // namespace durakk
