#pragma once

#include "game_state.h"
#include "move.h"

#include <vector>
#include <string>

namespace durakk {

// Результат валидации хода.
struct ValidationResult {
    bool ok = false;
    std::string reason;  // человекочитаемое объяснение, почему нельзя (для UI)
};

// ------------------------- Базовые правила -------------------------

// Лимит пар в одном коне (отбое): min(максПар, картУЗащищающегося).
// максПар = firstTrick ? firstTrickLimit : 6.
int maxPairsThisTrick(const GameState& s, Side defender);

// Можно ли побить `attacker` картой `defender` при козыре trump.
bool canBeat(Card attacker, Card defender, Suit trump);

// ------------------------- Валидация конкретных ходов -------------------------

ValidationResult validateAttack(const GameState& s, const Move& m);
ValidationResult validateDefend(const GameState& s, const Move& m);
ValidationResult validateTransfer(const GameState& s, const Move& m);
ValidationResult validateToss(const GameState& s, const Move& m);
ValidationResult validateTake(const GameState& s, const Move& m);
ValidationResult validateDone(const GameState& s, const Move& m);
ValidationResult validatePass(const GameState& s, const Move& m);

// Единая точка валидации: разбирает Move по action и вызывает нужную проверку.
ValidationResult validateMove(const GameState& s, const Move& m);

// ------------------------- Генерация легальных ходов -------------------------

// Все карты своей руки, которыми можно атаковать (фаза Attack, атакующий = я).
std::vector<Card> legalAttackCards(const GameState& s);

// Все варианты защиты от конкретной непобитой атаки (фаза Defense).
// Возвращает пары (защищающая карта, какую атакующую бьём).
struct DefendOption { Card card; Card target; };
std::vector<DefendOption> legalDefendCards(const GameState& s);

// Все карты, которыми можно перевести (если режим переводной и лимит позволяет).
std::vector<Card> legalTransferCards(const GameState& s);

// Все карты, которыми можно подкинуть (фаза Attack после отбоя части карт,
// либо сразу после успешной защиты — см. applyDefend).
std::vector<Card> legalTossCards(const GameState& s);

// true, если атакующий сейчас может легально объявить «бито»
// (все атаки побиты и атакующий не обязан подкидывать).
bool canDeclareDone(const GameState& s);

} // namespace durakk
