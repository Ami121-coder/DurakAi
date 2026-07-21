#pragma once

// ============================================================================
// move_ordering.h — Доменные эвристики для упорядочивания ходов в MCTS/α-β.
//
// Цель: правильное move ordering даёт 2-3× ускорение сходимости ISMCTS и
// 3-10× ускорение α-β (отсечения срабатывают раньше).
//
// Подход:
//   1. Каждому легальному ходу присваивается приоритет (0..1) на основе
//      доменных правил Дурака.
//   2. Приоритет используется тремя способами:
//        - В ISMCTS без сети: progressive bias в UCB1
//          bonus = k_pbias * prior / (1 + visits)
//        - В ISMCTS с сетью: дополнение к PUCT prior (если нет сети — fallback)
//          effective_prior = (1-w) * nn_prior + w * heuristic_prior
//        - В α-β эндшпиле: сортировка ходов перед перебором
//   3. Эвристики основаны на принципах сильной игры в переводной дурак:
//        - Атака: подкидывать парные, беречь козыри, ходить мелочью
//        - Защита: бить минимально достаточной картой, перевод > козырь > Take
//        - Конец кона: Done, если все побито и нет выгоды от подкидывания
//
// ВАЖНО: эвристики НЕ заменяют поиск, они лишь направляют его. Если эвристика
// плохая — поиск всё равно найдёт правильный ход, просто медленнее.
// ============================================================================

#include "match.h"
#include "move.h"
#include "rules_fast.h"
#include "bitboard.h"
#include "card.h"

#include <algorithm>
#include <cstdint>

namespace durakk {

// Вес эвристического prior в гибридной схеме (0 = только сеть, 1 = только эвристика).
// 0.15 — лёгкое смещение в пользу эвристики, когда сеть неуверенна.
constexpr double kHeuristicWeightWithNet = 0.15;
// Сила progressive bias в UCB1 без сети. Классическое значение из шахмат — 5-10.
constexpr double kProgressiveBiasC = 6.0;

// Рассчитать приоритет хода в диапазоне [0, 1].
//   s — текущее состояние (ДО применения хода).
//   m — оцениваемый ход.
//   hand — маска руки ходящего (для проверки парности и т.д.).
//
// Возвращает double в [0, 1]: 1 = отличный ход, 0 = ужасный.
inline double moveHeuristic(const MatchState& s, const Move& m, CardMask hand) {
    const Suit trump = s.trump;
    const bool isTrump = (m.card.suit == trump);

    switch (m.action) {
        // ---------- Атакующие действия ----------
        case Action::Attack: {
            // Первая карта кона. Принципы:
            //   • Мелкая не-козырная > крупная не-козырная (беречь старшие).
            //   • Не козырь > козырь (козырь — резерв для побития).
            //   • Если есть пара этой карте в руке — приоритет выше (парность).
            //   • Если на руке есть козырь старше — можно рискнуть.
            int rank = static_cast<int>(m.card.rank); // 6..14
            double base = (15.0 - rank) / 9.0;         // 6→1.0, 14→0.111
            if (isTrump) base *= 0.4;                   // штраф за козырь в атаке

            // Бонус за парность: если в руке ещё одна карта того же ранга.
            CardMask sameRank = 0;
            for (int su = 0; su < 4; ++su) {
                if (su == static_cast<int>(m.card.suit)) continue;
                sameRank |= cardBit(static_cast<Suit>(su), m.card.rank);
            }
            if (hand & sameRank) base += 0.2;

            return std::min(1.0, base);
        }

        case Action::Toss: {
            // Подкидывание карты ранга, уже на столе. Это почти всегда хорошо:
            //   • Ломает защиту соперника.
            //   • Использует «лишние» мелкие карты.
            //   • Козырь подкидывать менее выгодно (жаль козыря).
            int rank = static_cast<int>(m.card.rank);
            double base = (15.0 - rank) / 9.0 + 0.3;  // бонус за подкидывание в целом
            if (isTrump) base *= 0.6;
            return std::min(1.0, base);
        }

        case Action::Transfer: {
            // Перевод: положить карту того же ранга, переняв атаку.
            // Сильный приём, но требует осторожности:
            //   • Перевод мелким рангом — хорошо (вынуждает оппонента защищаться).
            //   • Перевод козырем — плохая идея (тратим козырь на атаку).
            int rank = static_cast<int>(m.card.rank);
            double base = (15.0 - rank) / 9.0 + 0.4;
            if (isTrump) base *= 0.3;  // штраф за перевод козырем
            return std::min(1.0, base);
        }

        // ---------- Защитные действия ----------
        case Action::Defend: {
            // Побить конкретную атакующую карту. Принципы:
            //   • Минимально достаточная карта той же масти — лучше всего.
            //   • Козырь бьёт не-козырь — нормально, но если можно обойтись без козыря, лучше.
            //   • Козырь бьёт козырь — дорого, только если нет выбора.
            //   • Крупная карта для побития мелкой — расточительство.
            int defRank = static_cast<int>(m.card.rank);
            int atkRank = static_cast<int>(m.target.rank);

            // Базовая оценка: чем меньше «переплата» по рангу, тем лучше.
            double base = 1.0 - (defRank - atkRank - 1) / 8.0;
            base = std::max(0.1, std::min(1.0, base));

            // Штраф за использование козыря.
            bool atkIsTrump = (m.target.suit == trump);
            if (isTrump && !atkIsTrump) base *= 0.6;
            if (isTrump && atkIsTrump)  base *= 0.4;

            return base;
        }

        case Action::Take: {
            // Взять карты со стола. Это плохо — растим руку, даём сопернику инициативу.
            // Но иногда необходимо: нет карт для побития.
            // ВАЖНО: если у нас и так мало карт, Take может быть оправдан (добор из колоды).
            int myCards = popCount(hand);
            double base = 0.05;  // почти всегда плохо
            if (myCards <= 3) base = 0.2;  // малая рука — простительно
            if (myCards <= 1) base = 0.4;  // одна карта — можно взять и добрать
            return base;
        }

        case Action::Done: {
            // Завершить кон «бито». Это хорошо, если мы атакующий и все побито.
            // Если есть ещё карты для подкидывания — Done менее выгоден (можно дожать).
            if (s.phase == MatchPhase::Attack || s.phase == MatchPhase::Pursue) {
                // Атакующий. Проверим, есть ли что подкинуть.
                int headroom = s.pairsHeadroom();
                if (headroom > 0) {
                    // Можно ещё подкинуть — Done менее привлекателен.
                    return 0.4;
                }
                // Нечего подкидывать — Done хорош.
                return 0.8;
            }
            // Защищающийся Done — это «бито» после полного побития. Хорошо.
            return 0.7;
        }

        case Action::Pass:
            // Pass в Pursue = завершить подкидывание. Нейтрально-хорошо.
            return 0.5;

        default:
            return 0.3;
    }
}

// Упорядочить ходы по убыванию эвристического приоритета.
// Использует стабильную сортировку, чтобы сохранить порядок genLegalMoves
// для одинаковых приоритетов (детерминизм).
//   buf — массив ходов (сортируется in-place).
//   n — количество ходов.
//   s — текущее состояние.
//   hand — маска руки ходящего.
inline void orderMoves(MoveBuffer& buf, int n, const MatchState& s, CardMask hand) {
    if (n <= 1) return;
    std::stable_sort(buf.begin(), buf.begin() + n,
        [&](const Move& a, const Move& b) {
            return moveHeuristic(s, a, hand) > moveHeuristic(s, b, hand);
        });
}

// Упорядочить ходы, но всегда ставить TT-best-ход первым.
// Полезно в α-β: TT-best ход даёт earliest cutoff.
inline void orderMovesWithTTBest(MoveBuffer& buf, int n, const MatchState& s,
                                  CardMask hand, const Move* ttBest) {
    if (n <= 1) return;
    // Сначала обычное упорядочивание.
    orderMoves(buf, n, s, hand);
    // Затем поднимем TT-best на первую позицию (если он есть).
    if (!ttBest) return;
    for (int i = 0; i < n; ++i) {
        if (buf[i].action == ttBest->action &&
            buf[i].card == ttBest->card &&
            buf[i].hasTarget == ttBest->hasTarget &&
            (!buf[i].hasTarget || buf[i].target == ttBest->target)) {
            if (i != 0) std::swap(buf[0], buf[i]);
            break;
        }
    }
}

} // namespace durakk
