#pragma once

// ============================================================================
// knowledge.h — Bayesian Belief State для PIMCTS (Task 2)
//
// Что добавлено относительно исходной версии:
//   1. oppProbs[36] — вероятность каждой карты быть у оппонента.
//      Значения в [0, 1]. Если карта точно известна (моя / в бите / на столе /
//      точно у оппонента) — значение фиксировано. Иначе — аналитически
//      пересчитывается из условия нормировки.
//   2. updateFromEvent() — пересчёт вероятностей по событиям:
//        • RefusalEvent: соперник НЕ отбил козырного валета → все козыри
//          выше валета получают prob=0 (их у него точно нет).
//        • TakeEvent: соперник забрал карты со стола → эти карты у него
//          точно (prob=1).
//        • RefillEvent: добор из колоды → неизвестные карты распределяются
//          равномерно между оппонентом и колодой (с учётом размера колоды).
//        • SuitRefusalEvent: соперник НЕ перевёл масть X → вероятность
//          карт масти X у него уменьшается (он бы перевёл, если бы мог).
//   3. sampleOppHandWeighted() — weighted Fisher-Yates sampling руки
//      оппонента из unknownPool по oppProbs. Заменяет uniform sampleSubset().
//
// ВАЖНО: вероятности используются ТОЛЬКО в C++ поиске (PIMCTS). Они НЕ
// меняют напрямую инференс нейросети — для этого есть Task 3 (NB Fusion).
// ============================================================================

#include "bitboard.h"
#include "card.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <utility>

namespace durakk {

// «Абсолютная память» бота: что профессиональный игрок точно знает о партии.
//
// В дураке игрок не видит руку соперника и порядок колоды, но ТОЧНО знает:
//   • свою руку;
//   • козырь;
//   • каждую ушедшую в отбой (биту) карту;
//   • какие карты соперник ЗАБРАЛ со стола (он их видел, когда соперник «брал»).
// Отсюда методом исключения строится «неизвестный пул» — множество карт, чьё
// расположение (у соперника / в колоде) боту неизвестно. ISMCTS семплирует
// из этого пула детерминизации.
struct Knowledge {
    CardMask myHand       = 0;  // моя рука (точно)
    CardMask discard      = 0;  // все карты, ушедшие в отбой (память)
    CardMask oppKnownTaken = 0; // карты, которые соперник ЗАБРАЛ (видели на столе → у него)
    CardMask tableKnown   = 0;  // карты, лежащие сейчас на столе (видим все)

    Suit trump = Suit::Spades;

    int oppHandCount  = 0;  // сколько карт сейчас у соперника
    int deckRemaining = 0;  // сколько карт осталось в колоде

    // Task 2: Байесовские вероятности карт у оппонента.
    // Для каждой из 36 карт хранится P(карта у оппонента | наблюдения).
    // Значения в [0.0, 1.0]. Карты с фиксированным состоянием:
    //   - моя карта: prob = 0
    //   - в бите:    prob = 0
    //   - на столе:  prob = 0
    //   - точно у оппонента (Take-наблюдение): prob = 1
    // Карты из unknownPool: пропорционально P_unknown = oppHandCount / (oppHandCount + deckRemaining),
    // с корректировками по событиям (refusal, suit-empty и т.д.).
    float oppProbs[36] = {};

    // Task 2: Какие масти оппонент уже «отказался» иметь (не перевёл, не отбил).
    // Битовая маска: бит i = масть i (0..3) помечена как «оппонент точно не имеет
    // карт выше порога refusalThreshold[i]».
    // refusalThreshold[suit] — если оппонент не отбил карту ранга R мастью S,
    // то у него нет карт масти S с рангом > R. Храним МИНИМУМ по всем наблюдениям
    // (минимальный ранг, выше которого у оппонента точно нет карт этой масти).
    int refusalThreshold[4] = {14, 14, 14, 14};  // по умолчанию — нет информации (14 = туз = верхний предел)

    // Множество карт, чьё местоположение боту НЕ известно (ни у меня, ни в бите,
    // ни у соперника-известно-взятых, ни на столе). Эти карты распределены между
    // рукой соперника и колодой в неизвестной пропорции.
    CardMask unknownPool() const {
        // unknown = всё − (моя рука ∪ бита ∪ взято соперником ∪ стол)
        CardMask known = myHand | discard | oppKnownTaken | tableKnown;
        return (~known) & kFullDeckMask;
    }

    // Содержит ли неизвестный пул достаточно карт для раздачи сопернику.
    bool canSample() const {
        return popCount(unknownPool()) >= oppHandCount;
    }

    // Проверка целостности: ни одна карта не должна встречаться в двух местах.
    // Возвращает true, если множества не пересекаются.
    bool consistent() const {
        CardMask a = myHand & discard;
        CardMask b = myHand & oppKnownTaken;
        CardMask c = discard & oppKnownTaken;
        CardMask d = (myHand | discard | oppKnownTaken) & tableKnown;
        return (a | b | c | d) == 0;
    }

    // Очистка.
    void reset() {
        myHand = discard = oppKnownTaken = tableKnown = 0;
        oppHandCount = deckRemaining = 0;
        trump = Suit::Spades;
        // Task 2: сброс вероятностей и порогов.
        std::memset(oppProbs, 0, sizeof(oppProbs));
        for (int i = 0; i < 4; ++i) refusalThreshold[i] = 14;
    }

    // ========================================================================
    // Task 2: Байесовское обновление
    // ========================================================================

    // Пересчитать oppProbs из текущих известных множеств + порогов отказа.
    // Вызывается после любого изменения myHand/discard/oppKnownTaken/tableKnown/
    // oppHandCount/deckRemaining/refusalThreshold.
    void recomputeProbs() {
        // Базовая вероятность для неизвестной карты:
        //   P = oppHandCount / (oppHandCount + deckRemaining)
        // Если deckRemaining == 0 — все неизвестные точно у оппонента (P=1).
        // Если oppHandCount == 0 — P=0.
        int denom = oppHandCount + deckRemaining;
        float baseP = (denom > 0) ? float(oppHandCount) / float(denom) : 0.0f;
        if (deckRemaining == 0 && oppHandCount > 0) baseP = 1.0f;
        if (oppHandCount == 0) baseP = 0.0f;

        CardMask unknown = unknownPool();
        for (int i = 0; i < 36; ++i) {
            CardMask bit = uint64_t(1) << i;
            if (unknown & bit) {
                // Карта в unknown pool.
                Card c = indexToCard(i);
                int suitIdx = static_cast<int>(c.suit);
                int rank = static_cast<int>(c.rank);

                // Task 2: если оппонент «отказался» от карт этой масти выше порога —
                // вероятность = 0 (он бы отбил/перевёл, если бы мог).
                // Порог означает: у оппонента нет карт масти suitIdx с рангом > threshold.
                if (rank > refusalThreshold[suitIdx]) {
                    oppProbs[i] = 0.0f;
                } else {
                    // Нормировка: если часть карт unknownPool обнулилась из-за порогов,
                    // остальные получают увеличенную долю.
                    oppProbs[i] = baseP;
                }
            } else if (oppKnownTaken & bit) {
                // Точно у оппонента.
                oppProbs[i] = 1.0f;
            } else {
                // Моя / в бите / на столе — точно НЕ у оппонента.
                oppProbs[i] = 0.0f;
            }
        }

        // Нормировка: сумма oppProbs по неизвестным должна ≈ oppHandCount.
        // Если из-за порогов сумма меньше — повышаем остальные пропорционально.
        // Если больше — понижаем. Это CORRECT Bayesian update по наблюдению отказа.
        normalizeUnknownToOppHandCount();
    }

    // Нормализовать вероятности неизвестных карт так, чтобы их сумма ≈ oppHandCount.
    void normalizeUnknownToOppHandCount() {
        if (oppHandCount <= 0) {
            // Все неизвестные = 0 (нечего раздать).
            CardMask unknown = unknownPool();
            for (int i = 0; i < 36; ++i) {
                if (unknown & (uint64_t(1) << i)) oppProbs[i] = 0.0f;
            }
            return;
        }

        // Сумма вероятностей по неизвестным картам.
        CardMask unknown = unknownPool();
        double sum = 0.0;
        int nUnknown = 0;
        for (int i = 0; i < 36; ++i) {
            if (unknown & (uint64_t(1) << i)) {
                sum += oppProbs[i];
                ++nUnknown;
            }
        }

        if (nUnknown == 0) return;
        if (sum < 1e-6) {
            // Все обнулились — fallback на uniform.
            float uniform = float(oppHandCount) / float(nUnknown);
            if (uniform > 1.0f) uniform = 1.0f;
            for (int i = 0; i < 36; ++i) {
                if (unknown & (uint64_t(1) << i)) oppProbs[i] = uniform;
            }
            return;
        }

        // Цель: сумма неизвестных = oppHandCount (с учётом, что oppKnownTaken
        // уже даёт фиксированный вклад в общую руку оппонента).
        // Сумма oppProbs по unknown = oppHandCount - popCount(oppKnownTaken).
        int oppKnownCount = popCount(oppKnownTaken);
        double target = std::max(0, oppHandCount - oppKnownCount);
        double scale = target / sum;
        for (int i = 0; i < 36; ++i) {
            if (unknown & (uint64_t(1) << i)) {
                oppProbs[i] = std::min(1.0f, std::max(0.0f, float(oppProbs[i] * scale)));
            }
        }
    }

    // ========================================================================
    // Task 2: Обработчики событий
    // ========================================================================

    // Событие «отказ от побития»: оппонент НЕ отбил атакующую карту `atk`.
    // Это означает, что у оппонента нет карт той же масти с рангом > atk.rank
    // и нет козырей (если atk — не козырь) ИЛИ нет козырей > atk.rank (если atk — козырь).
    //
    // Логика:
    //   • Если atk — не козырь, оппонент мог побить:
    //       - картой той же масти, рангом выше
    //       - козырем (любого ранга)
    //     Раз не побил — нет карт той же масти > atk.rank И нет козырей.
    //   • Если atk — козырь, оппонент мог побить только козырем выше.
    //     Раз не побил — нет козырей > atk.rank.
    void onRefusalToDefend(Card atk) {
        int atkRank = static_cast<int>(atk.rank);
        int atkSuit = static_cast<int>(atk.suit);

        if (atk.suit != trump) {
            // Штраф по масти атаки.
            if (refusalThreshold[atkSuit] > atkRank) {
                refusalThreshold[atkSuit] = atkRank;
            }
            // Штраф по козырю: нет козырей вообще.
            int trumpSuit = static_cast<int>(trump);
            if (refusalThreshold[trumpSuit] > 5) {
                refusalThreshold[trumpSuit] = 5;  // 6 — минимальный ранг
            }
        } else {
            // atk — козырь. Нет козырей > atkRank.
            if (refusalThreshold[atkSuit] > atkRank) {
                refusalThreshold[atkSuit] = atkRank;
            }
        }
        recomputeProbs();
    }

    // Событие «взятие карт»: оппонент взял карты со стола (Take).
    // Эти карты теперь точно у него.
    void onTakeCards(CardMask takenCards) {
        oppKnownTaken |= takenCards;
        // oppHandCount обновляется вызывающим кодом.
        recomputeProbs();
    }

    // Событие «отказ от перевода»: оппонент НЕ перевёл масть X.
    // Это означает, что у оппонента нет карт масти X с рангом >= requiredRank.
    void onRefusalToTransfer(Suit refusedSuit, int requiredRank) {
        int suitIdx = static_cast<int>(refusedSuit);
        // Нет карт этой масти с рангом >= requiredRank
        // → порог = requiredRank - 1 (если есть карта ранга < requiredRank, она могла быть,
        // но не подошла для перевода).
        // Но мы не можем сказать, что нет карт РАНЬШЕ requiredRank — могли быть, но не перевёл.
        // Поэтому ограничиваем только верх: нет карт >= requiredRank.
        if (refusalThreshold[suitIdx] >= requiredRank) {
            refusalThreshold[suitIdx] = requiredRank - 1;
        }
        recomputeProbs();
    }

    // Событие «отказ от подкидывания» (атакующий не подкинул карту ранга R):
    // неинформативно для нашего viewpoint (мы и так видим все свои карты).
    // Игнорируется.
    void onRefusalToToss(Rank /*r*/) {
        // No-op: атакующий (это мы или оппонент) не обязан подкидывать.
    }

    // ========================================================================
    // Task 2: Weighted sampling
    // ========================================================================

    // Weighted Fisher-Yates: выбрать `need` карт из `pool`, используя oppProbs как веса.
    // Возвращает битовую маску выбранных карт.
    //
    // Алгоритм:
    //   1. Собрать все карты из pool в массив.
    //   2. Для каждой карты вычислить вес = oppProbs[card] (clamp > 0).
    //   3. Повторить need раз:
    //        - Выбрать карту с вероятностью пропорционально весу.
    //        - Удалить её из массива.
    //   4. Вернуть маску выбранных карт.
    //
    // Сложность: O(need * n), где n = popCount(pool). Для Durak n ≤ 24, need ≤ 6 — быстро.
    CardMask sampleOppHandWeighted(CardMask pool, int need, uint64_t seed) const {
        int n = popCount(pool);
        if (n == 0 || need <= 0) return 0;
        if (need >= n) return pool;  // все неизвестные = у оппонента

        // Собрать карты и веса.
        int idx[36];
        float w[36];
        int cnt = 0;
        CardMask p = pool;
        while (p) {
            CardMask bit = p & (~p + 1);
            p ^= bit;
            int i;
#if defined(_MSC_VER)
            unsigned long ul; _BitScanForward64(&ul, bit); i = (int)ul;
#else
            i = __builtin_ctzll(bit);
#endif
            idx[cnt] = i;
            // Вес: если prob=0 — даём минимальный epsilon (для случая ошибки в трекинге).
            // Иначе — prob.
            float prob = oppProbs[i];
            w[cnt] = (prob > 1e-6f) ? prob : 1e-6f;
            ++cnt;
        }

        // Простой RNG из seed (thread-local).
        uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ULL;
        auto next_rand = [&state]() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            return state;
        };

        // Weighted reservoir-like sampling без замены.
        CardMask out = 0;
        for (int k = 0; k < need; ++k) {
            // Сумма весов оставшихся.
            double sum = 0.0;
            for (int i = k; i < cnt; ++i) sum += w[i];
            if (sum <= 0) break;

            // Выбираем элемент пропорционально весу.
            double r = (double)(next_rand() % 1000000) / 1000000.0 * sum;
            double acc = 0.0;
            int chosen = k;
            for (int i = k; i < cnt; ++i) {
                acc += w[i];
                if (r <= acc) { chosen = i; break; }
            }
            // Меняем местами с k-м.
            std::swap(idx[k], idx[chosen]);
            std::swap(w[k], w[chosen]);
        }

        for (int k = 0; k < need; ++k) {
            out |= (uint64_t(1) << idx[k]);
        }
        return out;
    }
};

} // namespace durakk
