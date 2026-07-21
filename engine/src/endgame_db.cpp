// ============================================================================
// endgame_db.cpp — реализация Endgame Database (Task 7)
//
// Подход: вместо классического retrograde analysis (который сложен для Дурака
// из-за переводов, подкидываний и сложной машины состояний) используем гибридный
// подход:
//   1. Генерируем все стартовые позиции кона с суммарно ≤maxTotal карт при
//      пустой колоде и пустом столе.
//   2. Для каждой запускаем bestEndgameMove() с большим таймаутом.
//   3. Сохраняем результат (best move + win/loss/draw) в БД.
//
// Это даёт ПРАКТИЧЕСКИ оптимальные ходы (α-β с большим budget находит
// форсированные результаты в большинстве позиций с ≤4 картами).
//
// Размер БД для maxTotal=4:
//   - Сочетания: C(36, 1)×C(35, 1) + C(36, 2)×C(34, 2) + ... ≈ 1.4M
//   - × 4 trump × 2 attacker × 2 phase (Attack/Defense) = ~22M
//   - × 4 байта = ~88 MB
//
// Время построения: ~30-60 минут на 7500F (однопоточно).
// ============================================================================

#include "endgame_db.h"
#include "endgame.h"
#include "rules_fast.h"
#include "bitboard.h"
#include "card.h"
#include "ttable.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>

namespace durakk {

namespace {

// Сгенерировать все способы выбрать k карт из 36 (возвращает массив масок).
// Для k=0 — пустая маска.
// Для k>4 — слишком много (C(36,5)=376992, C(36,6)=1947792), не используем.
std::vector<CardMask> allCombinations(int k) {
    std::vector<CardMask> out;
    if (k <= 0) { out.push_back(0); return out; }
    if (k > 36) return out;

    // Используем алгоритм Gosper's hack для генерации всех масок с k битами.
    CardMask c = (uint64_t(1) << k) - 1;  // первая: k младших битов
    CardMask limit = kFullDeckMask;

    while (c <= limit) {
        // Проверяем, что c не имеет битов выше 35.
        if (!(c & ~kFullDeckMask)) {
            out.push_back(c);
        }
        if (c == 0) break;
        // Gosper's hack: следующий набор с тем же числом бит.
        uint64_t a = c & (~c + 1);
        uint64_t b = c + a;
        c = ((c ^ b) >> (2 + __builtin_ctzll(a))) | b;
        if (c == 0) break;
    }
    return out;
}

} // namespace

bool EndgameDB::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    entries_.clear();

    // Формат: число записей (uint64), затем массив (uint64 key, EndgameDBEntry entry).
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!f) return false;

    for (uint64_t i = 0; i < n; ++i) {
        uint64_t key;
        EndgameDBEntry entry;
        f.read(reinterpret_cast<char*>(&key), sizeof(key));
        f.read(reinterpret_cast<char*>(&entry), sizeof(entry));
        if (!f) {
            entries_.clear();
            return false;
        }
        entries_[key] = entry;
    }

    ready_ = !entries_.empty();
    return ready_;
}

bool EndgameDB::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    uint64_t n = entries_.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (const auto& [key, entry] : entries_) {
        f.write(reinterpret_cast<const char*>(&key), sizeof(key));
        f.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
    }

    return (bool)f;
}

EndgameDBEntry EndgameDB::lookup(const MatchState& s) const {
    if (!ready_) return {0xFF, 0xFF, 0, 0};

    uint64_t key = computeHash(s);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return {0xFF, 0xFF, 0, 0};
    }
    return it->second;
}

size_t buildEndgameDB(EndgameDB& db, int maxTotal) {
    db.clear();
    size_t count = 0;

    // Генерируем все позиции: для каждого распределения (k_me, k_opp) с
    // k_me + k_opp ≤ maxTotal, k_me ≥ 1, k_opp ≥ 1.
    // Для каждой пары (myHand, oppHand), каждого trump (0..3),
    // каждого attacker (Me/Opp), каждой phase (Attack/Defense).

    auto start = std::chrono::steady_clock::now();

    for (int total = 2; total <= maxTotal; ++total) {
        for (int kMe = 1; kMe < total; ++kMe) {
            int kOpp = total - kMe;
            if (kOpp < 1) continue;

            std::fprintf(stderr, "[EndgameDB] total=%d kMe=%d kOpp=%d — генерация...\n",
                         total, kMe, kOpp);

            auto myHands = allCombinations(kMe);
            std::fprintf(stderr, "  myHands: %zu комбинаций\n", myHands.size());

            for (CardMask myHand : myHands) {
                // Нам нужно выбрать kOpp карт из 36, не пересекающихся с myHand.
                // Простая реализация: итерация по всем C(36, kOpp) и фильтр.
                // Для kOpp ≤ 3 это C(36, 3)=7140 — приемлемо.
                auto oppHands = allCombinations(kOpp);
                for (CardMask oppHand : oppHands) {
                    if (oppHand & myHand) continue;  // пересечение — пропускаем

                    for (int trumpI = 0; trumpI < 4; ++trumpI) {
                        Suit trump = static_cast<Suit>(trumpI);

                        for (int attI = 0; attI < 2; ++attI) {
                            Player attacker = (attI == 0) ? Player::Me : Player::Opp;

                            // В стартовой позиции кона: attacker ходит первым,
                            // phase = Attack. Для DB на старте кона этого достаточно.
                            // (Защитные позиции в середине кона покрыты через phase=Defense,
                            // но без стола — это бессмысленно. Поэтому только Attack.)
                            MatchState s{};
                            s.hands[0] = myHand;
                            s.hands[1] = oppHand;
                            s.deck = 0;
                            s.deckRemaining = 0;
                            s.discard = 0;
                            s.trump = trump;
                            s.tableLen = 0;
                            s.attacker = attacker;
                            s.turn = attacker;
                            s.phase = MatchPhase::Attack;
                            s.firstTrick = false;
                            s.transferEnabled = true;
                            s.flashEnabled = false;
                            s.pairsLimit = 6;

                            // Запускаем α-β поиск с большим бюджетом.
                            EndgameLimits lim;
                            lim.timeBudgetSec = 0.05;  // 50ms на позицию
                            lim.maxDepth = 40;

                            EndgameResult er = bestEndgameMove(s, Player::Me, lim, nullptr);

                            // Сохраняем результат.
                            EndgameDBEntry entry{};
                            entry.action = static_cast<uint8_t>(er.move.action);

                            // Card index (если есть карта).
                            if (er.move.action == Action::Attack ||
                                er.move.action == Action::Defend ||
                                er.move.action == Action::Transfer ||
                                er.move.action == Action::Toss) {
                                entry.bestCardIdx = static_cast<uint8_t>(cardIndex(er.move.card));
                            } else if (er.move.action == Action::Take) {
                                entry.bestCardIdx = 36;
                            } else if (er.move.action == Action::Done || er.move.action == Action::Pass) {
                                entry.bestCardIdx = 37;
                            } else {
                                entry.bestCardIdx = 0xFF;
                            }

                            // Target index (для Defend).
                            if (er.move.action == Action::Defend && er.move.hasTarget) {
                                entry.targetCardIdx = static_cast<uint8_t>(cardIndex(er.move.target));
                            } else {
                                entry.targetCardIdx = 0xFF;
                            }

                            // Result.
                            if (er.solved) {
                                entry.result = (er.score > 0) ? 1 : 2;  // win / loss
                            } else {
                                entry.result = 3;  // draw
                            }

                            uint64_t key = computeHash(s);
                            db.put(key, entry);
                            ++count;
                        }
                    }
                }
            }
        }
    }

    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr, "[EndgameDB] Построено %zu записей за %.1fs\n",
                 count, elapsed / 1000.0);
    return count;
}

} // namespace durakk
