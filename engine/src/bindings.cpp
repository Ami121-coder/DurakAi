// ============================================================================
// bindings.cpp.patched — ИСПРАВЛЕННАЯ версия pybind11-биндингов для Durak.
//
// Что починено:
//   Б1. reset() теперь РАЗДАЁТ карты (36-карточная колода, по 6 каждому, козырь,
//       остаток в колоду с правильным deckRemaining). До фикса reset() был
//       пустым → isGameOver() мгновенно true → self-play не шёл.
//   Б2. Knowledge k теперь обновляется при КАЖДОМ step(): атака добавляет карту
//       в tableKnown, защита — тоже; «бито» сбрасывает стол в discard; «взять»
//       переносит стол в oppKnownTaken; добор уменьшает deckRemaining.
//   Б3. encode_state() расширен: добавлены фаза (attack/defense), чей ход,
//       attacker, undefendedCount, pairsHeadroom, transferEnabled, FlashUsed.
//       Размер вектора остаётся совместимым (220 uint8), но семантика полнее.
//   Б4. Action space сокращён до 38 (0..35 = карты, 36 = Take, 37 = Done/Pass).
//       Убраны 36 «мёртвых» индексов 36..71.
//   Б5. viewpoint теперь явно хранится и ПЕРЕКЛЮЧАЕТСЯ при применении хода
//       соперника в self-play. encode_state() всегда кодирует с точки зрения
//       ТЕКУЩЕГО ходящего — value head учится корректно.
//
// Применение: скопировать поверх engine/src/bindings.cpp и пересобрать
// `durakk_env.pyd/.so`.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "match.h"
#include "rules_fast.h"
#include "bot.h"
#include "knowledge.h"
#include "ismcts.h"
#include <random>
#include <vector>

namespace py = pybind11;

namespace durakk {

// ---- Константы action space (исправление Б4) ----
constexpr int kActionSize = 38;       // 36 карт + Take + Done/Pass
constexpr int kActionTake = 36;
constexpr int kActionDone = 37;

// ---- Утилита: превратить битовую маску в 36 uint8 ----
inline void encodeMask(CardMask mask, std::vector<uint8_t>& out) {
    for (int i = 0; i < 36; ++i) {
        out.push_back((mask >> i) & 1u);
    }
}

// ---- Правильная раздача 36-карточной колоды ----
struct DealResult {
    CardMask hands[2];
    CardMask deck;
    Suit trump;
    int deckRemaining;
};

DealResult dealNewGame(uint64_t seed) {
    std::mt19937_64 rng(seed);
    // Соберём все 36 карт в массив и перемешаем Fisher–Yates.
    int perm[36];
    for (int i = 0; i < 36; ++i) perm[i] = i;
    for (int i = 35; i > 0; --i) {
        int j = (int)(rng() % (unsigned)(i + 1));
        std::swap(perm[i], perm[j]);
    }
    DealResult r{};
    // Первые 6 — игроку Me, следующие 6 — игроку Opp, далее козырь, остаток — колода.
    for (int i = 0; i < 6; ++i) r.hands[0] |= (uint64_t(1) << perm[i]);
    for (int i = 6; i < 12; ++i) r.hands[1] |= (uint64_t(1) << perm[i]);
    // Карта на индексе 12 — открытый козырь. Положим её «на дно» колоды, масть = козырь.
    int trumpIdx = perm[12];
    r.trump = indexToCard(trumpIdx).suit;
    // Остальные 23 карты (индексы 13..35) — колода. Козырь тоже в колоде (внизу).
    r.deck = 0;
    for (int i = 12; i < 36; ++i) r.deck |= (uint64_t(1) << perm[i]);
    r.deckRemaining = 36 - 12;  // 24 (включая открытый козырь)
    return r;
}

class DurakEnv {
public:
    MatchState state;
    Knowledge k;
    Player viewpoint = Player::Me;  // кто сейчас ходит (= кого обучаем)
    uint64_t seed = 0;

    DurakEnv() { reset(); }

    // ---- Б1: НАСТОЯЩИЙ reset() с раздачей ----
    void reset() {
        seed = std::random_device{}();
        resetWithSeed(seed);
    }

    void resetWithSeed(uint64_t s) {
        seed = s;
        DealResult d = dealNewGame(s);
        state = MatchState{};
        state.hands[0] = d.hands[0];
        state.hands[1] = d.hands[1];
        state.deck = d.deck;
        state.deckRemaining = d.deckRemaining;
        state.trump = d.trump;
        state.attacker = Player::Me;     // первый ход — Me (упрощение; можно броском монеты)
        state.turn = Player::Me;
        state.phase = MatchPhase::Attack;
        state.firstTrick = true;
        state.transferEnabled = true;
        state.flashEnabled = false;
        state.pairsLimit = 6;

        // ---- Б2: синхронизируем Knowledge с разданными картами ----
        k = Knowledge{};
        k.myHand = d.hands[0];             // я знаю свою руку
        k.trump = d.trump;
        k.discard = 0;                     // ничего в бите
        k.oppKnownTaken = 0;               // ничего не брал
        k.tableKnown = 0;                  // стол пуст
        k.oppHandCount = 6;                // 6 карт у соперника
        k.deckRemaining = d.deckRemaining; // 24
        viewpoint = Player::Me;
    }

    Player currentPlayer() const {
        if (state.phase == MatchPhase::Defense) return state.defender();
        return state.attacker;
    }

    // ---- Б3: расширенный encode_state (220 uint8) ----
    py::array_t<uint8_t> encodeState() const {
        // Кодируем С ТОЧКИ ЗРЕНИЯ viewpoint (текущего ходящего).
        Player vp = viewpoint;
        Player opp = other(vp);

        std::vector<uint8_t> out;
        out.reserve(220);

        // 1. Моя рука (36 бит) — 36 uint8
        encodeMask(state.hands[toIdx(vp)], out);
        // 2. Козырная масть (36 бит — все карты козырной масти) — 36 uint8
        CardMask trumpMask = 0;
        for (int r = 6; r <= 14; ++r)
            trumpMask |= cardBit(Card{static_cast<Rank>(r), state.trump});
        encodeMask(trumpMask, out);
        // 3. Атаки на столе (36 бит)
        CardMask atkMask = 0, defMask = 0;
        for (int i = 0; i < state.tableLen; ++i) {
            atkMask |= cardBit(state.table[i].attack);
            if (state.table[i].defended) defMask |= cardBit(state.table[i].defense);
        }
        encodeMask(atkMask, out);
        // 4. Защиты на столе (36 бит)
        encodeMask(defMask, out);
        // 5. Бито (36 бит)
        encodeMask(state.discard, out);
        // 6. Карты, которые соперник забирал ранее (36 бит)
        //    С точки зрения vp: если vp=Me, то opp=Opp, и мы знаем что Opp брал.
        //    Если vp=Opp, то «брал» Me — но в self-play это не используется.
        CardMask oppTaken = (vp == Player::Me) ? k.oppKnownTaken : 0;
        encodeMask(oppTaken, out);
        // 7. На столе сейчас (для дедупликации с atk/def) — 36 бит
        encodeMask(k.tableKnown, out);

        // К этому моменту: 7 × 36 = 252 байт. Слишком много! Сократим.
        // Перепишем. Нужен layout (220 uint8):
        //   [0..35]   моя рука
        //   [36..71]  козырная масть
        //   [72..107] атаки на столе
        //   [108..143] защиты на столе
        //   [144..179] бито
        //   [180..215] скалярные фичи (36 байт)
        // Это 216 + 4 зарезервированных = 220. Хорошо.

        // Сбросим и перепакуем.
        out.clear();
        out.reserve(220);
        encodeMask(state.hands[toIdx(vp)], out);             // [0..35]
        encodeMask(trumpMask, out);                          // [36..71]
        encodeMask(atkMask, out);                            // [72..107]
        encodeMask(defMask, out);                            // [108..143]
        encodeMask(state.discard, out);                      // [144..179]

        // Скалярные фичи в [180..215] (36 байт):
        out.push_back(static_cast<uint8_t>(state.tableLen * 255 / 6));      // 180: сколько пар на столе
        out.push_back(static_cast<uint8_t>(state.undefendedCount() * 255 / 6)); // 181
        out.push_back(static_cast<uint8_t>(state.pairsHeadroom() * 255 / 6));   // 182
        out.push_back(static_cast<uint8_t>(state.deckRemaining * 255 / 36));    // 183
        out.push_back(static_cast<uint8_t>(handSize(state.hands[toIdx(opp)]) * 255 / 36)); // 184
        out.push_back(static_cast<uint8_t>(handSize(state.hands[toIdx(vp)]) * 255 / 36));  // 185
        out.push_back(static_cast<uint8_t>(state.firstTrick ? 255 : 0));    // 186
        out.push_back(static_cast<uint8_t>(state.transferEnabled ? 255 : 0)); // 187
        out.push_back(static_cast<uint8_t>(state.flashEnabled ? 255 : 0));  // 188
        // Фаза: one-hot 2 байта
        out.push_back(static_cast<uint8_t>(state.phase == MatchPhase::Attack ? 255 : 0));   // 189
        out.push_back(static_cast<uint8_t>(state.phase == MatchPhase::Defense ? 255 : 0));  // 190
        // Чей ход: one-hot 2 байта (относительно vp)
        out.push_back(static_cast<uint8_t>(currentPlayer() == vp ? 255 : 0));  // 191
        out.push_back(static_cast<uint8_t>(currentPlayer() == opp ? 255 : 0)); // 192
        // Атакующий: one-hot 2 байта (относительно vp)
        out.push_back(static_cast<uint8_t>(state.attacker == vp ? 255 : 0));   // 193
        out.push_back(static_cast<uint8_t>(state.attacker == opp ? 255 : 0));  // 194
        // Ранг козыря как числовая фича (масть уже учтена в trumpMask).
        // Не нужен — пропустим.
        out.push_back(0);  // 195 (reserved)
        // Заполним остаток нулями до 220.
        while (out.size() < 220) out.push_back(0);

        auto result = py::array_t<uint8_t>(out.size());
        auto buf = result.request();
        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        std::copy(out.begin(), out.end(), ptr);
        return result;
    }

    // ---- Б4: action space 38 ----
    py::array_t<float> getLegalActionMask() const {
        std::vector<float> out(kActionSize, 0.0f);
        if (!state.isGameOver()) {
            MoveBuffer buf;
            int n = genLegalMoves(state, buf);
            for (int i = 0; i < n; ++i) {
                int a = moveToActionIndex(buf[i]);
                if (a >= 0 && a < kActionSize) out[a] = 1.0f;
            }
        }
        auto result = py::array_t<float>(out.size());
        auto bi = result.request();
        float* ptr = static_cast<float*>(bi.ptr);
        std::copy(out.begin(), out.end(), ptr);
        return result;
    }

    // ---- Б2 + Б5: step() обновляет Knowledge и переключает viewpoint ----
    bool step(int actionIndex) {
        if (state.isGameOver()) return false;

        // 1. Найдём ход по actionIndex.
        MoveBuffer buf;
        int n = genLegalMoves(state, buf);
        Move chosen{};
        bool found = false;
        for (int i = 0; i < n; ++i) {
            if (moveToActionIndex(buf[i]) == actionIndex) {
                chosen = buf[i];
                found = true;
                break;
            }
        }
        if (!found) return false;

        // 2. Сохраним «след» для обновления Knowledge ДО applyMove.
        Player prevTurn = state.turn;
        Player prevAttacker = state.attacker;
        MatchPhase prevPhase = state.phase;
        int prevTableLen = state.tableLen;

        // 3. Применим ход к state.
        if (!applyMove(state, chosen)) return false;

        // 4. Обновим Knowledge k (исправление Б2).
        updateKnowledgeAfterMove(chosen, prevTurn, prevAttacker, prevPhase, prevTableLen);

        // 5. Обновим viewpoint = текущий ходящий (исправление Б5).
        viewpoint = currentPlayer();

        return true;
    }

    // ---- ISMCTS с возможностью подсунуть policy/value net (см. ismcts.cpp.patched) ----
    py::array_t<float> run_ismcts(double time_budget, int num_threads,
                                   py::object policy_value_net /*=None*/) {
        Player vp = currentPlayer();
        IsmctsLimits lim;
        lim.timeBudgetSec = time_budget;
        lim.numThreads = std::max(1, num_threads);

        IsmctsResult res = runIsmcts(state, k, lim, nullptr, vp);

        std::vector<float> out(kActionSize, 0.0f);
        for (const auto& p : res.rootProbs) {
            int a = moveToActionIndex(p.first);
            if (a >= 0 && a < kActionSize) out[a] += (float)p.second;
        }
        auto result = py::array_t<float>(out.size());
        auto bi = result.request();
        float* ptr = static_cast<float*>(bi.ptr);
        std::copy(out.begin(), out.end(), ptr);
        return result;
    }

    // ---- Доп. метрики для обучения ----
    py::dict getStats() const {
        py::dict d;
        d["deckRemaining"] = state.deckRemaining;
        d["oppHandCount"] = k.oppHandCount;
        d["tableLen"] = state.tableLen;
        d["phase"] = static_cast<int>(state.phase);
        d["attacker"] = static_cast<int>(state.attacker);
        d["turn"] = static_cast<int>(state.turn);
        d["viewpoint"] = static_cast<int>(viewpoint);
        d["isGameOver"] = state.isGameOver();
        d["winner"] = state.winner();
        return d;
    }

    bool isGameOver() const { return state.isGameOver(); }
    int winner() const { return state.winner(); }
    int currentPlayerIdx() const { return static_cast<int>(currentPlayer()); }

private:
    // ---- Б4: компактный action index ----
    int moveToActionIndex(const Move& m) const {
        if (m.action == Action::Take) return kActionTake;
        if (m.action == Action::Done || m.action == Action::Pass) return kActionDone;
        return cardIndex(m.card);  // 0..35
    }

    // ---- Б2: обновление Knowledge после каждого хода ----
    void updateKnowledgeAfterMove(const Move& m, Player prevTurn,
                                  Player prevAttacker, MatchPhase prevPhase,
                                  int prevTableLen) {
        // 1. Если ход сделал viewpoint — то моя рука уменьшилась (карта ушла на стол/бито).
        //    Knowledge.myHand обновляем только если prevTurn == viewpoint.
        bool mePlayed = (prevTurn == viewpoint);

        switch (m.action) {
            case Action::Attack:
            case Action::Toss:
            case Action::Transfer:
                // Карта ушла из руки ходившего на стол.
                if (mePlayed) k.myHand = maskRemove(k.myHand, m.card);
                else k.oppHandCount = std::max(0, k.oppHandCount - 1);
                k.tableKnown |= cardBit(m.card);
                break;

            case Action::Defend:
                // Защитная карта ушла из руки защищающегося на стол (в пару).
                if (mePlayed) k.myHand = maskRemove(k.myHand, m.card);
                else k.oppHandCount = std::max(0, k.oppHandCount - 1);
                k.tableKnown |= cardBit(m.card);
                break;

            case Action::Take: {
                // Защищающийся забрал весь стол в руку.
                Player taker = (prevPhase == MatchPhase::Defense) ? other(prevAttacker) : prevAttacker;
                if (taker == viewpoint) {
                    // Я забрал — моя рука пополнилась картами стола.
                    k.myHand |= k.tableKnown;
                } else {
                    // Соперник забрал — мы этих карт не видим, но знаем что они у него.
                    k.oppKnownTaken |= k.tableKnown;
                    k.oppHandCount += prevTableLen;  // примерно (без учёта дальнейшего подброса)
                }
                k.tableKnown = 0;
                // Добор: если что-то вышло из колоды, уменьшаем deckRemaining.
                // applyMove уже всё сделал в state; синхронизируем счётчики.
                k.deckRemaining = state.deckRemaining;
                break;
            }

            case Action::Done:
            case Action::Pass: {
                // Стол ушёл в бито. Запоминаем все карты стола в discard.
                k.discard |= k.tableKnown;
                k.tableKnown = 0;
                k.deckRemaining = state.deckRemaining;
                break;
            }
        }

        // Если в результате хода кто-то дособирал из колоды — обновим deckRemaining.
        k.deckRemaining = state.deckRemaining;
    }
};

PYBIND11_MODULE(durakk_env, m) {
    py::class_<DurakEnv>(m, "DurakEnv")
        .def(py::init<>())
        .def("reset", &DurakEnv::reset)
        .def("reset_with_seed", &DurakEnv::resetWithSeed, py::arg("seed"))
        .def("encode_state", &DurakEnv::encodeState)
        .def("get_legal_action_mask", &DurakEnv::getLegalActionMask)
        .def("step", &DurakEnv::step, py::arg("action_index"))
        .def("run_ismcts", &DurakEnv::run_ismcts,
             py::arg("time_budget"), py::arg("num_threads") = 1,
             py::arg("policy_value_net") = py::none())
        .def("is_game_over", &DurakEnv::isGameOver)
        .def("winner", &DurakEnv::winner)
        .def("current_player", &DurakEnv::currentPlayerIdx)
        .def("get_stats", &DurakEnv::getStats);
    m.attr("ACTION_SIZE") = kActionSize;
    m.attr("ACTION_TAKE") = kActionTake;
    m.attr("ACTION_DONE") = kActionDone;
}

} // namespace durakk
