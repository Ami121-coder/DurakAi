// ============================================================================
// bindings.cpp — FIX #3: oppHandCount + dead code + encode_state cleanup
//
// Что исправлено:
//   1. oppHandCount теперь читается НАПРЯМУЮ из state.hands в конце step().
//      В self-play у нас полная информация — нет нужды跟踪ить вручную.
//      Старый ручной трекинг имел 3 бага:
//        - Take: += prevTableLen (пар) вместо popCount(tableKnown) (карт)
//        - Не учитывал doRefill после Take/Done
//        - Накапливал ошибки на длинных партиях
//   2. encode_state: убран мёртвый код (строил 252 байта, выбрасывал, строил 220).
//      Теперь сразу строит 220 в правильном layout.
//   3. oppKnownTaken теперь ВКЛЮЧЁН в encode_state (раньше выбрасывался).
//      Сеть должна знать какие карты у соперника — это критично для стратегии.
//
// Совместимость с onnx_net.cpp: layout остался [0..179] = 5 масок × 36,
// [180..219] = скаляры. [36..71] = trumpMask (козырь). oppKnownTaken — только скаляр
// popCount в байте 199. Полная маска oppTaken НЕ передаётся.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "match.h"
#include "rules_fast.h"
#include "bot.h"
#include "knowledge.h"
#include "ismcts.h"
#include "nnet/policy_value_net.h"
#include <random>
#include <vector>

namespace py = pybind11;

namespace durakk {

constexpr int kActionSize = 74;   // 36 atk/toss/def + 36 transfer + take + done
constexpr int kActionTake = 72;
constexpr int kActionDone = 73;
constexpr int kStateSize  = 220;

// ---- Хелпер для экспорта ранга в JSON-совместимый символ ----
// Возвращаем строковый код: "6".."10","J","Q","K","A".
// FIX strict_arena: раньше использовался статический char, но для ранга "10"
// нужно два символа. Поэтому std::string.
inline std::string rankToChar(Rank r) {
    switch (r) {
        case Rank::Six:   return "6";
        case Rank::Seven: return "7";
        case Rank::Eight: return "8";
        case Rank::Nine:  return "9";
        case Rank::Ten:   return "10";
        case Rank::Jack:  return "J";
        case Rank::Queen: return "Q";
        case Rank::King:  return "K";
        case Rank::Ace:   return "A";
    }
    return "?";
}

// ---- Раздача 36-карточной колоды (без изменений) ----
struct DealResult {
    CardMask hands[2];
    CardMask deck;
    Suit trump;
    int deckRemaining;
};

DealResult dealNewGame(uint64_t seed) {
    std::mt19937_64 rng(seed);
    int perm[36];
    for (int i = 0; i < 36; ++i) perm[i] = i;
    for (int i = 35; i > 0; --i) {
        int j = (int)(rng() % (unsigned)(i + 1));
        std::swap(perm[i], perm[j]);
    }
    DealResult r{};
    for (int i = 0; i < 6; ++i) r.hands[0] |= (uint64_t(1) << perm[i]);
    for (int i = 6; i < 12; ++i) r.hands[1] |= (uint64_t(1) << perm[i]);
    int trumpIdx = perm[12];
    r.trump = indexToCard(trumpIdx).suit;
    r.deck = 0;
    for (int i = 12; i < 36; ++i) r.deck |= (uint64_t(1) << perm[i]);
    r.deckRemaining = 36 - 12;
    return r;
}

class DurakEnv {
public:
    MatchState state;
    Knowledge k;
    Player viewpoint = Player::Me;
    uint64_t seed = 0;
    std::unique_ptr<PolicyValueNet> net_;

    DurakEnv() { reset(); }

    void load_model(const std::string& path, const std::string& device) {
#ifdef DURAKK_USE_ONNX
        try {
            net_ = std::make_unique<OnnxNet>(path, device, 0);
            if (net_->isReady()) {
                std::fprintf(stderr, "[DurakEnv] ONNX модель загружена: %s\n", path.c_str());
            } else {
                std::fprintf(stderr, "[DurakEnv] ONNX модель НЕ готова, fallback на RandomNet\n");
                net_ = std::make_unique<RandomNet>();
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[DurakEnv] Ошибка загрузки ONNX: %s\n", e.what());
            net_ = std::make_unique<RandomNet>();
        }
#else
        throw std::runtime_error("ONNX Runtime не включён в сборку (DURAKK_USE_ONNX=OFF)");
#endif
    }

    bool has_model() const {
        return net_ != nullptr && net_->isReady();
    }

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
        state.attacker = Player::Me;
        state.turn = Player::Me;
        state.phase = MatchPhase::Attack;
        state.firstTrick = true;
        state.transferEnabled = true;
        state.flashEnabled = false;
        state.pairsLimit = 6;

        k = Knowledge{};
        k.myHand = d.hands[0];
        k.trump = d.trump;
        k.discard = 0;
        k.oppKnownTaken = 0;
        k.tableKnown = 0;
        k.oppHandCount = 6;
        k.deckRemaining = d.deckRemaining;
        viewpoint = Player::Me;
    }

    Player currentPlayer() const {
        if (state.phase == MatchPhase::Defense) return state.defender();
        return state.attacker;
    }

    // ---- FIX #2: encode_state без мёртвого кода, с oppKnownTaken ----
    py::array_t<uint8_t> encodeState() const {
        Player vp = viewpoint;
        Player opp = other(vp);

        std::array<uint8_t, kStateSize> out{};
        uint8_t* p = out.data();

        // 5 масок × 36 = 180 байт.
        // ВАЖНО: бит маски пишем как 255 (а не 1). Модель делит uint8-вход
        // на 255 (model_resnet.py forward), поэтому 255 → 1.0 — ровно как
        // в onnx_net.cpp::encodeStateToBuffer при инференсе (там float 0/1).
        // Со значением 1 обучение видело маски как 1/255≈0.004, а инференс
        // как 1.0 — рассогласование в 255 раз, сеть работала бы некорректно.
        auto writeMask = [&](CardMask m) {
            for (int i = 0; i < 36; ++i) *p++ = (uint8_t)(((m >> i) & 1u) ? 255 : 0);
        };

        CardMask trumpMask = 0;
        for (int r = 6; r <= 14; ++r)
            trumpMask |= cardBit(Card{static_cast<Rank>(r), state.trump});

        CardMask atkMask = 0, defMask = 0;
        for (int i = 0; i < state.tableLen; ++i) {
            atkMask |= cardBit(state.table[i].attack);
            if (state.table[i].defended) defMask |= cardBit(state.table[i].defense);
        }

        CardMask oppTaken = (vp == Player::Me) ? k.oppKnownTaken : 0;

        writeMask(state.hands[toIdx(vp)]);     // [0..35]   моя рука
        writeMask(trumpMask);                   // [36..71]  козырная масть
        writeMask(atkMask);                     // [72..107] атаки
        writeMask(defMask);                     // [108..143] защиты
        writeMask(state.discard);               // [144..179] бито

        // Скаляры [180..219] — 40 байт
        // Козырь one-hot (4 байта) + скалярные фичи
        for (int s = 0; s < 4; ++s)
            *p++ = (uint8_t)((int)state.trump == s ? 255 : 0);  // 180-183

        *p++ = (uint8_t)(state.tableLen * 255 / 6);                    // 184
        *p++ = (uint8_t)(state.undefendedCount() * 255 / 6);          // 185
        *p++ = (uint8_t)(state.pairsHeadroom() * 255 / 6);            // 186
        *p++ = (uint8_t)(state.deckRemaining * 255 / 36);             // 187
        *p++ = (uint8_t)(handSize(state.hands[toIdx(opp)]) * 255 / 36); // 188
        *p++ = (uint8_t)(handSize(state.hands[toIdx(vp)]) * 255 / 36);  // 189
        *p++ = state.firstTrick ? 255 : 0;                            // 190
        *p++ = state.transferEnabled ? 255 : 0;                      // 191
        *p++ = state.flashEnabled ? 255 : 0;                         // 192
        *p++ = (state.phase == MatchPhase::Attack) ? 255 : 0;        // 193
        *p++ = (state.phase == MatchPhase::Defense) ? 255 : 0;       // 194
        *p++ = (state.turn == vp) ? 255 : 0;                         // 195
        *p++ = (state.turn == opp) ? 255 : 0;                        // 196
        *p++ = (state.attacker == vp) ? 255 : 0;                     // 197
        *p++ = (state.attacker == opp) ? 255 : 0;                    // 198
        *p++ = (uint8_t)(popCount(oppTaken) * 255 / 36);             // 199: сколько карт забрал opp

        // Остаток — нули (200..219)
        while (p < out.data() + kStateSize) *p++ = 0;

        auto result = py::array_t<uint8_t>(out.size());
        auto buf = result.request();
        std::copy(out.begin(), out.end(), static_cast<uint8_t*>(buf.ptr));
        return result;
    }

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
        std::copy(out.begin(), out.end(), static_cast<float*>(bi.ptr));
        return result;
    }

    // ---- FIX #1: step() обновляет Knowledge НАПРЯМУЮ из state ----
    bool step(int actionIndex) {
        if (state.isGameOver()) return false;

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

        // БАГ D: сохраняем карты стола ДО applyMove
        CardMask tableBefore = 0;
        for (int i = 0; i < state.tableLen; ++i) {
          tableBefore |= cardBit(state.table[i].attack);
          if (state.table[i].defended)
            tableBefore |= cardBit(state.table[i].defense);
        }

        if (!applyMove(state, chosen)) return false;
        syncKnowledgeFromState(chosen, tableBefore);  // ← передаём tableBefore
        viewpoint = currentPlayer();

        return true;
    }

    py::array_t<float> run_ismcts(double time_budget, int num_threads) {
        // FIX #4 (viewpoint bug): viewpoint должен быть currentPlayer(),
        // чтобы ISMCTS максимизировал для стороны, которая реально ходит.
        // Раньше стоял vp = Player::Me всегда — это означало, что когда ход
        // соперника, ISMCTS максимизировал для Me, а не для Opp. Бот отдавал
        // Opp'у ходы, лучшие для Me (т.е. худшие для Opp), и в self-play
        // генерились противоречивые policy-метки: encode_state() кодировал
        // состояние с перспективы currentPlayer, а ISMCTS искал лучший ход
        // для Me даже в позициях, где ходит Opp.
        Player vp = currentPlayer();
        IsmctsLimits lim;
        lim.timeBudgetSec = time_budget;
        lim.numThreads = std::max(1, num_threads);
        lim.rootTemperature = 0.0;  // inference: детерминированный argmax
        lim.dirichletEps = 0.0;    // без шума при inference

        IsmctsResult res = runIsmcts(state, k, lim, nullptr, vp, net_.get());

        std::vector<float> out(kActionSize, 0.0f);
        for (const auto& pr : res.rootProbs) {
            int a = moveToActionIndex(pr.first);
            if (a >= 0 && a < kActionSize) out[a] += (float)pr.second;
        }
        auto result = py::array_t<float>(out.size());
        auto bi = result.request();
        std::copy(out.begin(), out.end(), static_cast<float*>(bi.ptr));
        return result;
    }

    // ---- Возвращает root value из сети (для мониторинга) ----
    float get_root_value() {
        if (!has_model()) return 0.5f;
        // FIX #4: тот же фикс viewpoint, что и в run_ismcts.
        Player vp = currentPlayer();
        IsmctsLimits lim;
        lim.timeBudgetSec = 0.01;
        lim.numThreads = 1;
        IsmctsResult res = runIsmcts(state, k, lim, nullptr, vp, net_.get());
        return (float)res.rootValue;
    }

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
        d["hasModel"] = has_model();
        return d;
    }

    // ---- FIX strict_arena: экспорт полного состояния для синхронизации ----
    // Раньше арена генерировала свою раздачу (через Python random.Random),
    // что не совпадало с раздачей durakk_env (через std::mt19937_64 в C++).
    // Это приводило к ложным штрафам: бот выбирал карту из своей РЕАЛЬНОЙ
    // руки, а арена валидировала против своей ВЫДУМАННОЙ — карта "не в руке".
    //
    // Теперь арена запрашивает getStateSnapshot() у воркера и строит
    // канонический снимок из РЕАЛЬНОГО состояния движка.
    py::dict getStateSnapshot() const {
        py::dict d;

        // Настройки стола.
        d["trump"] = static_cast<int>(state.trump);
        d["transferEnabled"] = state.transferEnabled;
        d["flashEnabled"] = state.flashEnabled;
        d["firstTrick"] = state.firstTrick;
        d["pairsLimit"] = state.pairsLimit;
        d["deckRemaining"] = state.deckRemaining;

        // Чей ход / роли / фаза.
        d["attacker"] = static_cast<int>(state.attacker);  // 0=Me, 1=Opp
        d["turn"] = static_cast<int>(state.turn);
        // MatchPhase: Attack=0, Defense=1, Pursue=2, Done=3, GameOver=4
        d["phase"] = static_cast<int>(state.phase);

        // viewpoint: с чьей перспективы видим руку. В arena ussually = 0 (Me).
        d["viewpoint"] = static_cast<int>(viewpoint);

        // Моя рука (полная видимость).
        py::list my_hand;
        CardMask myMask = state.hands[toIdx(viewpoint)];
        CardMask m = myMask;
        while (m) {
            CardMask bit = m & (~m + 1);
            m ^= bit;
            int idx;
#if defined(_MSC_VER)
            unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
            idx = __builtin_ctzll(bit);
#endif
            Card c = indexToCard(idx);
            py::dict card_obj;
            card_obj["r"] = rankToChar(c.rank);
            card_obj["s"] = static_cast<int>(c.suit);
            my_hand.append(card_obj);
        }
        d["myHand"] = my_hand;

        // Сколько карт у соперника (точное содержимое не раскрываем —
        // арена использует только количество, как и должно быть в реальной игре).
        d["oppHandCount"] = popCount(state.hands[toIdx(other(viewpoint))]);

        // Стол.
        py::list table_list;
        for (int i = 0; i < state.tableLen; ++i) {
            py::dict pair;
            py::dict atk;
            atk["r"] = rankToChar(state.table[i].attack.rank);
            atk["s"] = static_cast<int>(state.table[i].attack.suit);
            pair["attack"] = atk;
            if (state.table[i].defended) {
                py::dict def;
                def["r"] = rankToChar(state.table[i].defense.rank);
                def["s"] = static_cast<int>(state.table[i].defense.suit);
                pair["defense"] = def;
                pair["defended"] = true;
            } else {
                pair["defense"] = py::none();
                pair["defended"] = false;
            }
            table_list.append(pair);
        }
        d["table"] = table_list;

        // Бито (discard).
        py::list discard_list;
        CardMask disc = state.discard;
        while (disc) {
            CardMask bit = disc & (~disc + 1);
            disc ^= bit;
            int idx;
#if defined(_MSC_VER)
            unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
            idx = __builtin_ctzll(bit);
#endif
            Card c = indexToCard(idx);
            py::dict card_obj;
            card_obj["r"] = rankToChar(c.rank);
            card_obj["s"] = static_cast<int>(c.suit);
            discard_list.append(card_obj);
        }
        d["discard"] = discard_list;

        d["isGameOver"] = state.isGameOver();
        d["winner"] = state.winner();
        return d;
    }

    bool isGameOver() const { return state.isGameOver(); }
    int winner() const { return state.winner(); }
    int currentPlayerIdx() const { return static_cast<int>(currentPlayer()); }

private:
    int moveToActionIndex(const Move& m) const {
        if (m.action == Action::Take) return kActionTake;
        if (m.action == Action::Done || m.action == Action::Pass) return kActionDone;
        if (m.action == Action::Transfer) return cardIndex(m.card) + 36;  // ← НОВОЕ
        return cardIndex(m.card);
    }

    void syncKnowledgeFromState(const Move& m, CardMask tableBeforeTake) {
        Player vp = viewpoint;
        k.myHand = state.hands[toIdx(vp)];
        k.tableKnown = 0;
        for (int i = 0; i < state.tableLen; ++i) {
          k.tableKnown |= cardBit(state.table[i].attack);
          if (state.table[i].defended)
            k.tableKnown |= cardBit(state.table[i].defense);
        }
        k.discard = state.discard;
        k.oppHandCount = handSize(state.hands[toIdx(other(vp))]);
        k.deckRemaining = state.deckRemaining;
        k.trump = state.trump;

        // БАГ D: при Take обновляем oppKnownTaken
        if (m.action == Action::Take) {
          Player taker = other(state.attacker);
          if (taker == other(vp)) {
            k.oppKnownTaken |= tableBeforeTake;  // ← НОВОЕ
          }
        }
    }
};

PYBIND11_MODULE(durakk_env, m) {
    py::class_<DurakEnv>(m, "DurakEnv")
        .def(py::init<>())
        .def("load_model", &DurakEnv::load_model, py::arg("path"), py::arg("device") = "CPU")
        .def("has_model", &DurakEnv::has_model)
        .def("reset", &DurakEnv::reset)
        .def("reset_with_seed", &DurakEnv::resetWithSeed, py::arg("seed"))
        .def("encode_state", &DurakEnv::encodeState)
        .def("get_legal_action_mask", &DurakEnv::getLegalActionMask)
        .def("step", &DurakEnv::step, py::arg("action_index"))
        .def("run_ismcts", &DurakEnv::run_ismcts,
             py::arg("time_budget"), py::arg("num_threads") = 1)
        .def("get_root_value", &DurakEnv::get_root_value)
        .def("is_game_over", &DurakEnv::isGameOver)
        .def("winner", &DurakEnv::winner)
        .def("current_player", &DurakEnv::currentPlayerIdx)
        .def("get_stats", &DurakEnv::getStats)
        // FIX strict_arena: экспорт полного состояния для синхронизации арены.
        .def("get_state_snapshot", &DurakEnv::getStateSnapshot);
    m.attr("ACTION_SIZE") = kActionSize;
    m.attr("ACTION_TAKE") = kActionTake;
    m.attr("ACTION_DONE") = kActionDone;
    m.attr("STATE_SIZE") = kStateSize;
}

} // namespace durakk
