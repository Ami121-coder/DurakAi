#include "protocol.h"

#include <stdexcept>

namespace durakk {

// ---------- Карты ----------

static std::string rankToString(Rank r) {
    switch (r) {
        case Rank::Six: return "6";
        case Rank::Seven: return "7";
        case Rank::Eight: return "8";
        case Rank::Nine: return "9";
        case Rank::Ten: return "10";
        case Rank::Jack: return "J";
        case Rank::Queen: return "Q";
        case Rank::King: return "K";
        case Rank::Ace: return "A";
    }
    return "?";
}

static std::string suitToString(Suit s) {
    switch (s) {
        case Suit::Clubs: return "clubs";
        case Suit::Diamonds: return "diamonds";
        case Suit::Hearts: return "hearts";
        case Suit::Spades: return "spades";
    }
    return "?";
}

json cardToJson(Card c) {
    return {{"r", rankToString(c.rank)}, {"s", suitToString(c.suit)}};
}

Card cardFromJson(const json& j) {
    if (!j.is_object() || !j.contains("r") || !j.contains("s"))
        throw std::runtime_error("card JSON: expected {r,s}");
    Card c;
    c.rank = rankFromString(j["r"].get<std::string>());
    c.suit = suitFromString(j["s"].get<std::string>());
    return c;
}

// ---------- Ходы ----------

std::string actionToString(Action a) {
    switch (a) {
        case Action::Attack:   return "attack";
        case Action::Defend:   return "defend";
        case Action::Transfer: return "transfer";
        case Action::Toss:     return "toss";
        case Action::Take:     return "take";
        case Action::Done:     return "done";
        case Action::Pass:     return "pass";
    }
    return "?";
}

json moveToJson(const Move& m) {
    json j;
    j["action"] = actionToString(m.action);
    if (m.action == Action::Attack || m.action == Action::Defend ||
        m.action == Action::Transfer || m.action == Action::Toss) {
        const json c = cardToJson(m.card);
        j["rank"] = c["r"];
        j["suit"] = c["s"];
        if (m.hasTarget) j["target"] = cardToJson(m.target);
    }
    if (!m.reason.empty()) j["reason"] = m.reason;
    return j;
}

// ---------- Сила ----------

Strength strengthFromString(const std::string& s) {
    if (s == "fast") return Strength::Fast;
    if (s == "deep") return Strength::Deep;
    return Strength::Normal; // default
}

// ---------- Парсинг ----------

static Side sideFromString(const std::string& s) {
    if (s == "me") return Side::Me;
    if (s == "opp") return Side::Opp;
    throw std::runtime_error("unknown side: " + s);
}

static Phase phaseFromString(const std::string& s) {
    if (s == "attack") return Phase::Attack;
    if (s == "defense") return Phase::Defense;
    if (s == "done") return Phase::Done;
    throw std::runtime_error("unknown phase: " + s);
}

static DeckSize deckSizeFromInt(int n) {
    switch (n) {
        case 24: return DeckSize::Cards24;
        case 52: return DeckSize::Cards52;
        case 36: default: return DeckSize::Cards36;
    }
}

// Вспомогательная: прочитать массив карт → vector<Card>.
static std::vector<Card> parseCards(const json& j, const char* name) {
    if (!j.contains(name)) return {};
    if (!j[name].is_array()) return {};
    std::vector<Card> out;
    for (const auto& cj : j[name]) out.push_back(cardFromJson(cj));
    return out;
}

void parseState(const json& j, GameState& state, Knowledge& knowledge) {
    // Настройки стола.
    if (j.contains("deckSize"))
        state.deckSize = deckSizeFromInt(j["deckSize"].get<int>());
    if (j.contains("transferEnabled"))
        state.transferEnabled = j["transferEnabled"].get<bool>();
    if (j.contains("flashEnabled"))
        state.flashEnabled = j["flashEnabled"].get<bool>();
    if (j.contains("firstTrickLimit"))
        state.firstTrickLimit = j["firstTrickLimit"].get<int>();

    // Козырь и колода.
    if (j.contains("trump"))
        state.deck.trump = suitFromString(j["trump"].get<std::string>());
    if (j.contains("deckCount"))
        state.deck.remaining = j["deckCount"].get<int>();

    // Моя рука → GameState + Knowledge.
    auto myCards = parseCards(j, "myHand");
    state.myHand = myCards;
    knowledge.myHand = cardsToMask(myCards);
    knowledge.trump = state.deck.trump;

    // Карты в бите → Knowledge.
    auto discCards = parseCards(j, "discard");
    knowledge.discard = cardsToMask(discCards);

    // Карты, которые соперник забрал → Knowledge.
    auto takenCards = parseCards(j, "oppTaken");
    knowledge.oppKnownTaken = cardsToMask(takenCards);

    // Счётчики.
    if (j.contains("oppHandCount"))
        state.oppHandCount = j["oppHandCount"].get<int>();
    knowledge.oppHandCount = state.oppHandCount;
    knowledge.deckRemaining = state.deck.remaining;

    // Стол → GameState + Knowledge.tableKnown.
    if (j.contains("table") && j["table"].is_array()) {
        state.table.clear();
        knowledge.tableKnown = 0;
        for (const auto& tj : j["table"]) {
            TablePair p{};
            if (!tj.contains("attack"))
                throw std::runtime_error("table entry must have 'attack'");
            p.attack = cardFromJson(tj["attack"]);
            knowledge.tableKnown |= cardBit(p.attack);
            if (tj.contains("defense") && !tj["defense"].is_null()) {
                p.defense = cardFromJson(tj["defense"]);
                p.defended = true;
                knowledge.tableKnown |= cardBit(p.defense);
            }
            state.table.push_back(p);
        }
    }

    if (j.contains("attacker")) state.attacker = sideFromString(j["attacker"].get<std::string>());
    if (j.contains("turn"))     state.turn = sideFromString(j["turn"].get<std::string>());
    if (j.contains("phase"))    state.phase = phaseFromString(j["phase"].get<std::string>());
    if (j.contains("firstTrick")) state.firstTrick = j["firstTrick"].get<bool>();
    if (j.contains("flashUsedThisTrick")) state.flashUsedThisTrick = j["flashUsedThisTrick"].get<bool>();
}

// ---------- Статистика ----------

json statsToJson(const DecisionStats& st) {
    return {
        {"mode", st.mode},
        {"playouts", st.playouts},
        {"winrate", st.winrate},
        {"timeMs", st.timeMs},
        {"depthReached", st.depthReached},
        {"solved", st.solved},
    };
}

// ---------- Легальные ходы ----------

static json cardsToJson(const std::vector<Card>& v) {
    json arr = json::array();
    for (const auto& c : v) arr.push_back(cardToJson(c));
    return arr;
}

json legalMovesToJson(const GameState& s) {
    json out;
    out["attacks"]  = cardsToJson(legalAttackCards(s));
    out["tosses"]   = cardsToJson(legalTossCards(s));
    out["transfers"] = cardsToJson(legalTransferCards(s));

    json defends = json::array();
    for (const auto& d : legalDefendCards(s)) {
        defends.push_back({{"card", cardToJson(d.card)}, {"target", cardToJson(d.target)}});
    }
    out["defends"] = defends;
    out["canDone"] = canDeclareDone(s);
    return out;
}

} // namespace durakk
