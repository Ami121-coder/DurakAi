#include "protocol.h"
#include "game_state.h"
#include "rules.h"
#include "bot.h"
#include "../third_party/nlohmann/json.hpp"
#include <string>
#include <sstream>

using json = nlohmann::json;

// Парсинг карты из JSON: {"rank": 7, "suit": "clubs"}
static Card parseCard(const json& j) {
    Card c;
    c.rank = j.value("rank", 0);
    std::string s = j.value("suit", "clubs");
    if (s == "clubs") c.suit = 0;
    else if (s == "diamonds") c.suit = 1;
    else if (s == "hearts") c.suit = 2;
    else if (s == "spades") c.suit = 3;
    return c;
}

static json cardToJson(const Card& c) {
    const char* suits[] = {"clubs", "diamonds", "hearts", "spades"};
    return {{"rank", c.rank}, {"suit", suits[c.suit]}};
}

// Парсинг полного состояния из JSON
static GameState parseState(const json& j) {
    GameState st;

    // Рука бота
    if (j.contains("myHand")) {
        for (auto& cj : j["myHand"]) st.hands[0].push_back(parseCard(cj));
    }

    // Рука соперника (может быть пустой если неизвестна)
    if (j.contains("oppHand")) {
        for (auto& cj : j["oppHand"]) st.hands[1].push_back(parseCard(cj));
    }
    st.oppCardCount = j.value("oppCardCount", (int)st.hands[1].size());

    // Стол
    if (j.contains("table")) {
        for (auto& pj : j["table"]) {
            TablePair tp;
            tp.attack = parseCard(pj["attack"]);
            if (pj.contains("defense") && pj["defense"].contains("rank") && pj["defense"]["rank"] >= 0)
                tp.defense = parseCard(pj["defense"]);
            else
                tp.defense.rank = -1;
            st.table.push_back(tp);
        }
    }

    // Колода и козырь
    st.deckCount = j.value("deckCount", 0);
    std::string ts = j.value("trumpSuit", "clubs");
    if (ts == "clubs") st.trumpSuit = 0;
    else if (ts == "diamonds") st.trumpSuit = 1;
    else if (ts == "hearts") st.trumpSuit = 2;
    else if (ts == "spades") st.trumpSuit = 3;

    // Фаза
    std::string ph = j.value("phase", "attack");
    st.phase = (ph == "defend") ? Phase::DEFEND : Phase::ATTACK;
    st.attacker = j.value("attacker", 0);

    // Параметры правил
    st.pairsLimit = j.value("pairsLimit", 6);
    st.firstTrick = j.value("firstTrick", false);
    st.transferEnabled = j.value("transferEnabled", true);
    st.flashEnabled = j.value("flashEnabled", false);

    // Сброс (для Bayesian)
    if (j.contains("discard")) {
        for (auto& cj : j["discard"]) st.discard.push_back(parseCard(cj));
    }

    return st;
}

// Обработка одной строки JSON
std::string Protocol::handleMessage(const std::string& line, Bot& bot, Rules& rules, GameState& state) {
    json req;
    try {
        req = json::parse(line);
    } catch (...) {
        return json{{"error", "invalid JSON"}}.dump();
    }

    std::string cmd = req.value("cmd", "");

    if (cmd == "ping") {
        return json{{"status", "ok"}, {"engine", "durakk_gpu_mcts"}}.dump();
    }

    if (cmd == "setState") {
        state = parseState(req["state"]);
        return json{{"status", "ok"}}.dump();
    }

    if (cmd == "decide") {
        // Можно передать состояние прямо в decide или использовать setState ранее
        if (req.contains("state")) {
            state = parseState(req["state"]);
        }

        Move move = bot.decide(state);

        json resp;
        resp["status"] = "ok";
        resp["move"] = {
            {"card", cardToJson(move.card)},
            {"type", (int)move.type}
        };

        // Человекочитаемое описание
        const char* typeNames[] = {"attack", "defend", "take", "bito", "transfer"};
        int ti = (int)move.type;
        if (ti >= 0 && ti < 5) resp["description"] = typeNames[ti];

        return resp.dump();
    }

    if (cmd == "legalMoves") {
        if (req.contains("state")) state = parseState(req["state"]);

        auto moves = rules.genLegalMoves(state, 0); // для бота (player 0)
        json arr = json::array();
        for (auto& m : moves) {
            arr.push_back({
                {"card", cardToJson(m.card)},
                {"type", (int)m.type}
            });
        }
        return json{{"status", "ok"}, {"moves", arr}}.dump();
    }

    if (cmd == "validate") {
        if (req.contains("state")) state = parseState(req["state"]);
        Card card = parseCard(req["card"]);
        int moveType = req.value("moveType", 0);

        bool ok = rules.validateMove(state, card, moveType, 0);
        return json{{"status", "ok"}, {"valid", ok}}.dump();
    }

    return json{{"error", "unknown command: " + cmd}}.dump();
}
