#include "bot.h"
#include "protocol.h"
#include "rules.h"

#include <iostream>
#include <string>

namespace durakk {

namespace {

bool readLine(std::string& out) {
    std::getline(std::cin, out);
    return std::cin.good() || !out.empty();
}

void reply(const json& j) {
    std::cout << j.dump() << '\n';
    std::cout.flush();
}

json error(const std::string& msg) {
    return {{"ok", false}, {"error", msg}};
}

json handle(const json& req, GameState& state, Knowledge& knowledge, Bot& bot) {
    if (!req.contains("cmd"))
        return error("missing 'cmd'");

    const std::string cmd = req["cmd"].get<std::string>();

    if (cmd == "ping") {
        return {{"ok", true}, {"pong", true}};
    }

    if (cmd == "setState") {
        parseState(req, state, knowledge);
        return {{"ok", true}};
    }

    if (cmd == "decide") {
        SearchSettings settings;
        if (req.contains("strength"))
            settings.strength = strengthFromString(req["strength"].get<std::string>());

        DecisionStats stats;
        Move m = bot.decide(state, knowledge, settings, &stats);

        json j = moveToJson(m);
        j["ok"] = true;
        j["stats"] = statsToJson(stats);
        return j;
    }

    if (cmd == "legalMoves") {
        json j = legalMovesToJson(state);
        j["ok"] = true;
        return j;
    }

    if (cmd == "validate") {
        Move m{};
        const std::string a = req.value("action", "");
        if (a == "attack")         m.action = Action::Attack;
        else if (a == "defend")    m.action = Action::Defend;
        else if (a == "transfer")  m.action = Action::Transfer;
        else if (a == "toss")      m.action = Action::Toss;
        else if (a == "take")      m.action = Action::Take;
        else if (a == "done")      m.action = Action::Done;
        else if (a == "pass")      m.action = Action::Pass;
        else return error("unknown action");

        if (req.contains("rank") && req.contains("suit")) {
            m.card.rank = rankFromString(req["rank"].get<std::string>());
            m.card.suit = suitFromString(req["suit"].get<std::string>());
        }
        if (req.contains("target") && !req["target"].is_null()) {
            m.target = cardFromJson(req["target"]);
            m.hasTarget = true;
        }

        ValidationResult v = validateMove(state, m);
        json j;
        j["ok"] = v.ok;
        if (!v.reason.empty()) j["reason"] = v.reason;
        return j;
    }

    return error("unknown cmd: " + cmd);
}

} // namespace
} // namespace durakk

int main() {
    std::ios::sync_with_stdio(false);

    durakk::GameState state;
    durakk::Knowledge knowledge;
    durakk::Bot bot;
    std::string line;

    while (durakk::readLine(line)) {
        if (line.empty()) continue;
        try {
            durakk::json req = durakk::json::parse(line);
            durakk::json res = durakk::handle(req, state, knowledge, bot);
            durakk::reply(res);
        } catch (const std::exception& e) {
            durakk::reply(durakk::error(std::string("exception: ") + e.what()));
        }
    }
    return 0;
}
