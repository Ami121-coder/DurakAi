#include <iostream>
#include <string>
#include "protocol.h"
#include "bot.h"
#include "rules.h"
#include "game_state.h"

int main() {
    // Отключаем буферизацию для IPC
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Bot bot;
    Rules rules;
    GameState state;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        auto response = Protocol::handleMessage(line, bot, rules, state);
        std::cout << response << "\n";
        std::cout.flush();
    }

    return 0;
}
