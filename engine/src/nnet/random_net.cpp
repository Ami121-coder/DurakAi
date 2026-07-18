#include "policy_value_net.h"
#include "../rules_fast.h"

namespace durakk {

PVResult RandomNet::evaluate(const MatchState& s, Player /*viewpoint*/) {
    // Заглушка: равномерное распределение по всем легальным ходам.
    MoveBuffer buf;
    int n = genLegalMoves(s, buf);
    PVResult out;
    if (n > 0) {
        float p = 1.0f / static_cast<float>(n);
        out.policy.reserve(n);
        for (int i = 0; i < n; ++i) out.policy.emplace_back(buf[i], p);
    }
    out.value = 0.5f; // «не знаю»
    return out;
}

} // namespace durakk
