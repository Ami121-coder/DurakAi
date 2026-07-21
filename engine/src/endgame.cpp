#include "endgame.h"
#include "rules_fast.h"
#include "bitboard.h"
#include "card.h"
#include "move_ordering.h"  // Task 1: move ordering в α-β

#include <algorithm>
#include <chrono>

namespace durakk {

namespace {

using Clock = std::chrono::steady_clock;

// Контекст поиска с таймером и таблицей перестановок.
struct SearchCtx {
    TTable tt;
    Clock::time_point start;
    Clock::time_point deadline;
    long nodes = 0;
    Player viewpoint;
    bool stopped = false;

    bool timeUp() {
        if (stopped) return true;
        if ((nodes & 1023) == 0) { // проверяем не каждый узел — экономим такты
            if (Clock::now() >= deadline) stopped = true;
        }
        return stopped;
    }
};

// Нега-макс вариант: возвращаем оценку С ТОЧКИ ЗРЕНИЯ игрока, чей ход в s.
// Оценки: +10000 - (глубина) = форсированная победа; -10000 + (глубина) = поражение;
// 0 — не найдено в пределах глубины (эквивалент ничьей/неизвестно).
// Чем мельче победа (больше глубина), тем меньше бонус — это даёт «быстрые» маты.
int negamax(MatchState& s, int depth, int alpha, int beta, SearchCtx& ctx) {
    ++ctx.nodes;

    if (s.isGameOver()) {
        int w = s.winner();
        // Если победил тот, кто только что ХОДИЛ (не текущий ходящий) — это проигрыш
        // для текущего ходящего: текущий проиграл.
        Player cur = s.turn;
        int curIdx = toIdx(cur);
        if (w == curIdx) return 10000 - depth;      // текущий выиграл
        if (w == 1 - curIdx) return -(10000 - depth); // текущий проиграл
        return 0; // ничья
    }

    if (depth <= 0 || ctx.timeUp()) {
        // FIX #14: возврат 0 трактовался как «ничья», что искажало α-β
        // на обрезанных поддеревьях. Возвращаем простую эвристику:
        // разница рук со знаком текущего игрока. Козырь учитывается ×2.
        int myIdx = toIdx(s.turn);
        int myCards = popCount(s.hands[myIdx]);
        int oppCards = popCount(s.hands[1 - myIdx]);
        // Лёгкая поправка: козыри ценнее.
        CardMask trumpMask = 0;
        for (int r = 6; r <= 14; ++r)
            trumpMask |= cardBit(Card{static_cast<Rank>(r), s.trump});
        int myTrumps = popCount(s.hands[myIdx] & trumpMask);
        int oppTrumps = popCount(s.hands[1 - myIdx] & trumpMask);
        int heuristic = (oppCards - myCards) * 10 + (oppTrumps - myTrumps) * 4;
        return heuristic;
    }

    // Заглянем в TT.
    uint64_t key = computeHash(s);
    const TTEntry* e = ctx.tt.probe(key);
    if (e && e->depth >= depth) {
        if (e->flag == 0) return e->value;
        if (e->flag == 1 && e->value > alpha) alpha = e->value;
        if (e->flag == 2 && e->value < beta)  beta  = e->value;
        if (alpha >= beta) return e->value;
    }

    MoveBuffer buf;
    int n = genLegalMoves(s, buf);

    // Если ходов нет вообще (защищающийся не может побить) — это «взять»,
    // и applyMove(Take) обработает это. Но если генератор ничего не дал в защите
    // (брать должно быть всегда легально при непустом столе) — это аномалия;
    // вернём 0 как «ничья», чтобы не зациклиться.
    if (n == 0) return 0;

    // Task 1: упорядочивание ходов по доменной эвристике + TT-best первым.
    // Достаём TT-best (если есть) и передаём в orderMovesWithTTBest.
    const Move* ttBestPtr = (e && e->hasBest) ? &e->best : nullptr;
    orderMovesWithTTBest(buf, n, s, s.hands[toIdx(s.turn)], ttBestPtr);

    int origAlpha = alpha;
    int best = -20000;
    Move bestMove = buf[0];
    bool bestMoveSet = false;

    for (int i = 0; i < n; ++i) {
        if (ctx.timeUp()) break;

        MatchState child = s; // копия
        if (!applyMove(child, buf[i])) continue;

        int val = -negamax(child, depth - 1, -beta, -alpha, ctx);
        if (val > best) {
            best = val;
            bestMove = buf[i];
            bestMoveSet = true;
        }
        if (best > alpha) alpha = best;
        if (alpha >= beta) break; // отсечение
    }

    // Запишем в TT.
    uint8_t flag = 0;
    if (best <= origAlpha) flag = 2;       // upper bound
    else if (best >= beta) flag = 1;       // lower bound
    else flag = 0;                          // exact
    TTEntry entry;
    entry.depth = depth;
    entry.value = best;
    entry.flag = flag;
    entry.best = bestMove;
    entry.hasBest = bestMoveSet;
    entry.generation = ctx.tt.generation();
    ctx.tt.store(key, entry);

    return best;
}

} // namespace

EndgameResult bestEndgameMove(MatchState s, Player viewpoint, const EndgameLimits& lim,
                              std::atomic<bool>* stopFlag) {
    EndgameResult res;
    res.move = Move{ Action::Pass, Card{}, Card{}, false, {} };

    SearchCtx ctx;
    ctx.viewpoint = viewpoint;
    ctx.start = Clock::now();
    ctx.deadline = ctx.start + std::chrono::milliseconds(static_cast<long long>(lim.timeBudgetSec * 1000));
    if (stopFlag && stopFlag->load()) ctx.stopped = true;

    // Task 4: помечаем новую generation — старые записи в TT считаются устаревшими.
    ctx.tt.newGeneration();

    // Итеративное углубление: постепенно увеличиваем глубину, используем лучший
    // ход предыдущей итерации для упорядочивания. Это даёт «лучший за время».
    MoveBuffer rootBuf;
    int rootN = genLegalMoves(s, rootBuf);
    if (rootN == 0) {
        // Ходов нет — аномалия; вернём Pass.
        return res;
    }

    Move bestSoFar = rootBuf[0];
    int bestScoreSoFar = -20000;

    for (int depth = 1; depth <= lim.maxDepth; ++depth) {
        int alpha = -20000, beta = 20000;
        int bestThis = -20000;
        Move bestMoveThis = bestSoFar;

        // Task 4: упорядочиваем корневые ходы по эвристике + bestSoFar первым.
        // Это ускоряет сходимость — на каждой итерации отсечения срабатывают раньше.
        const Move* ttBestPtr = (depth > 1) ? &bestSoFar : nullptr;
        orderMovesWithTTBest(rootBuf, rootN, s, s.hands[toIdx(s.turn)], ttBestPtr);

        for (int i = 0; i < rootN; ++i) {
            if (ctx.timeUp()) break;
            MatchState child = s;
            if (!applyMove(child, rootBuf[i])) continue;
            int val = -negamax(child, depth - 1, -beta, -alpha, ctx);
            if (val > bestThis) {
                bestThis = val;
                bestMoveThis = rootBuf[i];
            }
            if (bestThis > alpha) alpha = bestThis;
        }

        if (ctx.timeUp() && depth > 1) break; // не используем неполную итерацию

        bestSoFar = bestMoveThis;
        bestScoreSoFar = bestThis;
        res.depthReached = depth;

        // Форсированный результат найден — можно остановиться.
        if (bestScoreSoFar > 5000 || bestScoreSoFar < -5000) {
            res.solved = true;
            res.score = (bestScoreSoFar > 0) ? 1 : -1;
            break;
        }
        if (bestScoreSoFar == 0 && depth >= 20) break; // похоже на ничью — выходим
    }

    if (stopFlag && stopFlag->load()) ctx.stopped = true;
    if (!ctx.stopped) {
        // Если не было таймаута и оценка недвусмысленная — отметим решённость.
        if (bestScoreSoFar > 5000)  { res.solved = true; res.score = 1; }
        if (bestScoreSoFar < -5000) { res.solved = true; res.score = -1; }
    }

    res.move = bestSoFar;
    res.nodes = ctx.nodes;
    res.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - ctx.start).count();
    return res;
}

} // namespace durakk
