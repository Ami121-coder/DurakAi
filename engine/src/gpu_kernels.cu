#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// ============================================================
// Bitboard-определения (дублируем для device-кода)
// ============================================================
__device__ __forceinline__ int d_rankOf(int idx) { return idx / 4 + 6; }
__device__ __forceinline__ int d_suitOf(int idx) { return idx % 4; }

__device__ __forceinline__ bool d_beats(int atkIdx, int defIdx, int trumpSuit) {
    int aRank = d_rankOf(atkIdx), aSuit = d_suitOf(atkIdx);
    int dRank = d_rankOf(defIdx), dSuit = d_suitOf(defIdx);
    bool aT = (aSuit == trumpSuit);
    bool dT = (dSuit == trumpSuit);
    if (dT && !aT) return true;
    if (dSuit == aSuit && dRank > aRank) return true;
    return false;
}

__device__ __forceinline__ int d_popcount(uint64_t m) {
    return __popcll(m);
}

__device__ __forceinline__ int d_popLowest(uint64_t& m) {
    int idx = __ffsll(m) - 1;
    m &= m - 1;
    return idx;
}

// ============================================================
// Структура состояния игры на GPU (64 байта, cache-friendly)
// ============================================================
struct GpuGameState {
    uint64_t hand[2];       // руки двух игроков (биты 0..35)
    uint64_t table_atk;     // карты атаки на столе
    uint64_t table_def;     // карты защиты на столе
    uint64_t deck;          // оставшаяся колода
    uint8_t  trump_suit;    // козырная масть
    uint8_t  phase;         // 0=attack, 1=defend
    uint8_t  attacker;      // 0 или 1 — кто атакует
    uint8_t  deck_count;    // карт в колоде
    uint8_t  table_pairs;   // пар на столе
    uint8_t  passed;        // 1 если "беру"
    uint8_t  padding[2];
};

// ============================================================
// Ядро: параллельные rollout'ы (playout'ы) до конца партии
// Каждый thread = одна полная симуляция
// ============================================================
__global__ void rolloutKernel(
    const GpuGameState* __restrict__ states,  // N состояний
    int* __restrict__ results,                 // N результатов: +1 win, -1 loss, 0 draw
    int N,
    int botPlayer,                             // за кого играем (0 или 1)
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // Инициализация RNG
    curandState rng;
    curand_init(seed, tid, 0, &rng);

    // Загружаем состояние в регистры
    uint64_t hand0 = states[tid].hand[0];
    uint64_t hand1 = states[tid].hand[1];
    uint64_t table_a = states[tid].table_atk;
    uint64_t table_d = states[tid].table_def;
    uint64_t deck = states[tid].deck;
    int trump = states[tid].trump_suit;
    int phase = states[tid].phase;
    int attacker = states[tid].attacker;
    int deckCnt = states[tid].deck_count;
    int pairs = states[tid].table_pairs;

    uint64_t trumpBits = 0;
    for (int r = 0; r < 9; ++r) trumpBits |= (1ULL << (r * 4 + trump));

    // Максимум 200 полуходов (защита от бесконечного цикла)
    for (int step = 0; step < 200; ++step) {
        int cur = (phase == 0) ? attacker : (1 - attacker);
        uint64_t& myHand = (cur == 0) ? hand0 : hand1;
        uint64_t& oppHand = (cur == 0) ? hand1 : hand0;

        if (d_popcount(myHand) == 0 && d_popcount(oppHand) == 0) break; // ничья

        if (phase == 0) {
            // АТАКА: выбираем случайную карту из руки
            if (d_popcount(myHand) == 0) {
                // Нет карт — победа
                break;
            }
            // Простая эвристика: бьём наименьшей не-козырной, или наименьшей
            uint64_t nonTrump = myHand & ~trumpBits;
            uint64_t pick;
            if (nonTrump && (curand(&rng) % 3 != 0)) {
                pick = nonTrump & (~nonTrump + 1); // lowest bit
            } else {
                pick = myHand & (~myHand + 1);
            }
            myHand &= ~pick;
            table_a |= pick;
            pairs++;
            phase = 1;
        } else {
            // ЗАЩИТА: пытаемся побить
            if (pairs == 0) { phase = 0; continue; }

            // Находим атаку без защиты
            uint64_t undefended = table_a & ~table_d;
            if (undefended == 0) {
                // Всё побито — "бито"
                table_a = 0; table_d = 0; pairs = 0;
                phase = 0;
                attacker = 1 - attacker;
                // Добор из колоды
                if (deckCnt > 0) {
                    // Упрощённый добор
                    int draw = (deckCnt >= 6) ? 6 : deckCnt;
                    for (int i = 0; i < draw && deckCnt > 0; ++i) {
                        int bit = d_popLowest(deck);
                        if (cur == 0) hand0 |= (1ULL << bit);
                        else hand1 |= (1ULL << bit);
                        deckCnt--;
                    }
                }
                continue;
            }

            int atkCard = d_popLowest(undefended);
            // Ищем чем побить
            uint64_t candidates = 0;
            uint64_t tmp = myHand;
            while (tmp) {
                int c = d_popLowest(tmp);
                if (d_beats(atkCard, c, trump)) candidates |= (1ULL << c);
            }

            if (candidates) {
                // Бьём наименьшей подходящей (предпочитаем не-козырь)
                uint64_t nonT = candidates & ~trumpBits;
                uint64_t pick = nonT ? (nonT & (~nonT + 1)) : (candidates & (~candidates + 1));
                myHand &= ~pick;
                table_d |= pick;
            } else {
                // "Беру" — забираем стол
                myHand |= table_a | table_d;
                table_a = 0; table_d = 0; pairs = 0;
                phase = 0;
                attacker = 1 - attacker;
                // Добор
                if (deckCnt > 0) {
                    int draw = (deckCnt >= 6) ? 6 : deckCnt;
                    for (int i = 0; i < draw && deckCnt > 0; ++i) {
                        int bit = d_popLowest(deck);
                        if (cur == 0) hand0 |= (1ULL << bit);
                        else hand1 |= (1ULL << bit);
                        deckCnt--;
                    }
                }
                continue;
            }
            // Проверяем: всё побито?
            if ((table_a & ~table_d) == 0) {
                table_a = 0; table_d = 0; pairs = 0;
                phase = 0;
                attacker = 1 - attacker;
                if (deckCnt > 0) {
                    int draw = (deckCnt >= 6) ? 6 : deckCnt;
                    for (int i = 0; i < draw && deckCnt > 0; ++i) {
                        int bit = d_popLowest(deck);
                        if (cur == 0) hand0 |= (1ULL << bit);
                        else hand1 |= (1ULL << bit);
                        deckCnt--;
                    }
                }
            }
        }
    }

    // Определяем результат для botPlayer
    uint64_t botH = (botPlayer == 0) ? hand0 : hand1;
    uint64_t oppH = (botPlayer == 0) ? hand1 : hand0;
    int botCnt = d_popcount(botH);
    int oppCnt = d_popcount(oppH);

    if (botCnt == 0 && oppCnt > 0) results[tid] = 1;
    else if (oppCnt == 0 && botCnt > 0) results[tid] = -1;
    else if (botCnt == 0 && oppCnt == 0) results[tid] = 0;
    else results[tid] = (botCnt < oppCnt) ? 1 : -1; // эвристика для незавершённых
}

// ============================================================
// Ядро: параллельная оценка позиций (для MCTS leaf evaluation)
// ============================================================
struct EvalParams {
    float w_hand_count;      // вес количества карт
    float w_trump_count;     // вес козырей
    float w_high_trump;      // вес старших козырей (A, K)
    float w_rank_spread;     // разнообразие рангов
    float w_low_cards;       // бонус за мелкие карты (легче отбиться)
    float w_opp_trumps;      // штраф за козыри соперника
    float w_deck_factor;     // влияние колоды
    float w_table_control;   // контроль стола
};

__device__ EvalParams d_params;

__device__ float d_evaluate(uint64_t myHand, uint64_t oppHand, int trump, int deckCnt) {
    uint64_t trumpBits = 0;
    for (int r = 0; r < 9; ++r) trumpBits |= (1ULL << (r * 4 + trump));

    float score = 0.0f;
    int myCnt = d_popcount(myHand);
    int oppCnt = d_popcount(oppHand);

    score += (float)(oppCnt - myCnt) * d_params.w_hand_count;

    int myTrumps = d_popcount(myHand & trumpBits);
    int oppTrumps = d_popcount(oppHand & trumpBits);
    score += (float)myTrumps * d_params.w_trump_count;
    score -= (float)oppTrumps * d_params.w_opp_trumps;

    // Старшие козыри (A=14, K=13)
    int aIdx = (14 - 6) * 4 + trump;
    int kIdx = (13 - 6) * 4 + trump;
    if (myHand & (1ULL << aIdx)) score += d_params.w_high_trump;
    if (myHand & (1ULL << kIdx)) score += d_params.w_high_trump * 0.7f;
    if (oppHand & (1ULL << aIdx)) score -= d_params.w_high_trump;
    if (oppHand & (1ULL << kIdx)) score -= d_params.w_high_trump * 0.7f;

    // Разнообразие рангов
    int ranks = 0;
    for (int r = 0; r < 9; ++r) {
        uint64_t rm = 0xFULL << (r * 4);
        if (myHand & rm) ranks++;
    }
    score += (float)ranks * d_params.w_rank_spread;

    // Мелкие карты (6,7,8) — бонус
    uint64_t lowMask = 0;
    for (int r = 0; r < 3; ++r) lowMask |= (0xFULL << (r * 4));
    score += (float)d_popcount(myHand & lowMask) * d_params.w_low_cards;

    // Фактор колоды
    score += (float)deckCnt * d_params.w_deck_factor * (myCnt > oppCnt ? 1.0f : -1.0f);

    return score;
}

// ============================================================
// Ядро: массовая оценка для MCTS (N позиций сразу)
// ============================================================
__global__ void evalKernel(
    const GpuGameState* __restrict__ states,
    float* __restrict__ scores,
    int N,
    int botPlayer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    uint64_t myH = states[tid].hand[botPlayer];
    uint64_t oppH = states[tid].hand[1 - botPlayer];
    int trump = states[tid].trump_suit;
    int deckCnt = states[tid].deck_count;

    scores[tid] = d_evaluate(myH, oppH, trump, deckCnt);
}

// ============================================================
// Ядро: параллельное семплирование рук соперника (curand)
// ============================================================
__global__ void sampleHandsKernel(
    uint64_t* __restrict__ sampledHands,  // выход: N рук
    const uint64_t knownCards,            // карты, которые точно НЕ у соперника
    const float* __restrict__ weights,    // веса [36] для каждой карты
    int handSize,                         // сколько карт у соперника
    int N,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandState rng;
    curand_init(seed, tid + 1000000, 0, &rng);

    // Weighted reservoir sampling
    uint64_t result = 0;
    int chosen = 0;

    // Собираем доступные карты
    uint64_t available = ~knownCards & ((1ULL << 36) - 1);

    // Простой weighted sampling без повторений
    float w[36];
    float totalW = 0.0f;
    for (int i = 0; i < 36; ++i) {
        if (available & (1ULL << i)) {
            w[i] = weights[i];
            totalW += weights[i];
        } else {
            w[i] = 0.0f;
        }
    }

    for (int c = 0; c < handSize && chosen < handSize; ++c) {
        if (totalW <= 0.0f) break;
        float r = curand_uniform(&rng) * totalW;
        float acc = 0.0f;
        for (int i = 0; i < 36; ++i) {
            if (w[i] <= 0.0f) continue;
            acc += w[i];
            if (acc >= r) {
                result |= (1ULL << i);
                totalW -= w[i];
                w[i] = 0.0f;
                chosen++;
                break;
            }
        }
    }

    sampledHands[tid] = result;
}

// ============================================================
// Ядро: редукция результатов (подсчёт win/loss/draw)
// ============================================================
__global__ void reduceResults(
    const int* __restrict__ results,
    int* __restrict__ wins,
    int* __restrict__ losses,
    int* __restrict__ draws,
    int N
) {
    __shared__ int s_wins[256];
    __shared__ int s_losses[256];
    __shared__ int s_draws[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    s_wins[tid] = 0; s_losses[tid] = 0; s_draws[tid] = 0;

    for (int i = gid; i < N; i += blockDim.x * gridDim.x) {
        if (results[i] > 0) s_wins[tid]++;
        else if (results[i] < 0) s_losses[tid]++;
        else s_draws[tid]++;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_wins[tid] += s_wins[tid + s];
            s_losses[tid] += s_losses[tid + s];
            s_draws[tid] += s_draws[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(wins, s_wins[0]);
        atomicAdd(losses, s_losses[0]);
        atomicAdd(draws, s_draws[0]);
    }
}
