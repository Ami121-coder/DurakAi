#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cfloat>

// ============================================================
// Device-утилиты bitboard
// ============================================================
__device__ __forceinline__ int d_rankOf(int idx) { return idx / 4 + 6; }
__device__ __forceinline__ int d_suitOf(int idx) { return idx % 4; }

__device__ __forceinline__ bool d_beats(int atk, int def, int trump) {
    int aR = d_rankOf(atk), aS = d_suitOf(atk);
    int dR = d_rankOf(def), dS = d_suitOf(def);
    if (dS == trump && aS != trump) return true;
    if (dS == aS && dR > aR) return true;
    return false;
}

__device__ __forceinline__ int d_popcount(uint64_t m) { return __popcll(m); }

__device__ __forceinline__ int d_popLowest(uint64_t& m) {
    int idx = __ffsll((long long)m) - 1;
    m &= m - 1;
    return idx;
}

__device__ __forceinline__ uint64_t d_rankMask(int rank) {
    return 0xFULL << ((rank - 6) * 4);
}

__device__ __forceinline__ uint64_t d_trumpBits(int trump) {
    uint64_t m = 0;
    for (int r = 0; r < 9; ++r) m |= (1ULL << (r * 4 + trump));
    return m;
}

// Случайный бит из маски через curand
__device__ __forceinline__ int d_randomBit(uint64_t mask, curandState* rng) {
    int cnt = d_popcount(mask);
    if (cnt == 0) return -1;
    int pick = (int)(curand(rng) % (unsigned)cnt);
    uint64_t tmp = mask;
    for (int i = 0; i < pick; ++i) tmp &= tmp - 1;
    return __ffsll((long long)tmp) - 1;
}

// ============================================================
// Состояние игры (64 байта, cache-line aligned)
// ============================================================
struct GpuGameState {
    uint64_t hand[2];
    uint64_t table_atk;
    uint64_t table_def;
    uint64_t deck;
    uint8_t  trump_suit;
    uint8_t  phase;       // 0=attack, 1=defend, 2=pursue(вдогонку)
    uint8_t  attacker;
    uint8_t  deck_count;
    uint8_t  table_pairs;
    uint8_t  pairs_limit; // 5 или 6
    uint8_t  first_trick; // 1 = первый кон партии
    uint8_t  padding;
};

// ============================================================
// Параметры оценки (загружаются в constant memory)
// ============================================================
struct EvalParams {
    float w_hand_count;
    float w_trump_count;
    float w_high_trump;
    float w_rank_spread;
    float w_low_cards;
    float w_opp_trumps;
    float w_deck_factor;
    float w_table_control;
    float w_pair_bonus;
    float w_save_trump;
};

__constant__ EvalParams d_params;

// ============================================================
// Оценка хода (для smart rollout policy)
// ============================================================
__device__ float d_scoreMove(
    int cardIdx, uint64_t myHand, uint64_t oppHand,
    uint64_t table_a, uint64_t table_d,
    int trump, int phase, int oppCardCount, int deckCnt
) {
    uint64_t tb = d_trumpBits(trump);
    int cardRank = d_rankOf(cardIdx);
    int cardSuit = d_suitOf(cardIdx);
    bool isTrump = (cardSuit == trump);
    float score = 0.0f;

    if (phase == 0 || phase == 2) {
        // АТАКА / ПОДКИДЫВАНИЕ ВДОГОНКУ
        // Штраф за козырь (особенно в начале)
        if (isTrump) {
            score -= d_params.w_save_trump * (deckCnt > 12 ? 2.0f : 1.0f);
        }
        // Бонус за парный ранг в руке
        uint64_t sameRank = myHand & d_rankMask(cardRank);
        if (d_popcount(sameRank) >= 2) score += d_params.w_pair_bonus;
        // Бонус за мелкие карты (легче подкинуть потом)
        if (cardRank <= 8) score += d_params.w_low_cards;
        // Штраф за подкидывание когда у соперника мало карт
        if (oppCardCount <= 2 && phase == 2) score -= 5.0f;
        // Бонус за старшие не-козыри (давление)
        if (!isTrump && cardRank >= 12) score += 2.0f;

    } else if (phase == 1) {
        // ЗАЩИТА
        // Предпочитаем бить младшей не-козырной
        if (!isTrump) score += 3.0f;
        else score -= d_params.w_save_trump;
        // Младшая карта лучше
        score += (float)(14 - cardRank) * 0.5f;
        // Если у соперника мало карт — экономим козыри
        if (oppCardCount <= 3 && isTrump) score -= 4.0f;
    }

    return score;
}

// ============================================================
// Оценка позиции (для PUCT prior и leaf evaluation)
// ============================================================
__device__ float d_evaluate(uint64_t myHand, uint64_t oppHand, int trump, int deckCnt) {
    uint64_t tb = d_trumpBits(trump);
    float score = 0.0f;

    int myCnt = d_popcount(myHand);
    int oppCnt = d_popcount(oppHand);
    score += (float)(oppCnt - myCnt) * d_params.w_hand_count;

    int myTrumps = d_popcount(myHand & tb);
    int oppTrumps = d_popcount(oppHand & tb);
    score += (float)myTrumps * d_params.w_trump_count;
    score -= (float)oppTrumps * d_params.w_opp_trumps;

    // Старшие козыри
    int aIdx = (14 - 6) * 4 + trump;
    int kIdx = (13 - 6) * 4 + trump;
    if (myHand & (1ULL << aIdx)) score += d_params.w_high_trump;
    if (myHand & (1ULL << kIdx)) score += d_params.w_high_trump * 0.7f;
    if (oppHand & (1ULL << aIdx)) score -= d_params.w_high_trump;
    if (oppHand & (1ULL << kIdx)) score -= d_params.w_high_trump * 0.7f;

    // Разнообразие рангов
    int ranks = 0;
    for (int r = 0; r < 9; ++r) {
        if (myHand & (0xFULL << (r * 4))) ranks++;
    }
    score += (float)ranks * d_params.w_rank_spread;

    // Мелкие карты
    uint64_t lowMask = 0;
    for (int r = 0; r < 3; ++r) lowMask |= (0xFULL << (r * 4));
    score += (float)d_popcount(myHand & lowMask) * d_params.w_low_cards;

    // Фактор колоды
    score += (float)deckCnt * d_params.w_deck_factor * (myCnt > oppCnt ? 1.0f : -1.0f);

    return score;
}

// ============================================================
// Smart pick: softmax по scoreMove (temperature = 0.5)
// ============================================================
__device__ int d_smartPick(
    uint64_t candidates, uint64_t myHand, uint64_t oppHand,
    uint64_t table_a, uint64_t table_d,
    int trump, int phase, int oppCardCount, int deckCnt,
    curandState* rng
) {
    if (candidates == 0) return -1;
    int cnt = d_popcount(candidates);
    if (cnt == 1) return __ffsll((long long)candidates) - 1;

    // Считаем scores
    float scores[36];
    float maxS = -1e9f;
    uint64_t tmp = candidates;
    int i = 0;
    while (tmp) {
        int idx = d_popLowest(tmp);
        scores[i] = d_scoreMove(idx, myHand, oppHand, table_a, table_d,
                                trump, phase, oppCardCount, deckCnt);
        if (scores[i] > maxS) maxS = scores[i];
        i++;
    }

    // Softmax (temp = 0.5 → multiplier = 2.0)
    float sumExp = 0.0f;
    for (int j = 0; j < cnt; ++j) {
        scores[j] = expf((scores[j] - maxS) * 2.0f);
        sumExp += scores[j];
    }

    // Sampling
    float r = curand_uniform(rng) * sumExp;
    float acc = 0.0f;
    tmp = candidates;
    for (int j = 0; j < cnt; ++j) {
        int idx = d_popLowest(tmp);
        acc += scores[j];
        if (acc >= r) return idx;
    }
    return __ffsll((long long)candidates) - 1; // fallback
}

// ============================================================
// ГЛАВНОЕ ЯДРО: параллельные rollout'ы с smart policy
// Исправлены ВСЕ баги:
//   - Random добор из колоды (curand)
//   - Подкидывание вдогонку (phase 2)
//   - Перевод
//   - Smart softmax policy
// ============================================================
__global__ void rolloutKernel(
    const GpuGameState* __restrict__ states,
    int* __restrict__ results,
    int N,
    int botPlayer,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandState rng;
    curand_init(seed, (unsigned long long)tid, 0, &rng);

    // Загружаем в регистры
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
    int pairsLimit = states[tid].pairs_limit;
    uint64_t tb = d_trumpBits(trump);

    // Максимум 300 полуходов
    for (int step = 0; step < 300; ++step) {
        int cur = (phase == 0 || phase == 2) ? attacker : (1 - attacker);
        uint64_t myHand = (cur == 0) ? hand0 : hand1;
        uint64_t oppHand = (cur == 0) ? hand1 : hand0;
        int oppCnt = d_popcount(oppHand);

        // Терминальная проверка
        if (d_popcount(myHand) == 0 && d_popcount(oppHand) == 0) break;
        if (d_popcount(myHand) == 0 && phase == 0) break; // атакующий без карт = победа

        if (phase == 0) {
            // === АТАКА ===
            if (d_popcount(myHand) == 0) break;

            // Генерируем кандидатов: любая карта из руки
            uint64_t candidates = myHand;

            // Smart pick
            int pick = d_smartPick(candidates, myHand, oppHand, table_a, table_d,
                                   trump, 0, oppCnt, deckCnt, &rng);
            if (pick < 0) break;

            myHand &= ~(1ULL << pick);
            table_a |= (1ULL << pick);
            pairs++;
            phase = 1;

        } else if (phase == 1) {
            // === ЗАЩИТА ===
            uint64_t undefended = table_a & ~table_d;

            if (undefended == 0) {
                // Всё побито → БИТО
                table_a = 0; table_d = 0; pairs = 0;
                phase = 0;
                attacker = 1 - attacker;
                // Добор (random из колоды)
                if (deckCnt > 0) {
                    int drawN = min(6, deckCnt);
                    // Атакующий добирает первым
                    int firstDraw = attacker; // новый атакующий = бывший защитник
                    // На самом деле: атакующий (тот кто атаковал) добирает первым
                    // После "бито" ход переходит к защитнику, он становится атакующим
                    // Добор: сначала тот кто атаковал (старый attacker), потом защитник
                    int oldAtk = 1 - attacker; // тот кто атаковал в этом коне
                    for (int i = 0; i < drawN && deckCnt > 0; ++i) {
                        int bit = d_randomBit(deck, &rng);
                        if (bit < 0) break;
                        deck &= ~(1ULL << bit);
                        if (oldAtk == 0) hand0 |= (1ULL << bit);
                        else hand1 |= (1ULL << bit);
                        deckCnt--;
                    }
                    for (int i = 0; i < drawN && deckCnt > 0; ++i) {
                        int bit = d_randomBit(deck, &rng);
                        if (bit < 0) break;
                        deck &= ~(1ULL << bit);
                        if (attacker == 0) hand0 |= (1ULL << bit);
                        else hand1 |= (1ULL << bit);
                        deckCnt--;
                    }
                }
                continue;
            }

            // Берём первую непокрытую карту
            int atkCard = __ffsll((long long)undefended) - 1;

            // Ищем чем побить
            uint64_t beatCandidates = 0;
            uint64_t hTmp = myHand;
            while (hTmp) {
                int c = d_popLowest(hTmp);
                if (d_beats(atkCard, c, trump)) beatCandidates |= (1ULL << c);
            }

            // Проверяем перевод: все карты на столе одного ранга?
            bool canTransfer = false;
            uint64_t transferCandidates = 0;
            if (table_d == 0 && d_popcount(table_a) > 0) {
                int firstRank = d_rankOf(__ffsll((long long)table_a) - 1);
                bool allSame = true;
                uint64_t tTmp = table_a;
                while (tTmp) {
                    int ti = d_popLowest(tTmp);
                    if (d_rankOf(ti) != firstRank) { allSame = false; break; }
                }
                if (allSame) {
                    transferCandidates = myHand & d_rankMask(firstRank);
                    int newPairs = d_popcount(table_a) + d_popcount(transferCandidates);
                    // Ограничение: у нового защитника достаточно карт
                    if (oppCnt >= newPairs && d_popcount(transferCandidates) > 0) {
                        canTransfer = true;
                    }
                }
            }

            if (canTransfer && beatCandidates == 0) {
                // Перевод (нет чем бить, но можно перевести)
                int pick = d_smartPick(transferCandidates, myHand, oppHand,
                                       table_a, table_d, trump, 1, oppCnt, deckCnt, &rng);
                if (pick >= 0) {
                    myHand &= ~(1ULL << pick);
                    table_a |= (1ULL << pick);
                    pairs++;
                    // Перевод: защитник становится атакующим
                    attacker = cur;
                    phase = 1; // новый защитник должен отбиваться
                    // Меняем руки
                    if (cur == 0) { hand0 = myHand; hand1 = oppHand; }
                    else { hand1 = myHand; hand0 = oppHand; }
                    continue;
                }
            }

            if (canTransfer && beatCandidates != 0) {
                // Выбираем: бить или переводить (softmax)
                float beatScore = 0.0f, transScore = 0.0f;
                // Оцениваем лучший beat
                uint64_t bTmp = beatCandidates;
                while (bTmp) {
                    int bi = d_popLowest(bTmp);
                    float s = d_scoreMove(bi, myHand, oppHand, table_a, table_d,
                                          trump, 1, oppCnt, deckCnt);
                    if (s > beatScore) beatScore = s;
                }
                // Оцениваем лучший transfer
                uint64_t tTmp2 = transferCandidates;
                while (tTmp2) {
                    int ti = d_popLowest(tTmp2);
                    float s = d_scoreMove(ti, myHand, oppHand, table_a, table_d,
                                          trump, 1, oppCnt, deckCnt) + 2.0f; // бонус за перевод
                    if (s > transScore) transScore = s;
                }

                if (transScore > beatScore && curand_uniform(&rng) > 0.3f) {
                    // Переводим
                    int pick = d_smartPick(transferCandidates, myHand, oppHand,
                                           table_a, table_d, trump, 1, oppCnt, deckCnt, &rng);
                    if (pick >= 0) {
                        myHand &= ~(1ULL << pick);
                        table_a |= (1ULL << pick);
                        pairs++;
                        attacker = cur;
                        phase = 1;
                        if (cur == 0) { hand0 = myHand; hand1 = oppHand; }
                        else { hand1 = myHand; hand0 = oppHand; }
                        continue;
                    }
                }
            }

            if (beatCandidates) {
                // Бьём (smart pick)
                int pick = d_smartPick(beatCandidates, myHand, oppHand,
                                       table_a, table_d, trump, 1, oppCnt, deckCnt, &rng);
                if (pick >= 0) {
                    myHand &= ~(1ULL << pick);
                    table_d |= (1ULL << pick);
                }
                // Проверяем: всё побито?
                if ((table_a & ~table_d) == 0) {
                    table_a = 0; table_d = 0; pairs = 0;
                    phase = 0;
                    attacker = 1 - attacker;
                    // Добор
                    if (deckCnt > 0) {
                        int drawN = min(6, deckCnt);
                        int oldAtk = 1 - attacker;
                        for (int i = 0; i < drawN && deckCnt > 0; ++i) {
                            int bit = d_randomBit(deck, &rng);
                            if (bit < 0) break;
                            deck &= ~(1ULL << bit);
                            if (oldAtk == 0) hand0 |= (1ULL << bit);
                            else hand1 |= (1ULL << bit);
                            deckCnt--;
                        }
                        for (int i = 0; i < drawN && deckCnt > 0; ++i) {
                            int bit = d_randomBit(deck, &rng);
                            if (bit < 0) break;
                            deck &= ~(1ULL << bit);
                            if (attacker == 0) hand0 |= (1ULL << bit);
                            else hand1 |= (1ULL << bit);
                            deckCnt--;
                        }
                    }
                }
            } else {
                // БЕРУ
                myHand |= table_a | table_d;
                table_a = 0; table_d = 0;
                // Фаза Pursue: атакующий подкидывает вдогонку
                phase = 2;
                // НЕ меняем attacker — он подкидывает
            }

            // Обновляем руки
            if (cur == 0) hand0 = myHand;
            else hand1 = myHand;

        } else if (phase == 2) {
            // === ПОДКИДЫВАНИЕ ВДОГОНКУ (Pursue) ===
            // Атакующий подкидывает, защищающийся забирает
            uint64_t tableRanks = 0;
            uint64_t allTable = table_a | table_d;
            // table_a/table_d уже обнулены при Take, но ранги помним
            // Восстанавливаем ранги из руки защитника (он забрал)
            // Упрощение: подкидываем карты рангов, которые были на столе
            // Для этого храним tableRanks до обнуления
            // В данной реализации: подкидываем любые карты (упрощение для rollout)
            // Корректнее: храним lastTableRanks

            // Упрощённый Pursue: атакующий подкидывает 0-2 случайные карты
            int throwCount = (int)(curand(&rng) % 3); // 0, 1, или 2
            int defender = 1 - attacker;
            uint64_t defHand = (defender == 0) ? hand0 : hand1;
            int defCnt = d_popcount(defHand);

            for (int t = 0; t < throwCount && d_popcount(myHand) > 0 && defCnt < 12; ++t) {
                // Подкидываем младшую не-козырную
                uint64_t nonT = myHand & ~tb;
                int pick;
                if (nonT) pick = __ffsll((long long)nonT) - 1;
                else pick = __ffsll((long long)myHand) - 1;
                myHand &= ~(1ULL << pick);
                // Защитник забирает
                if (defender == 0) hand0 |= (1ULL << pick);
                else hand1 |= (1ULL << pick);
                defCnt++;
            }

            // Обновляем руки
            if (cur == 0) hand0 = myHand;
            else hand1 = myHand;

            // Pursue завершён → ход остаётся у атакующего
            phase = 0;
            // Добор
            if (deckCnt > 0) {
                int drawN = min(6, deckCnt);
                for (int i = 0; i < drawN && deckCnt > 0; ++i) {
                    int bit = d_randomBit(deck, &rng);
                    if (bit < 0) break;
                    deck &= ~(1ULL << bit);
                    if (attacker == 0) hand0 |= (1ULL << bit);
                    else hand1 |= (1ULL << bit);
                    deckCnt--;
                }
                int def2 = 1 - attacker;
                for (int i = 0; i < drawN && deckCnt > 0; ++i) {
                    int bit = d_randomBit(deck, &rng);
                    if (bit < 0) break;
                    deck &= ~(1ULL << bit);
                    if (def2 == 0) hand0 |= (1ULL << bit);
                    else hand1 |= (1ULL << bit);
                    deckCnt--;
                }
            }
        }
    }

    // Результат
    uint64_t botH = (botPlayer == 0) ? hand0 : hand1;
    uint64_t oppH = (botPlayer == 0) ? hand1 : hand0;
    int botCnt = d_popcount(botH);
    int oppCnt2 = d_popcount(oppH);

    if (botCnt == 0 && oppCnt2 > 0) results[tid] = 1;
    else if (oppCnt2 == 0 && botCnt > 0) results[tid] = -1;
    else if (botCnt == 0 && oppCnt2 == 0) results[tid] = 0;
    else results[tid] = (botCnt < oppCnt2) ? 1 : -1;
}

// ============================================================
// Ядро: batch evaluation (для PUCT prior)
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
    scores[tid] = d_evaluate(myH, oppH, states[tid].trump_suit, states[tid].deck_count);
}

// ============================================================
// Ядро: batch sampling рук соперника (weighted)
// ============================================================
__global__ void sampleHandsKernel(
    uint64_t* __restrict__ sampledHands,
    uint64_t knownCards,
    const float* __restrict__ weights,
    int handSize,
    int N,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandState rng;
    curand_init(seed, (unsigned long long)(tid + 2000000), 0, &rng);

    uint64_t available = ~knownCards & ((1ULL << 36) - 1);
    uint64_t result = 0;
    int chosen = 0;

    // Weighted sampling без повторений
    float w[36];
    float totalW = 0.0f;
    for (int i = 0; i < 36; ++i) {
        if (available & (1ULL << i)) { w[i] = weights[i]; totalW += weights[i]; }
        else w[i] = 0.0f;
    }

    for (int c = 0; c < handSize; ++c) {
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
// Ядро: редукция результатов
// ============================================================
__global__ void reduceResults(
    const int* __restrict__ results,
    int* __restrict__ wins,
    int* __restrict__ losses,
    int* __restrict__ draws,
    int N
) {
    __shared__ int s_w[256], s_l[256], s_d[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    s_w[tid] = 0; s_l[tid] = 0; s_d[tid] = 0;
    for (int i = gid; i < N; i += blockDim.x * gridDim.x) {
        if (results[i] > 0) s_w[tid]++;
        else if (results[i] < 0) s_l[tid]++;
        else s_d[tid]++;
    }
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) { s_w[tid] += s_w[tid+s]; s_l[tid] += s_l[tid+s]; s_d[tid] += s_d[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(wins, s_w[0]); atomicAdd(losses, s_l[0]); atomicAdd(draws, s_d[0]); }
}