const { engineAPI } = window;

// Состояние UI
let myHand = [];       // [{rank, suit}]
let table = [];        // [{attack: {rank,suit}, defense: {rank,suit}|null}]
let trumpSuit = 'clubs';
let deckCount = 24;
let oppCardCount = 6;
let phase = 'attack';
let attacker = 0;
let pairsLimit = 6;
let firstTrick = false;
let discard = [];

const SUITS = ['clubs', 'diamonds', 'hearts', 'spades'];
const SUIT_SYMBOLS = { clubs: '♣', diamonds: '♦', hearts: '♥', spades: '♠' };
const RANKS = [6, 7, 8, 9, 10, 11, 12, 13, 14];
const RANK_NAMES = { 6:'6', 7:'7', 8:'8', 9:'9', 10:'10', 11:'J', 12:'Q', 13:'K', 14:'A' };

function cardLabel(c) { return RANK_NAMES[c.rank] + SUIT_SYMBOLS[c.suit]; }

// === Рендер руки ===
function renderHand() {
    const el = document.getElementById('my-hand');
    el.innerHTML = '';
    for (let s of SUITS) {
        for (let r of RANKS) {
            const btn = document.createElement('button');
            btn.className = 'card-btn';
            btn.textContent = RANK_NAMES[r] + SUIT_SYMBOLS[s];
            const inHand = myHand.some(c => c.rank === r && c.suit === s);
            if (inHand) btn.classList.add('selected');
            btn.onclick = () => {
                if (inHand) myHand = myHand.filter(c => !(c.rank === r && c.suit === s));
                else myHand.push({ rank: r, suit: s });
                renderHand();
            };
            el.appendChild(btn);
        }
    }
}

// === Рендер стола ===
function renderTable() {
    const el = document.getElementById('table-pairs');
    el.innerHTML = '';
    table.forEach((pair, i) => {
        const div = document.createElement('div');
        div.className = 'table-pair';
        div.innerHTML = `
            <span class="atk">${cardLabel(pair.attack)}</span>
            <span class="def">${pair.defense ? cardLabel(pair.defense) : '—'}</span>
            <button onclick="removePair(${i})">✕</button>
        `;
        el.appendChild(div);
    });
}

function removePair(i) { table.splice(i, 1); renderTable(); }

// === Собрать состояние ===
function buildState() {
    return {
        myHand: myHand,
        oppHand: [],  // неизвестна
        oppCardCount: oppCardCount,
        table: table.map(p => ({
            attack: p.attack,
            defense: p.defense || { rank: -1, suit: 'clubs' }
        })),
        deckCount: deckCount,
        trumpSuit: trumpSuit,
        phase: phase,
        attacker: attacker,
        pairsLimit: pairsLimit,
        firstTrick: firstTrick,
        discard: discard
    };
}

// === Получить совет ===
async function getAdvice() {
    const statusEl = document.getElementById('status');
    const resultEl = document.getElementById('result');
    statusEl.textContent = '⏳ Двигаю... (MCTS + GPU, ~4 сек)';
    resultEl.textContent = '';

    try {
        const state = buildState();
        const resp = await engineAPI.send({ cmd: 'decide', state: state });

        if (resp.status === 'ok' && resp.move) {
            const m = resp.move;
            const label = cardLabel(m.card);
            const types = ['Атака', 'Защита', 'Беру', 'Бито', 'Перевод'];
            const typeName = types[m.type] || 'Ход';

            resultEl.innerHTML = `
                <div class="advice">
                    <strong>💡 ${typeName}: ${label}</strong>
                    ${resp.description ? `<br><small>${resp.description}</small>` : ''}
                </div>
            `;

            // Подсветить карту в руке
            document.querySelectorAll('.card-btn').forEach(btn => {
                btn.classList.remove('advice-highlight');
                if (btn.textContent === label) btn.classList.add('advice-highlight');
            });

            statusEl.textContent = '✅ Готово';
        } else {
            resultEl.textContent = 'Ошибка: ' + JSON.stringify(resp);
            statusEl.textContent = '❌ Ошибка';
        }
    } catch (e) {
        statusEl.textContent = '❌ ' + e.message;
    }
}

// === Легальные ходы ===
async function showLegalMoves() {
    const el = document.getElementById('legal-moves');
    try {
        const resp = await engineAPI.send({ cmd: 'legalMoves', state: buildState() });
        if (resp.moves) {
            el.innerHTML = resp.moves.map(m =>
                `<span class="legal">${cardLabel(m.card)}</span>`
            ).join(' ');
        }
    } catch (e) {
        el.textContent = 'Ошибка: ' + e.message;
    }
}

// === Привязка UI-элементов ===
function initUI() {
    renderHand();
    renderTable();

    // Козырь
    document.querySelectorAll('.trump-btn').forEach(btn => {
        btn.onclick = () => {
            trumpSuit = btn.dataset.suit;
            document.querySelectorAll('.trump-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        };
    });

    // Числовые поля
    document.getElementById('deck-count').oninput = e => { deckCount = parseInt(e.target.value) || 0; };
    document.getElementById('opp-count').oninput = e => { oppCardCount = parseInt(e.target.value) || 0; };

    // Фаза
    document.getElementById('phase-select').onchange = e => { phase = e.target.value; };

    // Атакующий
    document.getElementById('attacker-select').onchange = e => { attacker = parseInt(e.target.value); };

    // Первый кон
    document.getElementById('first-trick').onchange = e => {
        firstTrick = e.target.checked;
        pairsLimit = firstTrick ? 5 : 6;
    };

    // Добавить пару на стол
    document.getElementById('add-pair').onclick = () => {
        const atkInput = document.getElementById('atk-input').value.trim();
        const defInput = document.getElementById('def-input').value.trim();
        const atk = parseCardInput(atkInput);
        const def = defInput ? parseCardInput(defInput) : null;
        if (atk) {
            table.push({ attack: atk, defense: def });
            renderTable();
        }
    };

    // Кнопки
    document.getElementById('btn-advice').onclick = getAdvice;
    document.getElementById('btn-legal').onclick = showLegalMoves;
}

// Парсинг "7 clubs" или "K spades"
function parseCardInput(str) {
    if (!str) return null;
    const parts = str.toLowerCase().split(/\s+/);
    if (parts.length < 2) return null;
    let rank;
    const r = parts[0];
    if (r === 'j') rank = 11;
    else if (r === 'q') rank = 12;
    else if (r === 'k') rank = 13;
    else if (r === 'a') rank = 14;
    else rank = parseInt(r);
    const suit = parts[1];
    if (!SUITS.includes(suit) || rank < 6 || rank > 14) return null;
    return { rank, suit };
}

// Старт
document.addEventListener('DOMContentLoaded', initUI);
