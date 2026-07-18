'use strict';

// Логика UI: построение сетки карт, сборка состояния игры из формы, вызовы бота.
// Никакого Node.js — только window.bot из preload.

const RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
const SUITS = ['clubs', 'diamonds', 'hearts', 'spades'];
const SUIT_SYM = { clubs: '♣', diamonds: '♦', hearts: '♥', spades: '♠' };
const SUIT_COLOR = { clubs: 'black', diamonds: 'red', hearts: 'red', spades: 'black' };

const state = {
    trump: 'spades',
    hand: new Set(),          // набор "rank|suit"
    table: [],                // [{attack:'rank|suit'|null, defense:'rank|suit'|null}]
    adviceCards: new Set(),   // подсветка совета
    discard: new Set(),       // карты в бите (память бота)
    oppTaken: new Set(),      // карты, которые соперник забрал
    strength: 'normal',       // fast | normal | deep
};

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);

// ---------- Утилиты ----------
function cardKey(rank, suit) { return rank + '|' + suit; }
function splitKey(k) { const [r, s] = k.split('|'); return { rank: r, suit: s }; }

function setEngineStatus(info) {
    const el = $('engineStatus');
    if (!info) return;
    if (info.ok) {
        el.textContent = 'движок: готов ✓';
        el.className = 'badge badge--ok';
    } else if (info.reason === 'exe-not-found') {
        el.textContent = 'движок: не собран (npm run build:engine)';
        el.className = 'badge badge--err';
    } else if (info.reason === 'exit') {
        el.textContent = `движок: упал (code=${info.code})`;
        el.className = 'badge badge--err';
    } else {
        el.textContent = 'движок: ' + (info.reason || '?');
        el.className = 'badge badge--err';
    }
}

// ---------- Сетка руки ----------
function renderHand() {
    const grid = $('handGrid');
    grid.innerHTML = '';
    for (const suit of SUITS) {
        for (const rank of RANKS) {
            const k = cardKey(rank, suit);
            const card = document.createElement('div');
            card.className = 'card' + (SUIT_COLOR[suit] === 'red' ? ' red' : '');
            if (state.hand.has(k))            card.classList.add('is-selected');
            if (state.adviceCards.has(k))     card.classList.add('is-advice');

            card.innerHTML = `<div class="rank">${rank}</div><div class="suit">${SUIT_SYM[suit]}</div>`;
            card.addEventListener('click', () => {
                if (state.hand.has(k)) state.hand.delete(k);
                else state.hand.add(k);
                renderHand();
            });
            grid.appendChild(card);
        }
    }
}

// ---------- Козырь ----------
function renderTrump() {
    const picker = $('trumpPicker');
    picker.innerHTML = '';
    for (const s of SUITS) {
        const b = document.createElement('button');
        b.dataset.suit = s;
        b.textContent = SUIT_SYM[s];
        if (s === state.trump) b.classList.add('is-active');
        b.addEventListener('click', () => {
            state.trump = s;
            renderTrump();
        });
        picker.appendChild(b);
    }
}

// ---------- Стол ----------
function renderTable() {
    const grid = $('tableGrid');
    grid.innerHTML = '';
    state.table.forEach((pair, idx) => {
        const box = document.createElement('div');
        box.className = 'pair';

        const atk = pair.attack ? splitKey(pair.attack) : null;
        const def = pair.defense ? splitKey(pair.defense) : null;

        box.appendChild(miniCard(atk, 'атака', () => pickForTable(idx, 'attack')));
        const vs = document.createElement('span');
        vs.className = 'vs';
        vs.textContent = '←';
        box.appendChild(vs);
        box.appendChild(miniCard(def, 'защита', () => pickForTable(idx, 'defense')));

        const del = document.createElement('button');
        del.className = 'del';
        del.textContent = '✕';
        del.title = 'убрать пару';
        del.addEventListener('click', () => {
            state.table.splice(idx, 1);
            renderTable();
        });
        box.appendChild(del);

        grid.appendChild(box);
    });
}

function miniCard(card, placeholder, onclick) {
    const m = document.createElement('div');
    m.className = 'mini' + (card ? '' : ' empty');
    if (card) {
        m.innerHTML = `<div class="rank ${SUIT_COLOR[card.suit] === 'red' ? 'red' : ''}">${card.rank}</div><div>${SUIT_SYM[card.suit]}</div>`;
    } else {
        m.textContent = placeholder;
    }
    m.addEventListener('click', onclick);
    return m;
}

// Простой prompt выбора карты для ячейки стола (через window.prompt).
function pickForTable(idx, slot) {
    const input = window.prompt(
        `Введите карту для «${slot === 'attack' ? 'атаки' : 'защиты'}» в формате РАНК МАСТЬ\n` +
        `Ранг: ${RANKS.join(', ')}\nМасть: clubs/diamonds/hearts/spades`,
        ''
    );
    if (!input) return;
    const parts = input.trim().split(/\s+/);
    if (parts.length !== 2) return alert('Формат: РАНГ МАСТЬ (через пробел)');
    const [rank, suit] = parts;
    if (!RANKS.includes(rank) || !SUITS.includes(suit)) {
        return alert('Неверный ранг или масть');
    }
    state.table[idx][slot] = cardKey(rank, suit);
    renderTable();
}

// ---------- Сборка состояния для движка ----------
function buildState() {
    const table = state.table.map((p) => {
        const o = {};
        if (p.attack) {
            const c = splitKey(p.attack);
            o.attack = { r: c.rank, s: c.suit };
        }
        if (p.defense) {
            const c = splitKey(p.defense);
            o.defense = { r: c.rank, s: c.suit };
        }
        return o;
    }).filter((p) => p.attack);

    return {
        trump: state.trump,
        deckSize: 36,
        transferEnabled: $('transferEnabled').checked,
        flashEnabled: $('flashEnabled').checked,
        firstTrickLimit: 5,
        deckCount: parseInt($('deckCount').value, 10) || 0,
        oppHandCount: parseInt($('oppHandCount').value, 10) || 0,
        myHand: [...state.hand].map((k) => { const c = splitKey(k); return { r: c.rank, s: c.suit }; }),
        discard: [...state.discard].map((k) => { const c = splitKey(k); return { r: c.rank, s: c.suit }; }),
        oppTaken: [...state.oppTaken].map((k) => { const c = splitKey(k); return { r: c.rank, s: c.suit }; }),
        table,
        attacker: $('attacker').value,
        turn: $('turn').value,
        phase: $('phase').value,
        firstTrick: $('firstTrick').checked,
    };
}

// ---------- Запросы к боту ----------
function showAdvice(html, cls) {
    const el = $('advice');
    el.className = 'advice ' + (cls || '');
    el.innerHTML = html;
}

function renderStats(st) {
    if (!st) return '';
    const parts = [`<b>${st.mode}</b>`];
    if (st.playouts) parts.push(`${st.playouts} плейаутов`);
    if (st.winrate !== undefined) parts.push(`winrate ${(st.winrate * 100).toFixed(0)}%`);
    if (st.depthReached) parts.push(`depth ${st.depthReached}`);
    if (st.solved) parts.push(`<b style="color:var(--accent-2)">решён</b>`);
    if (st.timeMs) parts.push(`${(st.timeMs / 1000).toFixed(1)}s`);
    return `<div class="stats">${parts.join(' · ')}</div>`;
}

function renderMemGrid(gridId, set, cssClass) {
    const grid = $(gridId);
    grid.innerHTML = '';
    for (const suit of SUITS) {
        for (const rank of RANKS) {
            const k = cardKey(rank, suit);
            const card = document.createElement('div');
            card.className = 'card' + (SUIT_COLOR[suit] === 'red' ? ' red' : '');
            if (set.has(k)) card.classList.add('is-selected', cssClass);
            card.innerHTML = `<div class="rank">${rank}</div><div class="suit">${SUIT_SYM[suit]}</div>`;
            card.addEventListener('click', () => {
                if (set.has(k)) set.delete(k); else set.add(k);
                renderMemGrid(gridId, set, cssClass);
            });
            grid.appendChild(card);
        }
    }
}

function renderMoveCard(move) {
    if (!move || (!move.rank && !move.suit)) return '';
    return `<span class="a-card">${move.rank}${SUIT_SYM[move.suit] || ''}</span>`;
}

const ACTION_LABEL = {
    attack: 'Атаковать', defend: 'Защищаться', transfer: 'Перевести',
    toss: 'Подкинуть', take: 'Взять карты', done: 'Бито', pass: 'Пропустить',
};

async function onDecide() {
    const st = buildState();
    $('lastRequest').textContent = JSON.stringify(st, null, 2);
    state.adviceCards.clear();
    showAdvice('Думаю…', 'advice--empty');

    const res = await window.bot.decide(st, { strength: state.strength });
    $('lastResponse').textContent = JSON.stringify(res, null, 2);

    if (!res.ok) {
        showAdvice(`Ошибка: ${res.error}`, 'advice--err');
        return;
    }
    const move = res.move;
    if (move.rank && move.suit) {
        state.adviceCards.add(cardKey(move.rank, move.suit));
    }
    renderHand();

    const card = renderMoveCard(move);
    const reason = move.reason ? `<div class="a-reason">${move.reason}</div>` : '';
    const stats = res.stats ? renderStats(res.stats) : '';
    showAdvice(
        `<div class="a-title">${ACTION_LABEL[move.action] || move.action}</div>${card}${reason}${stats}`,
        'advice--ok'
    );
}

async function onLegal() {
    const st = buildState();
    const res = await window.bot.legalMoves(st);
    const box = $('legalList');
    if (!res.ok) {
        box.innerHTML = `<span style="color:var(--danger)">${res.error}</span>`;
        return;
    }
    const m = res.moves;
    const fmt = (arr) => arr.length
        ? arr.map((c) => `${c.r}${SUIT_SYM[c.s]}`).join(' ')
        : '—';
    const fmtDef = (arr) => arr.length
        ? arr.map((d) => `${d.card.r}${SUIT_SYM[d.card.s]}→${d.target.r}${SUIT_SYM[d.target.s]}`).join('  ')
        : '—';
    box.innerHTML = `
        <div class="group"><b>Атака/подкидывание:</b> ${fmt(m.attacks)}</div>
        <div class="group"><b>Подкидывание:</b> ${fmt(m.tosses)}</div>
        <div class="group"><b>Перевод:</b> ${fmt(m.transfers)}</div>
        <div class="group"><b>Защита:</b> ${fmtDef(m.defends)}</div>
        <div class="group"><b>Можно «бито»:</b> ${m.canDone ? 'да' : 'нет'}</div>
    `;
}

function onReset() {
    state.hand.clear();
    state.table = [];
    state.adviceCards.clear();
    state.discard.clear();
    state.oppTaken.clear();
    renderHand();
    renderTable();
    renderMemGrid('discardGrid', state.discard, 'is-discard');
    renderMemGrid('oppTakenGrid', state.oppTaken, 'is-taken');
    showAdvice('Заполните состояние и нажмите «Получить совет».', 'advice--empty');
    $('legalList').textContent = '—';
}

// ---------- Старт ----------
function init() {
    renderHand();
    renderTrump();
    renderTable();
    renderMemGrid('discardGrid', state.discard, 'is-discard');
    renderMemGrid('oppTakenGrid', state.oppTaken, 'is-taken');

    // Селектор силы.
    document.querySelectorAll('.strength-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.strength-btn').forEach((b) => b.classList.remove('strength-btn--active'));
            btn.classList.add('strength-btn--active');
            state.strength = btn.dataset.strength;
        });
    });

    $('addTablePair').addEventListener('click', () => {
        state.table.push({ attack: null, defense: null });
        renderTable();
    });
    $('clearTable').addEventListener('click', () => { state.table = []; renderTable(); });
    $('btnDecide').addEventListener('click', onDecide);
    $('btnLegal').addEventListener('click', onLegal);
    $('btnReset').addEventListener('click', onReset);

    window.bot.onStatus(setEngineStatus);
    window.bot.status().then((s) => {
        setEngineStatus(s.available ? { ok: true, path: s.path } : { ok: false, reason: 'exe-not-found' });
    });
}

document.addEventListener('DOMContentLoaded', init);
