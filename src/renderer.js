// ============================================================
//  Дурак AI — трекер партии + авто-советник
//  Колода / сброс / фаза / роль считаются из лога действий.
// ============================================================
const { engineAPI } = window;

const SUITS = ['clubs', 'diamonds', 'hearts', 'spades'];
const SUIT_SYM = { clubs: '♣', diamonds: '♦', hearts: '♥', spades: '♠' };
const SUIT_NAME = { 0: 'clubs', 1: 'diamonds', 2: 'hearts', 3: 'spades' };
const RANKS = [6, 7, 8, 9, 10, 11, 12, 13, 14];
const RANK_NAME = { 6:'6',7:'7',8:'8',9:'9',10:'10',11:'J',12:'Q',13:'K',14:'A' };

// cardKey = (rank-6)*4 + suit  (0..35), как в bitboard движка
const key = (r, s) => (r - 6) * 4 + s;
const rankOf = k => Math.floor(k / 4) + 6;
const suitOf = k => k % 4;
const label  = k => RANK_NAME[rankOf(k)] + SUIT_SYM[SUIT_NAME[suitOf(k)]];
const canBeat = (atk, def, trump) => {
    const aR = rankOf(atk), aS = suitOf(atk), dR = rankOf(def), dS = suitOf(def);
    if (dS === trump && aS !== trump) return true;
    if (dS === aS && dR > aR) return true;
    return false;
};

// ------------------------------------------------------------
//  Состояние трекера
// ------------------------------------------------------------
const T = {
    myHand: [],          // массив cardKey (реальные карты — вводит игрок)
    oppCount: 6,         // считается
    deck: 24,            // считается
    trump: 0,
    table: [],           // [{atk:key, def:key|null}]
    discard: [],         // считается (бито)
    phase: 'attack',     // 'attack' | 'defend'
    attacker: 0,         // 0 = я, 1 = соперник
    firstTrick: true,
    myExpected: 0,       // логический размер моей руки (для проверки добора)
    pendingPursue: false,// фаза «вдогонку» после «беру»
    tookBy: -1,          // кто взял (для pursue)
};
let targetMode = 'hand'; // hand | myPlace | oppPlace | transfer | pursue
let undoStack = [];
let autoAdvice = true;
let adviceTimer = null;
let adviceBusy = false;
let adviceDirty = false;

const pairsLimit = () => T.firstTrick ? 5 : 6;

// Занятые карты (не в руке и не в колоде)
function usedSet() {
    const s = new Set(T.myHand);
    T.table.forEach(p => { s.add(p.atk); if (p.def != null) s.add(p.def); });
    T.discard.forEach(k => s.add(k));
    return s;
}
const ranksOnTable = () => {
    const s = new Set();
    T.table.forEach(p => { s.add(rankOf(p.atk)); if (p.def != null) s.add(rankOf(p.def)); });
    return s;
};
const firstUndef = () => T.table.find(p => p.def == null);

// ------------------------------------------------------------
//  Undo
// ------------------------------------------------------------
function snapshot() { return JSON.stringify(T); }
function pushUndo() { undoStack.push(snapshot()); if (undoStack.length > 200) undoStack.shift(); }
function undo() {
    if (!undoStack.length) return toast('Нечего отменять');
    Object.assign(T, JSON.parse(undoStack.pop()));
    renderAll(); scheduleAdvice();
}

// ------------------------------------------------------------
//  Действия партии
// ------------------------------------------------------------
function newDeal() {
    pushUndo();
    T.myHand = []; T.table = []; T.discard = [];
    T.deck = 24; T.oppCount = 6; T.myExpected = 6;
    T.phase = 'attack'; T.firstTrick = true;
    T.pendingPursue = false; T.tookBy = -1;
    // attacker и trump оставляем как выбрал пользователь (или дефолт)
    renderAll();
    toast('🎲 Раздача. Кликните свои 6 карт.');
    scheduleAdvice();
}

// Клик по карте в палитре — по текущему targetMode
function paletteClick(k) {
    const used = usedSet();
    if (targetMode === 'hand') {
        pushUndo();
        const i = T.myHand.indexOf(k);
        if (i >= 0) T.myHand.splice(i, 1);
        else { if (used.has(k)) return toast('Карта уже в игре'); T.myHand.push(k); }
        T.myExpected = T.myHand.length; // ручная правка синхронизирует ожидание
    }
    else if (targetMode === 'myPlace')  placeCard(0, k);
    else if (targetMode === 'oppPlace') placeCard(1, k);
    else if (targetMode === 'transfer') doTransfer(k);
    else if (targetMode === 'pursue')   doPursue(k);
    renderAll(); scheduleAdvice();
}

// «Я / соперник кладёт карту» — умная расстановка по контексту
function placeCard(who, k) {
    const iAmAtk = (T.attacker === who);
    if (T.pendingPursue) return toast('Сначала завершите «вдогонку» или «беру»');

    if (iAmAtk) {
        // АТАКА / подкидывание
        const undef = firstUndef();
        if (undef) return toast('Соперник ещё не отбился — нельзя класть');
        if (T.table.length > 0) {
            // подкидывание: ранг должен быть на столе + лимиты
            if (!ranksOnTable().has(rankOf(k))) return toast('Подкидывать можно только ранги со стола');
            const defCount = T.attacker === 0 ? T.oppCount : handSize(0);
            if (T.table.length >= pairsLimit()) return toast('Лимит пар (' + pairsLimit() + ')');
            if (T.table.length >= defCount) return toast('Нельзя подкинуть больше, чем карт у защищающегося');
        }
        pushUndo();
        removeFromHand(who, k);
        T.table.push({ atk: k, def: null });
        T.phase = 'defend';
    } else {
        // ЗАЩИТА: бьём первую непобитую
        const undef = firstUndef();
        if (!undef) return toast('Нечего отбивать — все карты побиты');
        if (!canBeat(undef.atk, k, T.trump)) return toast('❌ ' + label(k) + ' не бьёт ' + label(undef.atk));
        pushUndo();
        removeFromHand(who, k);
        undef.def = k;
        // если всё побито — ход атакующему (подкинуть/бито)
        if (!firstUndef()) T.phase = 'attack';
    }
}

function doTransfer(k) {
    const undef = firstUndef();
    if (!undef || T.table.some(p => p.def != null)) return toast('Перевод возможен только пока никто не начал отбиваться');
    if (rankOf(k) !== rankOf(undef.atk)) return toast('Перевод — картой того же ранга (' + RANK_NAME[rankOf(undef.atk)] + ')');
    const who = (T.attacker === 0) ? 1 : 0; // переводит защитник
    const newDefHand = (T.attacker === 0) ? handSize(0) : T.oppCount; // рука нового защитника
    if (newDefHand < T.table.length + 1) return toast('У нового защитника не хватит карт');
    pushUndo();
    removeFromHand(who, k);
    T.table.push({ atk: k, def: null });
    T.attacker = who;          // переведший становится атакующим
    T.phase = 'defend';        // бывший атакующий теперь отбивается
}

function doPursue(k) {
    if (!T.pendingPursue) return;
    const atk = T.attacker;
    if (!ranksOnTable().has(rankOf(k))) return toast('Вдогонку — только ранги со стола');
    pushUndo();
    removeFromHand(atk, k);
    // карта идёт сразу в руку взявшего
    if (T.tookBy === 0) { T.myHand.push(k); } else { T.oppCount++; }
}

function finishPursueAndDraw() {
    pushUndo();
    T.pendingPursue = false;
    drawCards();                 // добор после кона
    T.phase = 'attack';          // ход остаётся у атакующего (взявший пропустил)
    T.firstTrick = false;
    renderAll(); scheduleAdvice();
}

function bito() {
    if (firstUndef()) return toast('Есть неотбитая карта — нельзя «бито»');
    if (!T.table.length) return toast('Стол пуст');
    pushUndo();
    T.table.forEach(p => { T.discard.push(p.atk); if (p.def != null) T.discard.push(p.def); });
    T.table = [];
    drawCards();                 // добор: атакующий первый, затем защитник
    T.attacker = 1 - T.attacker; // ход переходит к защищавшемуся
    T.phase = 'attack';
    T.firstTrick = false;
    renderAll(); scheduleAdvice();
}

function take(who) {
    if (T.pendingPursue) return toast('Уже в режиме «вдогонку»');
    if (!T.table.length) return toast('Стол пуст');
    pushUndo();
    const cards = [];
    T.table.forEach(p => { cards.push(p.atk); if (p.def != null) cards.push(p.def); });
    if (who === 0) { cards.forEach(k => T.myHand.push(k)); }
    else { T.oppCount += cards.length; }
    // ранги со стола запоминаем для подкидывания вдогонку
    T._pursueRanks = ranksOnTable();
    T.table = [];
    T.pendingPursue = true;
    T.tookBy = who;
    renderAll();
    toast((who === 0 ? 'Вы взяли' : 'Соперник взял') + '. Атакующий может подкинуть вдогонку → режим «⤵ Вдогонку», затем «Готово».');
    scheduleAdvice();
}

// Добор до 6 (атакующий первый). Колода уменьшается автоматически.
function drawCards() {
    const order = [T.attacker, 1 - T.attacker];
    for (const p of order) {
        const have = (p === 0) ? T.myExpected : T.oppCount;
        const need = Math.max(0, 6 - have);
        const takeN = Math.min(need, T.deck);
        T.deck -= takeN;
        if (p === 0) {
            T.myExpected += takeN;
            if (takeN > 0) toast('📥 Вы добираете ' + takeN + ' карт — кликните их в руке');
        } else {
            T.oppCount += takeN;
        }
    }
}

function removeFromHand(who, k) {
    if (who === 0) {
        const i = T.myHand.indexOf(k);
        if (i >= 0) T.myHand.splice(i, 1);
    } else {
        T.oppCount = Math.max(0, T.oppCount - 1);
    }
}
const handSize = p => (p === 0 ? T.myHand.length : T.oppCount);

// ------------------------------------------------------------
//  Сборка state для движка + авто-совет
// ------------------------------------------------------------
function buildState() {
    const toCard = k => ({ rank: rankOf(k), suit: SUIT_NAME[suitOf(k)] });
    return {
        myHand: T.myHand.map(toCard),
        oppHand: [],
        oppCardCount: T.oppCount,
        table: T.table.map(p => ({
            attack: toCard(p.atk),
            defense: p.def != null ? toCard(p.def) : { rank: -1, suit: 'clubs' }
        })),
        deckCount: T.deck,
        trumpSuit: SUIT_NAME[T.trump],
        phase: T.phase,
        attacker: T.attacker,
        pairsLimit: pairsLimit(),
        firstTrick: T.firstTrick,
        discard: T.discard.map(toCard),
    };
}

function scheduleAdvice() {
    if (!autoAdvice) return;
    clearTimeout(adviceTimer);
    adviceTimer = setTimeout(runAdvice, 500); // дебаунс
}
async function runAdvice(force) {
    if (!force && adviceBusy) { adviceDirty = true; return; }
    adviceBusy = true; adviceDirty = false;
    setAdviceStatus('🧠 думаю…');
    try {
        const r = await engineAPI.send({ cmd: 'decide', state: buildState() }, 12000);
        if (r && r.status === 'ok' && r.move) showAdvice(r.move);
        else setAdviceStatus('нет хода / ошибка');
    } catch (e) { setAdviceStatus('⚠ ' + e.message); }
    adviceBusy = false;
    if (adviceDirty) { adviceDirty = false; runAdvice(); } // повтор, если состояние менялось
}

let lastAdviceKey = -1;
function showAdvice(m) {
    const k = key(m.card.rank, SUITS.indexOf(m.card.suit));
    lastAdviceKey = k;
    const types = ['АТАКА', 'ЗАЩИТА', 'БЕРУ', 'БИТО', 'ПЕРЕВОД'];
    setAdviceStatus('💡 ' + (types[m.type] || 'ХОД') + ': ' + label(k));
    renderPalette(); // перерисовать подсветку
}
function setAdviceStatus(t) { const e = document.getElementById('advice'); if (e) e.textContent = t; }

// ------------------------------------------------------------
//  Рендер
// ------------------------------------------------------------
function renderAll() { renderPalette(); renderTable(); renderInfo(); renderControls(); }

function renderPalette() {
    const el = document.getElementById('palette');
    el.innerHTML = '';
    const used = usedSet();
    for (const s of SUITS) {
        for (const r of RANKS) {
            const k = key(r, SUITS.indexOf(s));
            const b = document.createElement('button');
            b.className = 'pcard';
            b.textContent = RANK_NAME[r] + SUIT_SYM[s];
            if (T.myHand.includes(k)) b.classList.add('in-hand');
            if (used.has(k) && !T.myHand.includes(k)) b.classList.add('used');
            if (k === lastAdviceKey) b.classList.add('advice');
            b.onclick = () => paletteClick(k);
            el.appendChild(b);
        }
    }
}

function renderTable() {
    const el = document.getElementById('table');
    el.innerHTML = '';
    if (!T.table.length) { el.innerHTML = '<div class="empty">стол пуст</div>'; return; }
    T.table.forEach((p, i) => {
        const d = document.createElement('div');
        d.className = 'pair';
        d.innerHTML = `<span class="a">${label(p.atk)}</span>
                       <span class="arrow">→</span>
                       <span class="d">${p.def != null ? label(p.def) : '…'}</span>
                       <button class="x" title="убрать пару (undo-совместимо)">✕</button>`;
        d.querySelector('.x').onclick = () => { pushUndo(); T.table.splice(i, 1); renderAll(); scheduleAdvice(); };
        el.appendChild(d);
    });
}

function renderInfo() {
    document.getElementById('i-deck').textContent = T.deck;
    document.getElementById('i-opp').textContent = T.oppCount;
    document.getElementById('i-discard').textContent = T.discard.length;
    document.getElementById('i-phase').textContent =
        (T.phase === 'attack' ? '⚔ атака' : '🛡 защита') +
        (T.pendingPursue ? ' / ⤵ вдогонку' : '');
    document.getElementById('i-attacker').textContent = T.attacker === 0 ? 'Я' : 'Соперник';
    document.getElementById('i-limit').textContent = pairsLimit();
    // проверка согласованности руки
    const warn = document.getElementById('hand-warn');
    if (T.myHand.length !== T.myExpected)
        warn.textContent = `⚠ В руке ${T.myHand.length}, должно быть ${T.myExpected} (забыли добавить добранные карты?)`;
    else warn.textContent = '';
}

function renderControls() {
    // подсветка активного targetMode
    document.querySelectorAll('[data-mode]').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === targetMode));
    document.querySelectorAll('[data-trump]').forEach(b =>
        b.classList.toggle('active', parseInt(b.dataset.trump) === T.trump));
    document.getElementById('btn-pursue-done').style.display = T.pendingPursue ? '' : 'none';
}

function toast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg; t.classList.add('show');
    clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove('show'), 2600);
}

// ------------------------------------------------------------
//  Привязка UI
// ------------------------------------------------------------
function init() {
    // палитра целей
    document.querySelectorAll('[data-mode]').forEach(b =>
        b.onclick = () => { targetMode = b.dataset.mode; renderControls(); });
    // козырь
    document.querySelectorAll('[data-trump]').forEach(b =>
        b.onclick = () => { pushUndo(); T.trump = parseInt(b.dataset.trump); renderAll(); scheduleAdvice(); });
    // роль атакующего
    document.querySelectorAll('[data-att]').forEach(b =>
        b.onclick = () => { pushUndo(); T.attacker = parseInt(b.dataset.att); renderAll(); scheduleAdvice(); });
    // действия
    document.getElementById('btn-deal').onclick = newDeal;
    document.getElementById('btn-bito').onclick = bito;
    document.getElementById('btn-take-me').onclick = () => take(0);
    document.getElementById('btn-take-opp').onclick = () => take(1);
    document.getElementById('btn-pursue-done').onclick = finishPursueAndDraw;
    document.getElementById('btn-undo').onclick = undo;
    document.getElementById('btn-advice').onclick = () => runAdvice(true);
    document.getElementById('btn-legal').onclick = showLegal;
    document.getElementById('chk-auto').onchange = e => { autoAdvice = e.target.checked; if (autoAdvice) scheduleAdvice(); };

    newDeal(); // колода сразу 24, как просили
}

async function showLegal() {
    try {
        const r = await engineAPI.send({ cmd: 'legalMoves', state: buildState() }, 8000);
        if (r && r.moves) toast('Легальных ходов: ' + r.moves.length +
            ' → ' + r.moves.map(m => RANK_NAME[m.card.rank] + SUIT_SYM[m.card.suit]).join(' '));
    } catch (e) { toast('ошибка: ' + e.message); }
}

document.addEventListener('DOMContentLoaded', init);
