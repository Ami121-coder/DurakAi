'use strict';

const RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
const SUITS = ['clubs', 'diamonds', 'hearts', 'spades'];
const SUIT_SYM = { clubs: '♣', diamonds: '♦', hearts: '♥', spades: '♠' };
const SUIT_COLOR = { clubs: 'black', diamonds: 'red', hearts: 'red', spades: 'black' };

const state = {
  trump: 'spades',
  hand: new Set(),
  table: [],
  adviceCards: new Set(),
  discard: new Set(),
  oppTaken: new Set(),
  strength: 'normal',
};

const $ = (id) => document.getElementById(id);

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

function renderHand() {
  const grid = $('handGrid');
  grid.innerHTML = '';
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      const k = cardKey(rank, suit);
      const card = document.createElement('div');
      card.className = 'card' + (SUIT_COLOR[suit] === 'red' ? ' red' : '');
      if (state.hand.has(k)) card.classList.add('is-selected');
      if (state.adviceCards.has(k)) card.classList.add('is-advice');
      card.innerHTML = `<span class="rank">${rank}</span><span class="suit">${SUIT_SYM[suit]}</span>`;
      card.addEventListener('click', () => {
        if (state.hand.has(k)) state.hand.delete(k);
        else state.hand.add(k);
        renderHand();
      });
      grid.appendChild(card);
    }
  }
}

function renderTrump() {
  const picker = $('trumpPicker');
  picker.innerHTML = '';
  for (const s of SUITS) {
    const b = document.createElement('button');
    b.dataset.suit = s;
    b.textContent = SUIT_SYM[s];
    if (s === state.trump) b.classList.add('is-active');
    b.addEventListener('click', () => { state.trump = s; renderTrump(); });
    picker.appendChild(b);
  }
}

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
    vs.className = 'vs'; vs.textContent = '←';
    box.appendChild(vs);
    box.appendChild(miniCard(def, 'защита', () => pickForTable(idx, 'defense')));
    const del = document.createElement('button');
    del.className = 'del'; del.textContent = '✕'; del.title = 'убрать пару';
    del.addEventListener('click', () => { state.table.splice(idx, 1); renderTable(); });
    box.appendChild(del);
    grid.appendChild(box);
  });
}

function miniCard(card, placeholder, onclick) {
  const m = document.createElement('div');
  m.className = 'mini' + (card ? '' : ' empty');
  if (card) {
    m.innerHTML = `<span class="rank">${card.rank}</span><span class="suit">${SUIT_SYM[card.suit]}</span>`;
  } else { m.textContent = placeholder; }
  m.addEventListener('click', onclick);
  return m;
}

function pickForTable(idx, slot) {
  const input = window.prompt(
    `Введите карту для «${slot === 'attack' ? 'атаки' : 'защиты'}» в формате РАНГ МАСТЬ\n` +
    `Ранг: ${RANKS.join(', ')}\nМасть: clubs/diamonds/hearts/spades`, '');
  if (!input) return;
  const parts = input.trim().split(/\s+/);
  if (parts.length !== 2) return alert('Формат: РАНГ МАСТЬ');
  const [rank, suit] = parts;
  if (!RANKS.includes(rank) || !SUITS.includes(suit)) return alert('Неверный ранг или масть');
  state.table[idx][slot] = cardKey(rank, suit);
  renderTable();
}

function buildState() {
  const table = state.table.map((p) => {
    const o = {};
    if (p.attack) { const c = splitKey(p.attack); o.attack = { r: c.rank, s: c.suit }; }
    if (p.defense) { const c = splitKey(p.defense); o.defense = { r: c.rank, s: c.suit }; }
    return o;
  }).filter((p) => p.attack);
  return {
    trump: state.trump, deckSize: 36,
    transferEnabled: $('transferEnabled').checked,
    flashEnabled: $('flashEnabled').checked,
    firstTrickLimit: 5,
    deckCount: parseInt($('deckCount').value, 10) || 0,
    oppHandCount: parseInt($('oppHandCount').value, 10) || 0,
    myHand: [...state.hand].map((k) => { const c = splitKey(k); return { r: c.rank, s: c.suit }; }),
    discard: [...state.discard].map((k) => { const c = splitKey(k); return { r: c.rank, s: c.suit }; }),
    oppTaken: [...state.oppTaken].map((k) => { const c = splitKey(k); return { r: c.rank, s: c.suit }; }),
    table,
    attacker: $('attacker').value, turn: $('turn').value, phase: $('phase').value,
    firstTrick: $('firstTrick').checked,
  };
}

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
  if (st.solved) parts.push(`<b>решён</b>`);
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
      card.innerHTML = `<span class="rank">${rank}</span><span class="suit">${SUIT_SYM[suit]}</span>`;
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
  return `${move.rank}${SUIT_SYM[move.suit] || ''}`;
}

const ACTION_LABEL = {
  attack: 'Атаковать', defend: 'Защищаться', transfer: 'Перевести',
  toss: 'Подкинуть', take: 'Взять карты', done: 'Бито', pass: 'Пропустить',
};

// ====================================================================
// БАГ C (ИСПРАВЛЕН): stats в res.move.stats, а не res.stats
// БАГ I (ИСПРАВЛЕН): try/catch вокруг IPC-вызовов
// ====================================================================
async function onDecide() {
  const st = buildState();
  $('lastRequest').textContent = JSON.stringify(st, null, 2);
  state.adviceCards.clear();
  showAdvice('Думаю…', 'advice--empty');

  let res;
  try {
    res = await window.bot.decide(st, { strength: state.strength });
  } catch (e) {
    showAdvice(`Ошибка связи с движком: ${e.message}`, 'advice--err');
    return;
  }

  $('lastResponse').textContent = JSON.stringify(res, null, 2);
  if (!res.ok) { showAdvice(`Ошибка: ${res.error}`, 'advice--err'); return; }

  const move = res.move;
  if (move.rank && move.suit) state.adviceCards.add(cardKey(move.rank, move.suit));
  renderHand();

  const card = renderMoveCard(move);
  const reason = move.reason ? `<div class="reason">${move.reason}</div>` : '';
  // БАГ C: stats внутри move (res.move.stats), а не res.stats
  const stats = (move.stats) ? renderStats(move.stats) : '';
  showAdvice(
    `<div class="action-label">${ACTION_LABEL[move.action] || move.action}</div>` +
    `<div class="move-card">${card}</div>${reason}${stats}`,
    'advice--ok');
}

// БАГ I: try/catch
async function onLegal() {
  const st = buildState();
  let res;
  try {
    res = await window.bot.legalMoves(st);
  } catch (e) {
    $('legalList').innerHTML = `<span class="err">Ошибка: ${e.message}</span>`;
    return;
  }
  const box = $('legalList');
  if (!res.ok) { box.innerHTML = `<span class="err">${res.error}</span>`; return; }
  const m = res.moves;
  const fmt = (arr) => arr.length ? arr.map((c) => `${c.r}${SUIT_SYM[c.s]}`).join(' ') : '—';
  const fmtDef = (arr) => arr.length
    ? arr.map((d) => `${d.card.r}${SUIT_SYM[d.card.s]}→${d.target.r}${SUIT_SYM[d.target.s]}`).join(' ')
    : '—';
  box.innerHTML = `
    <p><b>Атака/подкидывание:</b> ${fmt(m.attacks)}</p>
    <p><b>Подкидывание:</b> ${fmt(m.tosses)}</p>
    <p><b>Перевод:</b> ${fmt(m.transfers)}</p>
    <p><b>Защита:</b> ${fmtDef(m.defends)}</p>
    <p><b>Можно «бито»:</b> ${m.canDone ? 'да' : 'нет'}</p>`;
}

function onReset() {
  state.hand.clear(); state.table = []; state.adviceCards.clear();
  state.discard.clear(); state.oppTaken.clear();
  renderHand(); renderTable();
  renderMemGrid('discardGrid', state.discard, 'is-discard');
  renderMemGrid('oppTakenGrid', state.oppTaken, 'is-taken');
  showAdvice('Заполните состояние и нажмите «Получить совет».', 'advice--empty');
  $('legalList').textContent = '—';
}

function init() {
  renderHand(); renderTrump(); renderTable();
  renderMemGrid('discardGrid', state.discard, 'is-discard');
  renderMemGrid('oppTakenGrid', state.oppTaken, 'is-taken');
  document.querySelectorAll('.strength-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.strength-btn').forEach((b) => b.classList.remove('strength-btn--active'));
      btn.classList.add('strength-btn--active');
      state.strength = btn.dataset.strength;
    });
  });
  $('addTablePair').addEventListener('click', () => {
    state.table.push({ attack: null, defense: null }); renderTable();
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
