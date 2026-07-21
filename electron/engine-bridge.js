'use strict';

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const readline = require('readline');
const { EventEmitter } = require('events');

const ROOT = path.resolve(__dirname, '..');

function findEnginePath() {
  const exeName = process.platform === 'win32' ? 'durakk_engine.exe' : 'durakk_engine';
  const candidates = [
    path.join(ROOT, exeName),
    path.join(ROOT, 'engine', 'dist', exeName),
    path.join(ROOT, 'engine', 'build', 'Release', exeName),
    path.join(ROOT, 'engine', 'build', 'Debug', exeName),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

class EngineBridge extends EventEmitter {
  constructor() {
    super();
    this.proc = null;
    this.rl = null;
    this.pending = new Map();
    this.nextId = 1;
    this.enginePath = null;
    this.starting = null;
    this.dead = false;
    // БАГ K: очередь для атомарных setState+decide
    this._queue = Promise.resolve();
  }

  isAvailable() { return this.enginePath !== null; }

  async ensureStarted() {
    if (this.dead) throw new Error('Движок недоступен (exe не найден или упал).');
    if (this.proc && this.proc.exitCode === null) return;
    if (this.starting) return this.starting;
    this.starting = this._start().finally(() => { this.starting = null; });
    return this.starting;
  }

  async _start() {
    const exe = findEnginePath();
    if (!exe) {
      this.dead = true;
      this.enginePath = null;
      this.emit('status', { ok: false, reason: 'exe-not-found' });
      throw new Error(
        'durakk_engine не собран. Запустите: npm run build:engine ' +
        '(сначала установите VS Build Tools — см. README).');
    }
    this.enginePath = exe;
    this.dead = false;
    this.proc = spawn(exe, [], { stdio: ['pipe', 'pipe', 'pipe'] });
    this.rl = readline.createInterface({ input: this.proc.stdout });
    this.rl.on('line', (line) => this._onLine(line));
    this.proc.stderr.on('data', (d) => { this.emit('stderr', d.toString()); });
    this.proc.on('exit', (code, signal) => {
      for (const [id, p] of this.pending) {
        p.reject(new Error(`движок завершился (code=${code}, signal=${signal})`));
        clearTimeout(p.timer);
      }
      this.pending.clear();
      this.emit('status', { ok: false, reason: 'exit', code, signal });
    });
    try {
      await this.request('ping');
      this.emit('status', { ok: true, path: exe });
    } catch (e) { throw e; }
  }

  _onLine(line) {
    let msg;
    try { msg = JSON.parse(line); }
    catch (e) { this.emit('parse-error', { line, error: e.message }); return; }
    const id = msg && msg._id;
    if (id && this.pending.has(id)) {
      const p = this.pending.get(id);
      this.pending.delete(id);
      clearTimeout(p.timer);
      p.resolve(msg);
    } else if (msg && msg.event) {
      this.emit('event', msg);
    }
  }

  // БАГ J (ИСПРАВЛЕН): try/catch вокруг stdin.write
  request(cmd, payload = {}, timeoutMs = 5000) {
    return this.ensureStarted().then(() => new Promise((resolve, reject) => {
      const id = this.nextId++;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`таймаут ожидания ответа движка (${cmd})`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      const frame = JSON.stringify({ cmd, _id: id, ...payload });
      try {
        this.proc.stdin.write(frame + '\n');
      } catch (e) {
        this.pending.delete(id);
        clearTimeout(timer);
        reject(new Error(`не удалось отправить команду движку (${cmd}): ${e.message}`));
      }
    }));
  }

  async setState(state) { return this.request('setState', state); }

  async decide(opts = {}) {
    const payload = {};
    if (opts && opts.strength) payload.strength = opts.strength;
    return this.request('decide', payload);
  }

  async legalMoves() { return this.request('legalMoves'); }
  async validate(action) { return this.request('validate', action); }

  // БАГ K (ИСПРАВЛЕН): атомарные операции через очередь
  async atomicDecide(state, opts = {}) {
    const result = this._queue.then(async () => {
      await this.setState(state);
      return this.decide(opts);
    });
    this._queue = result.catch(() => {});
    return result;
  }

  async atomicLegalMoves(state) {
    const result = this._queue.then(async () => {
      await this.setState(state);
      return this.legalMoves();
    });
    this._queue = result.catch(() => {});
    return result;
  }

  async atomicValidate(state, action) {
    const result = this._queue.then(async () => {
      await this.setState(state);
      return this.validate(action);
    });
    this._queue = result.catch(() => {});
    return result;
  }

  dispose() {
    this.dead = true;
    if (this.proc) {
      try { this.proc.stdin.end(); } catch (_) {}
      try { this.proc.kill(); } catch (_) {}
    }
  }
}

module.exports = { EngineBridge, findEnginePath };
