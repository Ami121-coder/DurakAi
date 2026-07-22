const { spawn } = require('child_process');
const path = require('path');
const EventEmitter = require('events');

class EngineBridge extends EventEmitter {
    constructor() {
        super();
        this.proc = null;
        this.buffer = '';
        this.pending = new Map(); // id → {resolve, reject, timer}
        this.nextId = 1;
        this.alive = false;
    }

    start() {
        const exePath = path.join(__dirname, '..', 'durakk_engine.exe');
        this.proc = spawn(exePath, [], {
            stdio: ['pipe', 'pipe', 'pipe'],
            windowsHide: true
        });

        this.proc.stdout.on('data', (data) => {
            this.buffer += data.toString();
            let idx;
            while ((idx = this.buffer.indexOf('\n')) >= 0) {
                const line = this.buffer.slice(0, idx).trim();
                this.buffer = this.buffer.slice(idx + 1);
                if (line) this._onLine(line);
            }
        });

        this.proc.stderr.on('data', (data) => {
            console.error('[engine stderr]', data.toString());
        });

        this.proc.on('exit', (code) => {
            console.log(`[engine] exited with code ${code}`);
            this.alive = false;
            this.emit('exit', code);
            // Авто-рестарт через 2 сек
            setTimeout(() => this.start(), 2000);
        });

        this.alive = true;
        this.emit('ready');

        // Ping для проверки
        this.send({ cmd: 'ping' }).then(r => {
            console.log('[engine] ping:', r);
        }).catch(() => {});
    }

    _onLine(line) {
        try {
            const msg = JSON.parse(line);
            // Простой протокол: один запрос — один ответ (без id)
            // Берём первый pending
            if (this.pending.size > 0) {
                const [id, p] = this.pending.entries().next().value;
                this.pending.delete(id);
                clearTimeout(p.timer);
                p.resolve(msg);
            }
        } catch (e) {
            console.error('[engine] parse error:', line);
        }
    }

    send(obj, timeoutMs = 10000) {
        return new Promise((resolve, reject) => {
            if (!this.alive || !this.proc) {
                reject(new Error('Движок не запущен'));
                return;
            }
            const id = this.nextId++;
            const timer = setTimeout(() => {
                this.pending.delete(id);
                reject(new Error('Таймаут ответа движка'));
            }, timeoutMs);

            this.pending.set(id, { resolve, reject, timer });
            this.proc.stdin.write(JSON.stringify(obj) + '\n');
        });
    }

    stop() {
        if (this.proc) {
            this.proc.kill();
            this.proc = null;
        }
        this.alive = false;
    }
}

module.exports = new EngineBridge();
