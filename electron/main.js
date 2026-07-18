'use strict';

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { EngineBridge, findEnginePath } = require('./engine-bridge');

let mainWindow = null;
const engine = new EngineBridge();

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 820,
        minWidth: 960,
        minHeight: 640,
        title: 'Помощник: переводной дурак онлайн',
        backgroundColor: '#0f1320',
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: false,
        },
    });

    mainWindow.loadFile(path.join(__dirname, '..', 'src', 'index.html'));

    // Передаём рендереру события статуса движка (жив/упал/exe не найден).
    const forward = (info) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('engine:status', info);
        }
    };
    engine.on('status', forward);
    engine.on('stderr', (s) => {
        console.error('[engine stderr]', s);
    });

    mainWindow.on('closed', () => { mainWindow = null; });
}

// ---- IPC: рендерер → главный процесс → движок ----

ipcMain.handle('bot:decide', async (_evt, state, opts) => {
    try {
        await engine.setState(state);
        const res = await engine.decide(opts || {});
        return { ok: true, move: res };
    } catch (e) {
        return { ok: false, error: e.message };
    }
});

ipcMain.handle('bot:legalMoves', async (_evt, state) => {
    try {
        await engine.setState(state);
        const res = await engine.legalMoves();
        return { ok: true, moves: res };
    } catch (e) {
        return { ok: false, error: e.message };
    }
});

ipcMain.handle('bot:validate', async (_evt, state, action) => {
    try {
        await engine.setState(state);
        const res = await engine.validate(action);
        return { ok: true, result: res };
    } catch (e) {
        return { ok: false, error: e.message };
    }
});

ipcMain.handle('engine:status', async () => {
    return { available: !!findEnginePath(), path: findEnginePath() };
});

// ---- Жизненный цикл приложения ----

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    engine.dispose();
    if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on('before-quit', () => engine.dispose());
