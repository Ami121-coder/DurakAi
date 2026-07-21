'use strict';

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { EngineBridge, findEnginePath } = require('./engine-bridge');

let mainWindow = null;
const engine = new EngineBridge();

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200, height: 820, minWidth: 960, minHeight: 640,
    title: 'Помощник: переводной дурак онлайн',
    backgroundColor: '#0f1320',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,  // БАГ L: было false
    },
  });
  mainWindow.loadFile(path.join(__dirname, '..', 'src', 'index.html'));
  const forward = (info) => {
    if (mainWindow && !mainWindow.isDestroyed())
      mainWindow.webContents.send('engine:status', info);
  };
  engine.on('status', forward);
  engine.on('stderr', (s) => { console.error('[engine stderr]', s); });
  mainWindow.on('closed', () => { mainWindow = null; });
}

// БАГ K: atomicDecide вместо setState + decide
ipcMain.handle('bot:decide', async (_evt, state, opts) => {
  try {
    const res = await engine.atomicDecide(state, opts || {});
    return { ok: true, move: res };
  } catch (e) { return { ok: false, error: e.message }; }
});

ipcMain.handle('bot:legalMoves', async (_evt, state) => {
  try {
    const res = await engine.atomicLegalMoves(state);
    return { ok: true, moves: res };
  } catch (e) { return { ok: false, error: e.message }; }
});

ipcMain.handle('bot:validate', async (_evt, state, action) => {
  try {
    const res = await engine.atomicValidate(state, action);
    return { ok: true, result: res };
  } catch (e) { return { ok: false, error: e.message }; }
});

// БАГ M: findEnginePath() один раз
ipcMain.handle('engine:status', async () => {
  const p = findEnginePath();
  return { available: !!p, path: p };
});

app.whenReady().then(createWindow);
app.on('window-all-closed', () => {
  engine.dispose();
  if (process.platform !== 'darwin') app.quit();
});
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
app.on('before-quit', () => engine.dispose());
