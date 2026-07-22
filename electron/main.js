const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const bridge = require('./engine-bridge');

let win;

app.whenReady().then(() => {
    win = new BrowserWindow({
        width: 900,
        height: 750,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false
        }
    });

    win.loadFile(path.join(__dirname, '..', 'src', 'index.html'));

    // Запуск движка
    bridge.start();
    bridge.on('ready', () => {
        win.webContents.send('engine-status', 'ready');
    });
    bridge.on('exit', () => {
        win.webContents.send('engine-status', 'restarting');
    });
});

// IPC: renderer → engine
ipcMain.handle('engine-send', async (event, obj) => {
    try {
        return await bridge.send(obj, 10000); // 10 сек таймаут (MCTS ~4.5с)
    } catch (e) {
        return { error: e.message };
    }
});

app.on('window-all-closed', () => {
    bridge.stop();
    app.quit();
});
