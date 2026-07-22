const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('engineAPI', {
    send: (obj) => ipcRenderer.invoke('engine-send', obj),
    onStatus: (cb) => ipcRenderer.on('engine-status', (_, s) => cb(s))
});
