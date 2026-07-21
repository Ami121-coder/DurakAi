'use strict';

const { contextBridge, ipcRenderer } = require('electron');

// Безопасный API рендерера: никаких require/Node, только три метода + подписка
// на статус движка. contextIsolation включён, nodeIntegration выключен.
contextBridge.exposeInMainWorld('bot', {
    // state — объект состояния игры в формате движка (см. protocol.h).
    // opts  — опциональный объект { strength: 'fast'|'normal'|'deep' }.
    // FIX #11: раньше opts не пробрасывался, и выбор силы в UI не влиял
    // на поведение бота (всегда использовался Normal).
    decide: (state, opts)       => ipcRenderer.invoke('bot:decide', state, opts),
    legalMoves: (state)         => ipcRenderer.invoke('bot:legalMoves', state),
    validate: (state, action)   => ipcRenderer.invoke('bot:validate', state, action),
    status: ()                  => ipcRenderer.invoke('engine:status'),
    onStatus: (cb) => {
        const handler = (_e, info) => cb(info);
        ipcRenderer.on('engine:status', handler);
        return () => ipcRenderer.removeListener('engine:status', handler);
    },
});
