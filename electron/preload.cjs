const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
    // Dialog operations
    openDirectory: () => ipcRenderer.invoke('dialog:openDirectory'),
    openFiles: (options) => ipcRenderer.invoke('dialog:openFiles', options),
    saveFile: (options) => ipcRenderer.invoke('dialog:saveFile', options),

    // File system operations
    exists: (path) => ipcRenderer.invoke('fs:exists', path),
    readDir: (path) => ipcRenderer.invoke('fs:readDir', path),
    mkdir: (path) => ipcRenderer.invoke('fs:mkdir', path),
    readFile: (path) => ipcRenderer.invoke('fs:readFile', path),
    writeFile: (path, content) => ipcRenderer.invoke('fs:writeFile', path, content),
    writeFileBuffer: (path, buffer) => ipcRenderer.invoke('fs:writeFileBuffer', path, buffer),

    // App utilities
    getPath: (name) => ipcRenderer.invoke('app:getPath', name),
    getBackendPort: () => ipcRenderer.invoke('backend:getPort'),

    // Platform info
    platform: process.platform,
});
