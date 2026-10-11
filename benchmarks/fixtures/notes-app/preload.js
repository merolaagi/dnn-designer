const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('notes', {
  search: (query) => ipcRenderer.invoke('notes:search', query),
  copy: (text) => ipcRenderer.invoke('clipboard:write', text),
});
