const { app, BrowserWindow, ipcMain, clipboard } = require('electron');
const path = require('path');
const store = require('./store');

function createWindow() {
  const win = new BrowserWindow({
    width: 900,
    height: 640,
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true },
  });
  win.loadFile('index.html');
}

ipcMain.handle('notes:search', (_event, query) => store.search(query));
ipcMain.handle('clipboard:write', (_event, text) => {
  clipboard.writeText(String(text));
  return true;
});

app.whenReady().then(createWindow);
