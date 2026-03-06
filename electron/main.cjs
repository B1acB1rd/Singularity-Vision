const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
// if (require('electron-squirrel-startup')) {
//   app.quit();
// }

let mainWindow = null;
let pythonProcess = null;

const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
const PYTHON_PORT = 8765;

// Python backend management
function startPythonBackend() {
  const backendPath = isDev
    ? path.join(__dirname, '..', 'backend')
    : path.join(process.resourcesPath, 'backend');

  const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';

  console.log('Starting Python backend from:', backendPath);

  pythonProcess = spawn(pythonExecutable, ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', String(PYTHON_PORT)], {
    cwd: backendPath,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python] ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error] ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python backend exited with code ${code}`);
  });

  return new Promise((resolve) => {
    // Wait for backend to be ready
    const checkBackend = async () => {
      try {
        const response = await fetch(`http://127.0.0.1:${PYTHON_PORT}/health`);
        if (response.ok) {
          console.log('Python backend is ready');
          resolve(true);
          return;
        }
      } catch (e) {
        // Not ready yet
      }
      setTimeout(checkBackend, 500);
    };
    setTimeout(checkBackend, 1000);
  });
}

function stopPythonBackend() {
  if (pythonProcess) {
    console.log('Stopping Python backend...');
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', pythonProcess.pid, '/f', '/t']);
    } else {
      pythonProcess.kill('SIGTERM');
    }
    pythonProcess = null;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    title: 'Singularity Vision',
    icon: path.join(__dirname, 'icons', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
    backgroundColor: '#0a0a0f',
    show: false,
  });

  // Load the app
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC Handlers
ipcMain.handle('dialog:openDirectory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });
  return result.filePaths[0] || null;
});

ipcMain.handle('dialog:openFiles', async (_, options) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: options?.filters || [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'] },
      { name: 'Videos', extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm'] },
    ],
  });
  return result.filePaths;
});

ipcMain.handle('dialog:saveFile', async (_, options) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    filters: options?.filters || [],
    defaultPath: options?.defaultPath,
  });
  return result.filePath || null;
});

ipcMain.handle('fs:exists', async (_, filePath) => {
  return fs.existsSync(filePath);
});

ipcMain.handle('fs:readDir', async (_, dirPath) => {
  return fs.readdirSync(dirPath, { withFileTypes: true }).map((dirent) => ({
    name: dirent.name,
    isDirectory: dirent.isDirectory(),
  }));
});

ipcMain.handle('fs:mkdir', async (_, dirPath) => {
  fs.mkdirSync(dirPath, { recursive: true });
  return true;
});

ipcMain.handle('fs:readFile', async (_, filePath) => {
  return fs.readFileSync(filePath, 'utf-8');
});

ipcMain.handle('fs:writeFile', async (_, filePath, content) => {
  fs.writeFileSync(filePath, content, 'utf-8');
  return true;
});

ipcMain.handle('fs:writeFileBuffer', async (_, filePath, buffer) => {
  fs.writeFileSync(filePath, Buffer.from(buffer));
  return true;
});

ipcMain.handle('app:getPath', async (_, name) => {
  return app.getPath(name);
});

ipcMain.handle('backend:getPort', () => {
  return PYTHON_PORT;
});

// App lifecycle
app.whenReady().then(async () => {
  // Register 'media' protocol to bypass "Not allowed to load local resource"
  // Usage: media://C:/path/to/file.png
  const { protocol } = require('electron');
  protocol.registerFileProtocol('media', (request, callback) => {
    let url = request.url;

    // Remove protocol prefix
    url = url.replace(/^media:\/\//, '');

    // Decode URI components (spaces become %20 etc)
    url = decodeURIComponent(url);

    // On Windows, handle potential leading slash before drive letter
    if (process.platform === 'win32') {
      // /C:/path -> C:/path
      if (url.startsWith('/') && /^[a-zA-Z]:/.test(url.slice(1))) {
        url = url.slice(1);
      }
    }

    // Normalize path separators for the OS
    const filePath = path.normalize(url);

    console.log('[Media Protocol] Resolved path:', filePath);

    // Check if file exists before returning
    if (!fs.existsSync(filePath)) {
      console.error('[Media Protocol] File not found:', filePath);
      return callback({ error: -6 }); // net::ERR_FILE_NOT_FOUND
    }

    return callback({ path: filePath });
  });

  await startPythonBackend();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopPythonBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopPythonBackend();
});
