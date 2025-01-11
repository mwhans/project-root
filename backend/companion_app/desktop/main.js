const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

let mainWindow;
let backendProcess;

function startBackendServer() {
  console.log('Starting backend server...');
  
  // Get the path to the companion_app directory
  const companionAppPath = path.join(__dirname, '..');
  
  // Start the server using python -m
  backendProcess = spawn('python3', [
    '-m',
    'companion_app.server',
    '--desktop'
  ], {
    cwd: path.join(companionAppPath, '..'),
    env: {
      ...process.env,
      PYTHONPATH: path.join(companionAppPath, '..')
    }
  });

  backendProcess.stdout.on('data', (data) => {
    console.log(`Backend stdout: ${data}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`Backend Error: ${data}`);
  });

  backendProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  // Wait for backend to start
  setTimeout(() => {
    mainWindow.loadURL('http://localhost:8000');
  }, 2000);

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.on('ready', () => {
  startBackendServer();
  createWindow();
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') {
    app.quit();
  }
  if (backendProcess) {
    backendProcess.kill();
  }
});

app.on('activate', function () {
  if (mainWindow === null) {
    createWindow();
  }
}); 