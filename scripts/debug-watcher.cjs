#!/usr/bin/env node
/**
 * Singularity Vision - Console Log Watcher
 * 
 * This script runs the app and captures all console output to a log file
 * that can be analyzed for debugging.
 * 
 * Usage: npm run dev:debug
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Create logs directory
const logsDir = path.join(__dirname, '..', 'debug-logs');
if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
}

// Create timestamped log file
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const logFile = path.join(logsDir, `session-${timestamp}.log`);
const logStream = fs.createWriteStream(logFile, { flags: 'a' });

console.log(`📝 Logging to: ${logFile}\n`);

// Write header
logStream.write(`\n${'='.repeat(80)}\n`);
logStream.write(`SINGULARITY VISION DEBUG SESSION\n`);
logStream.write(`Started: ${new Date().toISOString()}\n`);
logStream.write(`${'='.repeat(80)}\n\n`);

// Function to log with timestamp
function log(source, message, isError = false) {
    const time = new Date().toLocaleTimeString();
    const prefix = isError ? '❌ ERROR' : '📋';
    const line = `[${time}] [${source}] ${message}`;

    // Write to file
    logStream.write(line + '\n');

    // Also output to console
    if (isError) {
        console.error(line);
    } else {
        console.log(line);
    }
}

// Start the electron dev process
const child = spawn('npm', ['run', 'electron:dev'], {
    cwd: path.join(__dirname, '..'),
    shell: true,
    stdio: ['pipe', 'pipe', 'pipe']
});

// Capture stdout
child.stdout.on('data', (data) => {
    const lines = data.toString().split('\n').filter(l => l.trim());
    lines.forEach(line => {
        // Categorize the log
        if (line.includes('[Python]')) {
            log('BACKEND', line);
        } else if (line.includes('[Python Error]')) {
            log('BACKEND', line, true);
        } else if (line.includes('vite')) {
            log('VITE', line);
        } else if (line.includes('ERROR') || line.includes('error')) {
            log('APP', line, true);
        } else {
            log('APP', line);
        }
    });
});

// Capture stderr
child.stderr.on('data', (data) => {
    const lines = data.toString().split('\n').filter(l => l.trim());
    lines.forEach(line => {
        log('STDERR', line, true);
    });
});

// Handle process exit
child.on('close', (code) => {
    log('SYSTEM', `Process exited with code ${code}`);
    logStream.write(`\n${'='.repeat(80)}\n`);
    logStream.write(`Session ended: ${new Date().toISOString()}\n`);
    logStream.write(`Exit code: ${code}\n`);
    logStream.end();

    console.log(`\n📁 Log saved to: ${logFile}`);
});

// Handle Ctrl+C gracefully
process.on('SIGINT', () => {
    log('SYSTEM', 'Received SIGINT, shutting down...');
    child.kill('SIGINT');
});

console.log('🚀 Starting Singularity Vision in debug mode...\n');
