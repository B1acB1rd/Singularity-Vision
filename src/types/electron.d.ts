// Electron API types exposed via preload
export interface ElectronAPI {
    // Dialog operations
    openDirectory: () => Promise<string | null>;
    openFiles: (options?: {
        filters?: { name: string; extensions: string[] }[];
    }) => Promise<string[]>;
    saveFile: (options?: {
        filters?: { name: string; extensions: string[] }[];
        defaultPath?: string;
    }) => Promise<string | null>;

    // File system operations
    exists: (path: string) => Promise<boolean>;
    readDir: (path: string) => Promise<{ name: string; isDirectory: boolean }[]>;
    mkdir: (path: string) => Promise<boolean>;
    readFile: (path: string) => Promise<string>;
    writeFile: (path: string, content: string) => Promise<boolean>;
    writeFileBuffer: (path: string, data: Uint8Array) => Promise<boolean>;

    // App utilities
    getPath: (name: 'home' | 'appData' | 'userData' | 'documents' | 'downloads' | 'desktop') => Promise<string>;
    getBackendPort: () => Promise<number>;

    // Platform info
    platform: 'win32' | 'darwin' | 'linux';
}

declare global {
    interface Window {
        electronAPI: ElectronAPI;
    }
}

export { };
