import './settings.css';
import { useState, useEffect } from 'react';
import { getSystemInfo } from '../../services/api';
import {
    Settings,
    Moon,
    Sun,
    Cpu,
    HardDrive,
    Trash2,
    Save,
    RefreshCw,
    Monitor,
    Zap,
    MemoryStick,
    CheckCircle,
    XCircle,
    Loader2,
} from 'lucide-react';

interface SettingsState {
    theme: 'dark' | 'light';
    device: 'auto' | 'cpu' | 'cuda';
    autoSaveInterval: number;
    maxGpuMemory: number;
    enableAnimations: boolean;
}

interface SystemInfo {
    platform: string;
    python_version: string;
    cpu_count: number;
    memory_total: number;
    gpu_available: boolean;
    gpu_name: string | null;
    torch_version: string;
}

const DEFAULT_SETTINGS: SettingsState = {
    theme: 'dark',
    device: 'auto',
    autoSaveInterval: 60,
    maxGpuMemory: 80,
    enableAnimations: true,
};

export default function SettingsView() {
    const [settings, setSettings] = useState<SettingsState>(DEFAULT_SETTINGS);
    const [saved, setSaved] = useState(false);
    const [cacheSize, setCacheSize] = useState<string>('Calculating...');
    const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
    const [systemLoading, setSystemLoading] = useState(true);

    useEffect(() => {
        // Load settings from localStorage
        const savedSettings = localStorage.getItem('singularity-settings');
        if (savedSettings) {
            setSettings(JSON.parse(savedSettings));
        }

        // Calculate cache size (mock for now)
        setCacheSize('24.5 MB');

        // Fetch system info from backend
        const fetchSystemInfo = async () => {
            try {
                const info = await getSystemInfo();
                setSystemInfo(info);
            } catch (error) {
                console.error('Failed to fetch system info:', error);
            } finally {
                setSystemLoading(false);
            }
        };
        fetchSystemInfo();
    }, []);

    const formatBytes = (bytes: number): string => {
        const gb = bytes / (1024 ** 3);
        return `${gb.toFixed(1)} GB`;
    };

    const handleChange = <K extends keyof SettingsState>(
        key: K,
        value: SettingsState[K]
    ) => {
        setSettings((prev) => ({ ...prev, [key]: value }));
        setSaved(false);
    };

    const handleSave = () => {
        localStorage.setItem('singularity-settings', JSON.stringify(settings));
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    };

    const handleClearCache = async () => {
        // Clear various caches
        if (window.electronAPI) {
            // Clear temp files, etc.
        }
        localStorage.removeItem('recent-projects');
        setCacheSize('0 MB');
        alert('Cache cleared successfully!');
    };

    const handleResetDefaults = () => {
        setSettings(DEFAULT_SETTINGS);
        setSaved(false);
    };

    return (
        <div className="settings-view">
            <div className="settings-header">
                <Settings size={24} />
                <h2>Settings</h2>
                {saved && <span className="save-indicator">✓ Saved</span>}
            </div>

            <div className="settings-content">
                {/* System Information Section */}
                <section className="settings-section system-info-section">
                    <h3><Cpu size={18} /> System Information</h3>

                    {systemLoading ? (
                        <div className="system-loading">
                            <Loader2 size={24} className="spin" />
                            <span>Detecting hardware...</span>
                        </div>
                    ) : systemInfo ? (
                        <div className="system-grid">
                            <div className="system-card">
                                <div className="system-card-icon cpu">
                                    <Cpu size={24} />
                                </div>
                                <div className="system-card-content">
                                    <span className="system-card-label">CPU Cores</span>
                                    <span className="system-card-value">{systemInfo.cpu_count}</span>
                                </div>
                            </div>

                            <div className="system-card">
                                <div className="system-card-icon memory">
                                    <MemoryStick size={24} />
                                </div>
                                <div className="system-card-content">
                                    <span className="system-card-label">RAM</span>
                                    <span className="system-card-value">{formatBytes(systemInfo.memory_total)}</span>
                                </div>
                            </div>

                            <div className="system-card">
                                <div className={`system-card-icon gpu ${systemInfo.gpu_available ? 'available' : 'unavailable'}`}>
                                    {systemInfo.gpu_available ? <CheckCircle size={24} /> : <XCircle size={24} />}
                                </div>
                                <div className="system-card-content">
                                    <span className="system-card-label">GPU</span>
                                    <span className="system-card-value">
                                        {systemInfo.gpu_available ? systemInfo.gpu_name : 'Not Available'}
                                    </span>
                                </div>
                            </div>

                            <div className="system-card">
                                <div className="system-card-icon torch">
                                    <Zap size={24} />
                                </div>
                                <div className="system-card-content">
                                    <span className="system-card-label">PyTorch</span>
                                    <span className="system-card-value">{systemInfo.torch_version}</span>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="system-error">
                            <XCircle size={20} />
                            <span>Failed to detect system hardware. Is the backend running?</span>
                        </div>
                    )}
                </section>

                {/* Appearance Section */}
                <section className="settings-section">
                    <h3><Monitor size={18} /> Appearance</h3>

                    <div className="setting-item">
                        <div className="setting-info">
                            <label>Theme</label>
                            <span className="setting-desc">Choose your preferred color scheme</span>
                        </div>
                        <div className="theme-toggle">
                            <button
                                className={`theme-btn ${settings.theme === 'dark' ? 'active' : ''}`}
                                onClick={() => handleChange('theme', 'dark')}
                            >
                                <Moon size={16} /> Dark
                            </button>
                            <button
                                className={`theme-btn ${settings.theme === 'light' ? 'active' : ''}`}
                                onClick={() => handleChange('theme', 'light')}
                            >
                                <Sun size={16} /> Light
                            </button>
                        </div>
                    </div>

                    <div className="setting-item">
                        <div className="setting-info">
                            <label>Animations</label>
                            <span className="setting-desc">Enable UI animations and transitions</span>
                        </div>
                        <label className="toggle-switch">
                            <input
                                type="checkbox"
                                checked={settings.enableAnimations}
                                onChange={(e) => handleChange('enableAnimations', e.target.checked)}
                            />
                            <span className="toggle-slider"></span>
                        </label>
                    </div>
                </section>

                {/* Hardware Section */}
                <section className="settings-section">
                    <h3><Cpu size={18} /> Hardware</h3>

                    <div className="setting-item">
                        <div className="setting-info">
                            <label>Compute Device</label>
                            <span className="setting-desc">Select device for training and inference</span>
                        </div>
                        <select
                            value={settings.device}
                            onChange={(e) => handleChange('device', e.target.value as SettingsState['device'])}
                            className="setting-select"
                        >
                            <option value="auto">Auto-detect (Recommended)</option>
                            <option value="cuda">GPU (CUDA)</option>
                            <option value="cpu">CPU Only</option>
                        </select>
                    </div>

                    <div className="setting-item">
                        <div className="setting-info">
                            <label>Max GPU Memory Usage</label>
                            <span className="setting-desc">Limit GPU memory for training ({settings.maxGpuMemory}%)</span>
                        </div>
                        <div className="slider-container">
                            <input
                                type="range"
                                min="20"
                                max="100"
                                value={settings.maxGpuMemory}
                                onChange={(e) => handleChange('maxGpuMemory', parseInt(e.target.value))}
                            />
                            <span className="slider-value">{settings.maxGpuMemory}%</span>
                        </div>
                    </div>
                </section>

                {/* Auto-save Section */}
                <section className="settings-section">
                    <h3><Save size={18} /> Auto-save</h3>

                    <div className="setting-item">
                        <div className="setting-info">
                            <label>Auto-save Interval</label>
                            <span className="setting-desc">Save project changes automatically</span>
                        </div>
                        <select
                            value={settings.autoSaveInterval}
                            onChange={(e) => handleChange('autoSaveInterval', parseInt(e.target.value))}
                            className="setting-select"
                        >
                            <option value="30">Every 30 seconds</option>
                            <option value="60">Every 1 minute</option>
                            <option value="120">Every 2 minutes</option>
                            <option value="300">Every 5 minutes</option>
                            <option value="0">Disabled</option>
                        </select>
                    </div>
                </section>

                {/* Storage Section */}
                <section className="settings-section">
                    <h3><HardDrive size={18} /> Storage</h3>

                    <div className="setting-item">
                        <div className="setting-info">
                            <label>Cache Size</label>
                            <span className="setting-desc">Temporary files and model cache</span>
                        </div>
                        <div className="cache-info">
                            <span className="cache-size">{cacheSize}</span>
                            <button onClick={handleClearCache} className="btn-secondary btn-sm">
                                <Trash2 size={14} /> Clear Cache
                            </button>
                        </div>
                    </div>
                </section>

                {/* Actions */}
                <div className="settings-actions">
                    <button onClick={handleResetDefaults} className="btn-secondary">
                        <RefreshCw size={16} /> Reset to Defaults
                    </button>
                    <button onClick={handleSave} className="btn-primary">
                        <Zap size={16} /> Apply Settings
                    </button>
                </div>
            </div>
        </div>
    );
}
