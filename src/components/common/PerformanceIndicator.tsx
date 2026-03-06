import { useState, useEffect } from 'react';
import {
    Cpu,
    HardDrive,
    Activity,
    Zap,
    AlertTriangle,
    CheckCircle,
    ChevronDown,
    ChevronUp,
    Monitor
} from 'lucide-react';
import './PerformanceIndicator.css';

// API base URL
const API_BASE = 'http://127.0.0.1:8765';

interface HardwareInfo {
    cpu: {
        cores: number;
        logical_cores: number;
        frequency_mhz: number;
        usage_percent: number;
        score: number;
    };
    memory: {
        total_gb: number;
        available_gb: number;
        used_percent: number;
        score: number;
    };
    disk: {
        total_gb: number;
        free_gb: number;
        used_percent: number;
    };
    gpu: {
        available: boolean;
        name: string | null;
        memory_gb: number | null;
        memory_free_gb: number | null;
        cuda_version: string | null;
        score: number;
    };
    platform: {
        os: string;
        os_version: string;
        python_version: string;
    };
    overall_score: number;
}

interface PerformanceIndicatorProps {
    compact?: boolean;
}

export default function PerformanceIndicator({ compact = false }: PerformanceIndicatorProps) {
    const [hardware, setHardware] = useState<HardwareInfo | null>(null);
    const [loading, setLoading] = useState(true);
    const [expanded, setExpanded] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadHardwareInfo();
        // Refresh every 30 seconds
        const interval = setInterval(loadHardwareInfo, 30000);
        return () => clearInterval(interval);
    }, []);

    const loadHardwareInfo = async () => {
        try {
            const response = await fetch(`${API_BASE}/orchestrator/hardware`);
            if (!response.ok) throw new Error('Failed to load hardware info');
            const data = await response.json();
            setHardware(data);
            setError(null);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const getScoreColor = (score: number) => {
        if (score >= 80) return 'text-emerald-400';
        if (score >= 50) return 'text-yellow-400';
        return 'text-red-400';
    };

    const getScoreLabel = (score: number) => {
        if (score >= 80) return 'Excellent';
        if (score >= 60) return 'Good';
        if (score >= 40) return 'Fair';
        return 'Limited';
    };

    const getScoreBg = (score: number) => {
        if (score >= 80) return 'bg-emerald-500/10';
        if (score >= 50) return 'bg-yellow-500/10';
        return 'bg-red-500/10';
    };

    if (loading) {
        return (
            <div className="perf-indicator compact">
                <Activity size={14} className="animate-pulse" />
                <span className="text-xs text-gray-500">Loading...</span>
            </div>
        );
    }

    if (error || !hardware) {
        return (
            <div className="perf-indicator compact error">
                <AlertTriangle size={14} className="text-yellow-500" />
                <span className="text-xs text-gray-500">System info unavailable</span>
            </div>
        );
    }

    // Compact view for sidebar
    if (compact) {
        return (
            <div
                className={`perf-indicator compact ${getScoreBg(hardware.overall_score)}`}
                onClick={() => setExpanded(!expanded)}
            >
                <div className="perf-compact-main">
                    <Monitor size={14} className={getScoreColor(hardware.overall_score)} />
                    <span className={`text-xs font-medium ${getScoreColor(hardware.overall_score)}`}>
                        {hardware.overall_score}%
                    </span>
                    <span className="text-xs text-gray-500">
                        {hardware.gpu.available ? 'GPU' : 'CPU'}
                    </span>
                    {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                </div>

                {expanded && (
                    <div className="perf-compact-details">
                        <div className="perf-detail-row">
                            <Cpu size={12} />
                            <span>{hardware.cpu.cores} cores</span>
                        </div>
                        <div className="perf-detail-row">
                            <HardDrive size={12} />
                            <span>{hardware.memory.available_gb.toFixed(1)} GB free</span>
                        </div>
                        {hardware.gpu.available && (
                            <div className="perf-detail-row">
                                <Zap size={12} className="text-emerald-400" />
                                <span className="text-emerald-400">{hardware.gpu.name?.split(' ').pop()}</span>
                            </div>
                        )}
                    </div>
                )}
            </div>
        );
    }

    // Full view
    return (
        <div className="perf-indicator full">
            <div className="perf-header">
                <h3>System Performance</h3>
                <div className={`perf-score ${getScoreBg(hardware.overall_score)}`}>
                    <span className={`score-value ${getScoreColor(hardware.overall_score)}`}>
                        {hardware.overall_score}
                    </span>
                    <span className="score-label">{getScoreLabel(hardware.overall_score)}</span>
                </div>
            </div>

            <div className="perf-metrics">
                {/* CPU */}
                <div className="perf-metric">
                    <div className="metric-header">
                        <Cpu size={16} className="text-blue-400" />
                        <span className="metric-name">CPU</span>
                        <span className={`metric-score ${getScoreColor(hardware.cpu.score)}`}>
                            {hardware.cpu.score}%
                        </span>
                    </div>
                    <div className="metric-bar">
                        <div
                            className="metric-fill bg-blue-500"
                            style={{ width: `${hardware.cpu.score}%` }}
                        />
                    </div>
                    <div className="metric-details">
                        <span>{hardware.cpu.cores} cores ({hardware.cpu.logical_cores} threads)</span>
                        <span>{hardware.cpu.usage_percent?.toFixed(0)}% used</span>
                    </div>
                </div>

                {/* Memory */}
                <div className="perf-metric">
                    <div className="metric-header">
                        <HardDrive size={16} className="text-purple-400" />
                        <span className="metric-name">Memory</span>
                        <span className={`metric-score ${getScoreColor(hardware.memory.score)}`}>
                            {hardware.memory.score}%
                        </span>
                    </div>
                    <div className="metric-bar">
                        <div
                            className="metric-fill bg-purple-500"
                            style={{ width: `${hardware.memory.used_percent}%` }}
                        />
                    </div>
                    <div className="metric-details">
                        <span>{hardware.memory.total_gb.toFixed(1)} GB total</span>
                        <span>{hardware.memory.available_gb.toFixed(1)} GB free</span>
                    </div>
                </div>

                {/* GPU */}
                <div className="perf-metric">
                    <div className="metric-header">
                        <Zap size={16} className={hardware.gpu.available ? 'text-emerald-400' : 'text-gray-500'} />
                        <span className="metric-name">GPU</span>
                        {hardware.gpu.available ? (
                            <CheckCircle size={14} className="text-emerald-400" />
                        ) : (
                            <AlertTriangle size={14} className="text-yellow-500" />
                        )}
                    </div>
                    {hardware.gpu.available ? (
                        <>
                            <div className="metric-bar">
                                <div
                                    className="metric-fill bg-emerald-500"
                                    style={{ width: '100%' }}
                                />
                            </div>
                            <div className="metric-details">
                                <span>{hardware.gpu.name}</span>
                                {hardware.gpu.memory_gb && (
                                    <span>{hardware.gpu.memory_gb.toFixed(1)} GB VRAM</span>
                                )}
                            </div>
                            {hardware.gpu.cuda_version && (
                                <div className="gpu-cuda">
                                    CUDA {hardware.gpu.cuda_version}
                                </div>
                            )}
                        </>
                    ) : (
                        <div className="no-gpu">
                            <span>No GPU detected</span>
                            <span className="text-xs text-gray-500">Training will use CPU (slower)</span>
                        </div>
                    )}
                </div>

                {/* Disk */}
                <div className="perf-metric">
                    <div className="metric-header">
                        <Activity size={16} className="text-orange-400" />
                        <span className="metric-name">Disk</span>
                    </div>
                    <div className="metric-bar">
                        <div
                            className="metric-fill bg-orange-500"
                            style={{ width: `${hardware.disk.used_percent}%` }}
                        />
                    </div>
                    <div className="metric-details">
                        <span>{hardware.disk.total_gb.toFixed(0)} GB total</span>
                        <span>{hardware.disk.free_gb.toFixed(1)} GB free</span>
                    </div>
                </div>
            </div>

            <div className="perf-footer">
                <span>{hardware.platform.os} {hardware.platform.os_version}</span>
                <span>•</span>
                <span>Python {hardware.platform.python_version}</span>
            </div>
        </div>
    );
}
