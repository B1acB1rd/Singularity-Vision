import { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../../store/appStore';
import { getAvailableModels, startTraining, getTrainingStatus, stopTraining } from '../../services/api';
import {
    Brain,
    Play,
    Square,
    Terminal,
    Loader2
} from 'lucide-react';

interface TrainingStatus {
    status: string;
    progress: number;
    current_epoch: number;
    total_epochs: number;
    logs: string[];
    metrics?: {
        map50: number;
        map5095: number;
        loss: number;
    };
    error?: string;
}

export default function TrainingView() {
    const { projectPath } = useAppStore();
    const [models, setModels] = useState<any[]>([]);
    const [selectedModel, setSelectedModel] = useState('yolov8n.pt');
    const [epochs, setEpochs] = useState(10);
    const [batchSize, setBatchSize] = useState(16);
    const [imgSz, setImgSz] = useState(640);

    const [jobId, setJobId] = useState<string | null>(null);
    const [status, setStatus] = useState<TrainingStatus | null>(null);
    const [isPolling, setIsPolling] = useState(false);

    const logsEndRef = useRef<HTMLDivElement>(null);

    // Fetch models
    useEffect(() => {
        getAvailableModels('detection').then(res => setModels(res.models));
    }, []);

    // Poll status
    useEffect(() => {
        let interval: any;
        if (isPolling && jobId) {
            interval = setInterval(async () => {
                try {
                    const s = await getTrainingStatus(jobId);
                    setStatus(s);
                    if (s.status === 'completed' || s.status === 'failed') {
                        setIsPolling(false);
                    }
                } catch (e) {
                    console.error("Polling error", e);
                    setIsPolling(false);
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isPolling, jobId]);

    // Scroll logs
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [status?.logs.length]);

    const handleStart = async () => {
        if (!projectPath) return;
        try {
            const config = { epochs, batch_size: batchSize, imgsz: imgSz };
            const res = await startTraining(projectPath, selectedModel, config);
            setJobId(res); // API returns struct in api.ts but backend returns {"job_id":...}. Check api.ts parsing.
            // api.ts returns response.data.job_id, so res is string directly.
            setJobId(res);
            setIsPolling(true);
        } catch (e) {
            console.error(e);
        }
    };

    const handleStop = async () => {
        if (!jobId) return;
        await stopTraining(jobId);
        // Let polling catch the status change
    };

    return (
        <div className="flex h-full bg-[#09090b] text-white overflow-hidden">
            {/* Config Panel */}
            <div className="w-80 border-r border-gray-800 p-6 flex flex-col gap-6 overflow-y-auto">
                <div>
                    <h2 className="text-lg font-bold flex items-center gap-2 mb-1">
                        <Brain className="text-indigo-500" /> Model Training
                    </h2>
                    <p className="text-xs text-gray-400">Configure parameters for YOLOv8 training.</p>
                </div>

                <div className="space-y-4">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-300">Base Model</label>
                        <select
                            className="w-full bg-white/5 border border-gray-700 rounded p-2 text-sm focus:border-indigo-500 outline-none"
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            disabled={isPolling}
                        >
                            {models.map(m => (
                                <option key={m.id} value={m.id}>{m.name} ({m.size})</option>
                            ))}
                        </select>
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-300">Epochs</label>
                        <input
                            type="number"
                            className="w-full bg-white/5 border border-gray-700 rounded p-2 text-sm focus:border-indigo-500 outline-none"
                            value={epochs}
                            onChange={(e) => setEpochs(parseInt(e.target.value))}
                            min={1} max={1000}
                            disabled={isPolling}
                        />
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-300">Batch Size</label>
                        <select
                            className="w-full bg-white/5 border border-gray-700 rounded p-2 text-sm focus:border-indigo-500 outline-none"
                            value={batchSize}
                            onChange={(e) => setBatchSize(parseInt(e.target.value))}
                            disabled={isPolling}
                        >
                            <option value={8}>8</option>
                            <option value={16}>16</option>
                            <option value={32}>32</option>
                            <option value={64}>64</option>
                        </select>
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-300">Image Size</label>
                        <select
                            className="w-full bg-white/5 border border-gray-700 rounded p-2 text-sm focus:border-indigo-500 outline-none"
                            value={imgSz}
                            onChange={(e) => setImgSz(parseInt(e.target.value))}
                            disabled={isPolling}
                        >
                            <option value={320}>320</option>
                            <option value={640}>640</option>
                            <option value={1280}>1280</option>
                        </select>
                    </div>
                </div>

                <div className="mt-auto">
                    {isPolling ? (
                        <button
                            onClick={handleStop}
                            className="w-full py-3 rounded bg-red-600/20 text-red-500 hover:bg-red-600/30 border border-red-600/50 flex items-center justify-center gap-2 font-bold transition-all"
                        >
                            <Square size={18} fill="currentColor" /> Stop Training
                        </button>
                    ) : (
                        <button
                            onClick={handleStart}
                            className="w-full py-3 rounded bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center gap-2 font-bold shadow-lg shadow-indigo-600/20 transition-all"
                        >
                            <Play size={18} fill="currentColor" /> Start Training
                        </button>
                    )}
                </div>
            </div>

            {/* Main Content - Logs & Metrics */}
            <div className="flex-1 flex flex-col min-w-0 bg-[#0c0c0e]">
                {/* Header Stats */}
                <div className="h-16 border-b border-gray-800 flex items-center px-6 gap-8">
                    <div className="flex flex-col">
                        <span className="text-xs text-gray-500 uppercase tracking-wider">Status</span>
                        <span className={`text-sm font-bold capitalize flex items-center gap-2 ${status?.status === 'running' ? 'text-green-400' :
                            status?.status === 'failed' ? 'text-red-400' :
                                'text-gray-300'
                            }`}>
                            {status?.status || 'Idle'}
                            {status?.status === 'running' && <Loader2 size={12} className="animate-spin" />}
                        </span>
                    </div>

                    <div className="h-8 w-px bg-white/10"></div>

                    <div className="flex flex-col">
                        <span className="text-xs text-gray-500 uppercase tracking-wider">Progress</span>
                        <span className="text-sm font-bold text-gray-300">
                            {status ? `${Math.round(status.progress)}%` : '0%'}
                            <span className="text-xs text-gray-500 ml-1 font-normal">
                                ({status?.current_epoch || 0}/{status?.total_epochs || epochs} epochs)
                            </span>
                        </span>
                    </div>

                    <div className="h-8 w-px bg-white/10"></div>

                    <div className="flex items-center gap-6 flex-1">
                        <div className="flex flex-col">
                            <span className="text-xs text-gray-500 uppercase tracking-wider">mAP50</span>
                            <span className="text-sm font-mono text-blue-400">
                                {status?.metrics?.map50?.toFixed(4) || '0.000'}
                            </span>
                        </div>
                        <div className="flex flex-col">
                            <span className="text-xs text-gray-500 uppercase tracking-wider">Loss</span>
                            <span className="text-sm font-mono text-yellow-400">
                                {status?.metrics?.loss?.toFixed(4) || '0.000'}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Terminal / Logs */}
                <div className="flex-1 p-6 overflow-hidden flex flex-col">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-bold text-gray-400 flex items-center gap-2">
                            <Terminal size={16} /> Training Logs
                        </h3>
                        <button className="text-xs text-gray-500 hover:text-gray-300">Clear Logs</button>
                    </div>
                    <div className="flex-1 bg-[#1e1e1e] rounded-lg border border-gray-800 p-4 font-mono text-xs overflow-y-auto custom-scrollbar">
                        {!status?.logs || status.logs.length === 0 ? (
                            <div className="text-gray-600 italic">Waiting for training to start...</div>
                        ) : (
                            status.logs.map((log, i) => (
                                <div key={i} className="text-gray-300 mb-1 leading-relaxed border-b border-white/5 pb-1 last:border-0 hover:bg-white/5 transition-colors">
                                    <span className="text-gray-600 mr-2">[{new Date().toLocaleTimeString()}]</span>
                                    {log}
                                </div>
                            ))
                        )}
                        {status?.error && (
                            <div className="text-red-400 mt-2 p-2 bg-red-500/10 rounded border border-red-500/20">
                                <strong>Error:</strong> {status.error}
                            </div>
                        )}
                        <div ref={logsEndRef} />
                    </div>
                </div>
            </div>
        </div>
    );
}

// Add simple CSS for scrollbar if needed or rely on global
const style = document.createElement('style');
style.textContent = `
  .custom-scrollbar::-webkit-scrollbar { width: 8px; }
  .custom-scrollbar::-webkit-scrollbar-track { bg: #1e1e1e; }
  .custom-scrollbar::-webkit-scrollbar-thumb { bg: #333; border-radius: 4px; }
  .custom-scrollbar::-webkit-scrollbar-thumb:hover { bg: #444; }
`;
document.head.appendChild(style);
