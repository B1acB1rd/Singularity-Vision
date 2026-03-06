import './export.css';
import { useState, useEffect } from 'react';
import { useAppStore } from '../../store/appStore';
import { getExportableModels, getExportFormats, exportModel } from '../../services/api';
import {
    Download,
    Loader2,
    CheckCircle,
    Folder,
    FileCode,
    Package,
    Settings2,
} from 'lucide-react';

interface ExportableModel {
    name: string;
    path: string;
    size_mb: number;
}

interface ExportFormat {
    id: string;
    name: string;
    description: string;
}

export default function ExportView() {
    const { projectPath } = useAppStore();
    const [models, setModels] = useState<ExportableModel[]>([]);
    const [formats, setFormats] = useState<ExportFormat[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [selectedFormat, setSelectedFormat] = useState<string>('onnx');
    const [outputDir, setOutputDir] = useState<string>('');
    const [imgsz, setImgsz] = useState<number>(640);
    const [halfPrecision, setHalfPrecision] = useState<boolean>(false);
    const [dynamicAxes, setDynamicAxes] = useState<boolean>(false);
    const [isExporting, setIsExporting] = useState<boolean>(false);
    const [exportResult, setExportResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    // Load models and formats
    useEffect(() => {
        async function loadData() {
            if (!projectPath) return;
            try {
                const [modelList, formatList] = await Promise.all([
                    getExportableModels(projectPath),
                    getExportFormats()
                ]);
                setModels(modelList);
                setFormats(formatList);
                if (modelList.length > 0) {
                    setSelectedModel(modelList[0].path);
                }
                // Default output directory
                setOutputDir(`${projectPath}/exports`);
            } catch (e) {
                console.error('Failed to load export data:', e);
            }
        }
        loadData();
    }, [projectPath]);

    const handleSelectOutputDir = async () => {
        if (!window.electronAPI) return;
        try {
            const folder = await window.electronAPI.openDirectory();
            if (folder) {
                setOutputDir(folder);
            }
        } catch (e) {
            console.error(e);
        }
    };

    const handleExport = async () => {
        if (!selectedModel || !outputDir) return;

        setIsExporting(true);
        setError(null);
        setExportResult(null);

        try {
            const result = await exportModel(
                selectedModel,
                outputDir,
                selectedFormat,
                imgsz,
                halfPrecision,
                dynamicAxes
            );
            setExportResult(result);
        } catch (e: any) {
            setError(e.message || 'Export failed');
        } finally {
            setIsExporting(false);
        }
    };

    return (
        <div className="export-view">
            {/* Header */}
            <div className="export-header">
                <div>
                    <h2>Export Model</h2>
                    <p>Export your trained model for deployment</p>
                </div>
            </div>

            <div className="export-content">
                {/* Model Selection */}
                <div className="export-section">
                    <h3><Package size={18} /> Select Model</h3>
                    {models.length === 0 ? (
                        <div className="empty-state">
                            <p>No trained models found.</p>
                            <p className="text-muted">Train a model first to export it.</p>
                        </div>
                    ) : (
                        <div className="model-grid">
                            {models.map((model) => (
                                <div
                                    key={model.path}
                                    className={`model-card ${selectedModel === model.path ? 'selected' : ''}`}
                                    onClick={() => setSelectedModel(model.path)}
                                >
                                    <div className="model-icon">
                                        <FileCode size={24} />
                                    </div>
                                    <div className="model-info">
                                        <span className="model-name">{model.name}</span>
                                        <span className="model-size">{model.size_mb} MB</span>
                                    </div>
                                    {selectedModel === model.path && (
                                        <CheckCircle size={18} className="check-icon" />
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Format Selection */}
                <div className="export-section">
                    <h3><Download size={18} /> Export Format</h3>
                    <div className="format-grid">
                        {formats.map((fmt) => (
                            <div
                                key={fmt.id}
                                className={`format-card ${selectedFormat === fmt.id ? 'selected' : ''}`}
                                onClick={() => setSelectedFormat(fmt.id)}
                            >
                                <span className="format-name">{fmt.name}</span>
                                <span className="format-desc">{fmt.description}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Export Settings */}
                <div className="export-section">
                    <h3><Settings2 size={18} /> Export Settings</h3>

                    <div className="settings-grid">
                        <div className="setting-item">
                            <label>Output Directory</label>
                            <div className="path-input">
                                <input
                                    type="text"
                                    value={outputDir}
                                    onChange={(e) => setOutputDir(e.target.value)}
                                    placeholder="Select output folder..."
                                />
                                <button onClick={handleSelectOutputDir} className="btn-secondary">
                                    <Folder size={16} />
                                </button>
                            </div>
                        </div>

                        <div className="setting-item">
                            <label>Image Size</label>
                            <select value={imgsz} onChange={(e) => setImgsz(parseInt(e.target.value))}>
                                <option value={320}>320</option>
                                <option value={416}>416</option>
                                <option value={512}>512</option>
                                <option value={640}>640 (default)</option>
                                <option value={768}>768</option>
                                <option value={1024}>1024</option>
                            </select>
                        </div>

                        <div className="setting-item checkbox">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={halfPrecision}
                                    onChange={(e) => setHalfPrecision(e.target.checked)}
                                />
                                FP16 Half Precision
                            </label>
                            <span className="setting-hint">Reduces model size, may affect accuracy</span>
                        </div>

                        <div className="setting-item checkbox">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={dynamicAxes}
                                    onChange={(e) => setDynamicAxes(e.target.checked)}
                                />
                                Dynamic Input Size
                            </label>
                            <span className="setting-hint">Allows variable input dimensions</span>
                        </div>
                    </div>
                </div>

                {/* Export Button */}
                <div className="export-actions">
                    <button
                        onClick={handleExport}
                        disabled={!selectedModel || !outputDir || isExporting}
                        className="btn-primary btn-lg"
                    >
                        {isExporting ? (
                            <><Loader2 size={20} className="animate-spin" /> Exporting...</>
                        ) : (
                            <><Download size={20} /> Export Model</>
                        )}
                    </button>
                </div>

                {/* Result */}
                {exportResult && (
                    <div className="export-result success">
                        <CheckCircle size={24} />
                        <div>
                            <h4>Export Complete!</h4>
                            <p>Format: {exportResult.format.toUpperCase()}</p>
                            <p className="path">Saved to: {exportResult.export_path}</p>
                            {exportResult.script_path && (
                                <p className="path">Inference script: {exportResult.script_path}</p>
                            )}
                        </div>
                    </div>
                )}

                {error && (
                    <div className="export-result error">
                        <p>Error: {error}</p>
                    </div>
                )}
            </div>
        </div>
    );
}
