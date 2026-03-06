import './evaluation.css';
import { useState, useEffect } from 'react';
import { useAppStore } from '../../store/appStore';
import { runEvaluation, exportEvaluationReport, getTrainedModels } from '../../services/api';
import {
    BarChart3,
    Target,
    TrendingUp,
    FileText,
    Download,
    RefreshCw,
    CheckCircle,
} from 'lucide-react';

interface EvaluationMetrics {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    mAP: number;
    class_metrics: { [key: string]: { precision: number; recall: number; ap: number } };
    confusion_matrix: number[][];
    total_predictions: number;
    total_ground_truth: number;
    class_names?: string[];
}

export default function EvaluationView() {
    const { projectPath } = useAppStore();
    const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
    const [loading, setLoading] = useState(false);
    const [classNames, setClassNames] = useState<string[]>([]);
    const [models, setModels] = useState<any[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [error, setError] = useState<string | null>(null);

    // Load available models
    useEffect(() => {
        if (projectPath) {
            getTrainedModels(projectPath)
                .then((res) => {
                    setModels(res || []);
                    if (res && res.length > 0) {
                        setSelectedModel(res[0].path);
                    }
                })
                .catch(() => setModels([]));
        }
    }, [projectPath]);

    const handleRunEvaluation = async () => {
        if (!projectPath || !selectedModel) {
            setError('Please select a model first');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const result = await runEvaluation(projectPath, selectedModel);
            setMetrics(result);
            setClassNames(result.class_names || Object.keys(result.class_metrics));
        } catch (err: any) {
            setError(err.message || 'Evaluation failed');
        } finally {
            setLoading(false);
        }
    };

    const handleExportReport = async (format: 'pdf' | 'json' | 'csv') => {
        if (!metrics || !projectPath) return;

        try {
            if (format === 'json' || format === 'csv') {
                const result = await exportEvaluationReport(projectPath, format);
                alert(`${format.toUpperCase()} report exported to: ${result.output_path}`);
            } else {
                // For PDF, use local export
                const blob = new Blob([JSON.stringify(metrics, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'evaluation_report.json';
                a.click();
            }
        } catch (err: any) {
            alert(`Export failed: ${err.message}`);
        }
    };

    const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

    return (
        <div className="evaluation-view">
            <div className="eval-header">
                <div>
                    <h2><BarChart3 size={20} /> Evaluation Suite</h2>
                    <p>Analyze model performance with detailed metrics</p>
                </div>
                <div className="eval-actions">
                    <select
                        className="model-select"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        disabled={loading}
                    >
                        {models.length === 0 ? (
                            <option value="">No models available</option>
                        ) : (
                            models.map((m: any) => (
                                <option key={m.path} value={m.path}>
                                    {m.name || m.path.split('/').pop()}
                                </option>
                            ))
                        )}
                    </select>
                    <button
                        onClick={handleRunEvaluation}
                        disabled={loading || !selectedModel}
                        className="btn-primary"
                    >
                        {loading ? (
                            <><RefreshCw size={16} className="animate-spin" /> Evaluating...</>
                        ) : (
                            <><Target size={16} /> Run Evaluation</>
                        )}
                    </button>
                </div>
            </div>

            {error && (
                <div className="eval-error">
                    <span>⚠️ {error}</span>
                </div>
            )}

            <div className="eval-content">
                {!metrics ? (
                    <div className="eval-empty">
                        <BarChart3 size={48} />
                        <h3>No Evaluation Data</h3>
                        <p>Run an evaluation to see metrics and performance analysis</p>
                    </div>
                ) : (
                    <>
                        {/* Summary Cards */}
                        <div className="metrics-summary">
                            <div className="metric-card">
                                <div className="metric-icon"><Target size={20} /></div>
                                <div className="metric-info">
                                    <span className="metric-label">Accuracy</span>
                                    <span className="metric-value">{formatPercent(metrics.accuracy)}</span>
                                </div>
                            </div>
                            <div className="metric-card">
                                <div className="metric-icon"><TrendingUp size={20} /></div>
                                <div className="metric-info">
                                    <span className="metric-label">Precision</span>
                                    <span className="metric-value">{formatPercent(metrics.precision)}</span>
                                </div>
                            </div>
                            <div className="metric-card">
                                <div className="metric-icon"><CheckCircle size={20} /></div>
                                <div className="metric-info">
                                    <span className="metric-label">Recall</span>
                                    <span className="metric-value">{formatPercent(metrics.recall)}</span>
                                </div>
                            </div>
                            <div className="metric-card highlight">
                                <div className="metric-icon"><BarChart3 size={20} /></div>
                                <div className="metric-info">
                                    <span className="metric-label">mAP@0.5</span>
                                    <span className="metric-value">{formatPercent(metrics.mAP)}</span>
                                </div>
                            </div>
                        </div>

                        {/* Per-Class Metrics */}
                        <section className="eval-section">
                            <h3>Per-Class Performance</h3>
                            <table className="metrics-table">
                                <thead>
                                    <tr>
                                        <th>Class</th>
                                        <th>Precision</th>
                                        <th>Recall</th>
                                        <th>AP</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(metrics.class_metrics).map(([className, m]) => (
                                        <tr key={className}>
                                            <td>{className}</td>
                                            <td>{formatPercent(m.precision)}</td>
                                            <td>{formatPercent(m.recall)}</td>
                                            <td>{formatPercent(m.ap)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </section>

                        {/* Confusion Matrix */}
                        <section className="eval-section">
                            <h3>Confusion Matrix</h3>
                            <div className="confusion-matrix">
                                <div className="matrix-grid">
                                    <div className="matrix-header"></div>
                                    {classNames.map((name) => (
                                        <div key={name} className="matrix-header">{name}</div>
                                    ))}
                                    {metrics.confusion_matrix.map((row, i) => (
                                        <>
                                            <div key={`label-${i}`} className="matrix-label">{classNames[i]}</div>
                                            {row.map((val, j) => (
                                                <div
                                                    key={`${i}-${j}`}
                                                    className={`matrix-cell ${i === j ? 'diagonal' : ''}`}
                                                    style={{
                                                        backgroundColor: i === j
                                                            ? `rgba(16, 185, 129, ${val / 50})`
                                                            : `rgba(239, 68, 68, ${val / 50})`
                                                    }}
                                                >
                                                    {val}
                                                </div>
                                            ))}
                                        </>
                                    ))}
                                </div>
                            </div>
                        </section>

                        {/* Export Options */}
                        <div className="export-section">
                            <h3><FileText size={18} /> Export Report</h3>
                            <div className="export-buttons">
                                <button onClick={() => handleExportReport('json')} className="btn-secondary">
                                    <Download size={16} /> JSON
                                </button>
                                <button onClick={() => handleExportReport('csv')} className="btn-secondary">
                                    <Download size={16} /> CSV
                                </button>
                                <button onClick={() => handleExportReport('pdf')} className="btn-secondary">
                                    <Download size={16} /> PDF
                                </button>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
