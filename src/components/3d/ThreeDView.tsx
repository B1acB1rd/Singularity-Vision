import { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../../store/appStore';
import {
    Box,
    Play,
    Loader2,
    AlertCircle,
    CheckCircle,
    Ruler,
    Maximize,
    RotateCcw,
    Download,
    Settings,
    Layers,
    Eye,
    EyeOff
} from 'lucide-react';
import './ThreeDView.css';

// API base URL
const API_BASE = 'http://127.0.0.1:8765';

interface ReconstructionJob {
    job_id: string;
    status: string;
    progress: number;
    message: string;
    started_at?: string;
    completed_at?: string;
    error?: string;
    result?: {
        point_cloud_path: string;
        poses_path: string;
        num_points: number;
        num_cameras: number;
        num_images_used: number;
    };
}

interface Point3D {
    x: number;
    y: number;
    z: number;
    color?: [number, number, number];
}

interface ReconstructionConfig {
    feature_detector: 'ORB' | 'SIFT';
    max_features: number;
    match_ratio: number;
    min_matches: number;
}

export default function ThreeDView() {
    const { projectPath } = useAppStore();
    const [jobs, setJobs] = useState<ReconstructionJob[]>([]);
    const [activeJob, setActiveJob] = useState<ReconstructionJob | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showSettings, setShowSettings] = useState(false);
    const [config, setConfig] = useState<ReconstructionConfig>({
        feature_detector: 'ORB',
        max_features: 5000,
        match_ratio: 0.75,
        min_matches: 50
    });

    // Point cloud visualization
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [points, setPoints] = useState<Point3D[]>([]);
    const [rotation, setRotation] = useState({ x: 0, y: 0 });
    const [zoom, setZoom] = useState(1);
    const [showPoints, setShowPoints] = useState(true);

    // Measurement
    const [measureMode, setMeasureMode] = useState(false);
    const [selectedPoints, setSelectedPoints] = useState<Point3D[]>([]);
    const [measureResult, setMeasureResult] = useState<number | null>(null);

    // Polling for job status
    useEffect(() => {
        if (activeJob && activeJob.status !== 'completed' && activeJob.status !== 'failed') {
            const interval = setInterval(() => {
                pollJobStatus(activeJob.job_id);
            }, 1000);
            return () => clearInterval(interval);
        }
    }, [activeJob]);

    // Render point cloud
    useEffect(() => {
        if (points.length > 0 && canvasRef.current) {
            renderPointCloud();
        }
    }, [points, rotation, zoom, showPoints]);

    const pollJobStatus = async (jobId: string) => {
        try {
            const response = await fetch(`${API_BASE}/3d/status/${jobId}`);
            const job = await response.json();
            setActiveJob(job);

            if (job.status === 'completed' && job.result) {
                loadPointCloud(jobId);
            }
        } catch (err) {
            console.error('Failed to poll job status:', err);
        }
    };

    const loadJobs = async () => {
        try {
            const response = await fetch(`${API_BASE}/3d/jobs`);
            const data = await response.json();
            setJobs(data.jobs || []);
        } catch (err) {
            console.error('Failed to load jobs:', err);
        }
    };

    const startReconstruction = async () => {
        if (!projectPath) return;

        setLoading(true);
        setError(null);

        try {
            const imagesDir = `${projectPath}/datasets`;
            const outputDir = `${projectPath}/spatial/reconstructions/${Date.now()}`;

            const response = await fetch(`${API_BASE}/3d/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    images_dir: imagesDir,
                    output_dir: outputDir,
                    project_path: projectPath,
                    config
                })
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            setActiveJob({
                job_id: data.job_id,
                status: 'pending',
                progress: 0,
                message: 'Starting reconstruction...'
            });

        } catch (err: any) {
            setError(err.message || 'Failed to start reconstruction');
        } finally {
            setLoading(false);
        }
    };

    const loadPointCloud = async (jobId: string) => {
        try {
            const response = await fetch(`${API_BASE}/3d/point-cloud/${jobId}?format=json`);
            const data = await response.json();

            if (data.points) {
                setPoints(data.points);
            }
        } catch (err) {
            console.error('Failed to load point cloud:', err);
        }
    };

    const renderPointCloud = () => {
        const canvas = canvasRef.current;
        if (!canvas || !showPoints) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear canvas
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        if (points.length === 0) return;

        // Calculate bounds
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;

        for (const p of points) {
            minX = Math.min(minX, p.x);
            maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y);
            maxY = Math.max(maxY, p.y);
            minZ = Math.min(minZ, p.z);
            maxZ = Math.max(maxZ, p.z);
        }

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        const scale = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

        // Apply rotation
        const cosX = Math.cos(rotation.x);
        const sinX = Math.sin(rotation.x);
        const cosY = Math.cos(rotation.y);
        const sinY = Math.sin(rotation.y);

        // Sort points by depth for proper rendering
        const transformedPoints = points.map(p => {
            // Center the point
            let x = p.x - centerX;
            let y = p.y - centerY;
            let z = p.z - centerZ;

            // Apply Y rotation (horizontal)
            const x1 = x * cosY - z * sinY;
            const z1 = x * sinY + z * cosY;

            // Apply X rotation (vertical)
            const y2 = y * cosX - z1 * sinX;
            const z2 = y * sinX + z1 * cosX;

            return { x: x1, y: y2, z: z2, color: p.color };
        });

        // Sort by depth
        transformedPoints.sort((a, b) => b.z - a.z);

        // Draw points
        const canvasCenterX = canvas.width / 2;
        const canvasCenterY = canvas.height / 2;
        const displayScale = (Math.min(canvas.width, canvas.height) / scale) * 0.8 * zoom;

        for (const p of transformedPoints) {
            const screenX = canvasCenterX + p.x * displayScale;
            const screenY = canvasCenterY - p.y * displayScale;

            // Depth-based size and opacity
            const depth = (p.z + scale / 2) / scale;
            const size = 1 + depth * 2;
            const alpha = 0.3 + depth * 0.7;

            // Color
            const color = p.color || [100, 149, 237]; // Default: cornflower blue
            ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;

            ctx.beginPath();
            ctx.arc(screenX, screenY, size, 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw axes
        drawAxes(ctx, canvas, displayScale, cosX, sinX, cosY, sinY);
    };

    const drawAxes = (
        ctx: CanvasRenderingContext2D,
        canvas: HTMLCanvasElement,
        scale: number,
        cosX: number, sinX: number,
        cosY: number, sinY: number
    ) => {
        const origin = { x: 50, y: canvas.height - 50 };
        const length = 40;

        // Transform axes
        const axes = [
            { x: 1, y: 0, z: 0, color: '#ef4444', label: 'X' },  // Red
            { x: 0, y: 1, z: 0, color: '#22c55e', label: 'Y' },  // Green
            { x: 0, y: 0, z: 1, color: '#3b82f6', label: 'Z' }   // Blue
        ];

        for (const axis of axes) {
            // Apply rotations
            const x1 = axis.x * cosY - axis.z * sinY;
            const z1 = axis.x * sinY + axis.z * cosY;
            const y2 = axis.y * cosX - z1 * sinX;

            const endX = origin.x + x1 * length;
            const endY = origin.y - y2 * length;

            ctx.strokeStyle = axis.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(origin.x, origin.y);
            ctx.lineTo(endX, endY);
            ctx.stroke();

            ctx.fillStyle = axis.color;
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText(axis.label, endX + 5, endY);
        }
    };

    const handleCanvasMouseMove = (e: React.MouseEvent) => {
        if (e.buttons === 1) {  // Left mouse button
            setRotation(prev => ({
                x: prev.x + e.movementY * 0.01,
                y: prev.y + e.movementX * 0.01
            }));
        }
    };

    const handleCanvasWheel = (e: React.WheelEvent) => {
        e.preventDefault();
        setZoom(prev => Math.max(0.1, Math.min(10, prev - e.deltaY * 0.001)));
    };

    const resetView = () => {
        setRotation({ x: 0, y: 0 });
        setZoom(1);
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed': return 'text-emerald-400';
            case 'failed': return 'text-red-400';
            case 'pending': return 'text-yellow-400';
            default: return 'text-blue-400';
        }
    };

    return (
        <div className="threed-view">
            {/* Header */}
            <div className="threed-header">
                <div className="threed-title">
                    <Box size={24} className="text-purple-400" />
                    <h1>3D Reconstruction Lab</h1>
                    <span className="badge-core">Core Feature</span>
                </div>

                <div className="threed-actions">
                    <button
                        className="btn-secondary"
                        onClick={() => setShowSettings(!showSettings)}
                    >
                        <Settings size={16} />
                        Config
                    </button>

                    <button
                        className="btn-primary"
                        onClick={startReconstruction}
                        disabled={loading || !projectPath}
                    >
                        {loading ? (
                            <Loader2 size={16} className="animate-spin" />
                        ) : (
                            <Play size={16} />
                        )}
                        Start Reconstruction
                    </button>
                </div>
            </div>

            {/* Settings Panel */}
            {showSettings && (
                <div className="settings-panel">
                    <h3>Reconstruction Settings</h3>
                    <div className="settings-grid">
                        <div className="setting-item">
                            <label>Feature Detector</label>
                            <select
                                value={config.feature_detector}
                                onChange={e => setConfig(prev => ({
                                    ...prev,
                                    feature_detector: e.target.value as 'ORB' | 'SIFT'
                                }))}
                            >
                                <option value="ORB">ORB (Fast)</option>
                                <option value="SIFT">SIFT (Accurate)</option>
                            </select>
                        </div>

                        <div className="setting-item">
                            <label>Max Features</label>
                            <input
                                type="number"
                                value={config.max_features}
                                onChange={e => setConfig(prev => ({
                                    ...prev,
                                    max_features: parseInt(e.target.value)
                                }))}
                            />
                        </div>

                        <div className="setting-item">
                            <label>Match Ratio</label>
                            <input
                                type="number"
                                step="0.05"
                                min="0.5"
                                max="0.95"
                                value={config.match_ratio}
                                onChange={e => setConfig(prev => ({
                                    ...prev,
                                    match_ratio: parseFloat(e.target.value)
                                }))}
                            />
                        </div>

                        <div className="setting-item">
                            <label>Min Matches</label>
                            <input
                                type="number"
                                value={config.min_matches}
                                onChange={e => setConfig(prev => ({
                                    ...prev,
                                    min_matches: parseInt(e.target.value)
                                }))}
                            />
                        </div>
                    </div>
                </div>
            )}

            {/* Error Message */}
            {error && (
                <div className="error-banner">
                    <AlertCircle size={16} />
                    {error}
                </div>
            )}

            {/* Main Content */}
            <div className="threed-content">
                {/* Left Panel - Job Status */}
                <div className="status-panel">
                    <h3>Current Job</h3>

                    {activeJob ? (
                        <div className="job-status">
                            <div className="job-header">
                                <span className="job-id">#{activeJob.job_id}</span>
                                <span className={`job-status-badge ${getStatusColor(activeJob.status)}`}>
                                    {activeJob.status}
                                </span>
                            </div>

                            <div className="progress-bar">
                                <div
                                    className="progress-fill"
                                    style={{ width: `${activeJob.progress * 100}%` }}
                                />
                            </div>
                            <p className="progress-text">
                                {Math.round(activeJob.progress * 100)}% - {activeJob.message}
                            </p>

                            {activeJob.result && (
                                <div className="job-result">
                                    <div className="result-item">
                                        <span className="label">Points:</span>
                                        <span className="value">{activeJob.result.num_points.toLocaleString()}</span>
                                    </div>
                                    <div className="result-item">
                                        <span className="label">Cameras:</span>
                                        <span className="value">{activeJob.result.num_cameras}</span>
                                    </div>
                                    <div className="result-item">
                                        <span className="label">Images:</span>
                                        <span className="value">{activeJob.result.num_images_used}</span>
                                    </div>
                                </div>
                            )}

                            {activeJob.error && (
                                <div className="job-error">
                                    <AlertCircle size={14} />
                                    {activeJob.error}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="no-job">
                            <Box size={48} className="text-gray-600" />
                            <p>No active reconstruction</p>
                            <span>Click "Start Reconstruction" to begin</span>
                        </div>
                    )}

                    {/* View Controls */}
                    {points.length > 0 && (
                        <div className="view-controls">
                            <h4>View Controls</h4>
                            <div className="control-buttons">
                                <button onClick={resetView} title="Reset View">
                                    <RotateCcw size={16} />
                                </button>
                                <button
                                    onClick={() => setShowPoints(!showPoints)}
                                    title={showPoints ? "Hide Points" : "Show Points"}
                                >
                                    {showPoints ? <Eye size={16} /> : <EyeOff size={16} />}
                                </button>
                                <button
                                    onClick={() => setMeasureMode(!measureMode)}
                                    className={measureMode ? 'active' : ''}
                                    title="Measure Distance"
                                >
                                    <Ruler size={16} />
                                </button>
                            </div>

                            <div className="zoom-control">
                                <label>Zoom: {zoom.toFixed(1)}x</label>
                                <input
                                    type="range"
                                    min="0.1"
                                    max="5"
                                    step="0.1"
                                    value={zoom}
                                    onChange={e => setZoom(parseFloat(e.target.value))}
                                />
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Panel - 3D Viewer */}
                <div className="viewer-panel">
                    <canvas
                        ref={canvasRef}
                        width={800}
                        height={600}
                        onMouseMove={handleCanvasMouseMove}
                        onWheel={handleCanvasWheel}
                        className="point-cloud-canvas"
                    />

                    {points.length === 0 && !activeJob && (
                        <div className="viewer-placeholder">
                            <Box size={64} className="text-gray-600" />
                            <h3>3D Point Cloud Viewer</h3>
                            <p>Start a reconstruction to visualize the 3D model</p>
                            <ul>
                                <li>🖱️ Drag to rotate</li>
                                <li>🔍 Scroll to zoom</li>
                                <li>📏 Use ruler for measurements</li>
                            </ul>
                        </div>
                    )}

                    {points.length > 0 && (
                        <div className="viewer-info">
                            <span>{points.length.toLocaleString()} points</span>
                            <span>Rotation: X={rotation.x.toFixed(2)} Y={rotation.y.toFixed(2)}</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
