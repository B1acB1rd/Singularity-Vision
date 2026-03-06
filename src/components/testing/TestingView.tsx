import './testing.css';
import { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../../store/appStore';
import { getTrainedModels, runImageInference, runVideoInference, runBatchInference, getDatasetPreview } from '../../services/api';
import {
    Play,
    Upload,
    Loader2,
    XCircle,
    Image as ImageIcon,
    Video,
    Film,
    Camera,
    StopCircle,
    FolderOpen,
    Save,
    CheckCircle,
} from 'lucide-react';

interface Detection {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
    class_id: number;
    class_name: string;
}

interface InferenceResult {
    status: string;
    detections: Detection[];
    inference_time_ms: number;
    image_path: string;
}

interface TrainedModel {
    name: string;
    path: string;
    type: string;
}

interface VideoResult {
    status: string;
    frames_processed: number;
    total_detections: number;
    processing_time_s: number;
    avg_fps: number;
    output_path: string | null;
}

export default function TestingView() {
    const { projectPath, classes } = useAppStore();
    const [models, setModels] = useState<TrainedModel[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [testImages, setTestImages] = useState<string[]>([]);
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [confThreshold, setConfThreshold] = useState(0.25);
    const [iouThreshold, setIouThreshold] = useState(0.45);
    const [isRunning, setIsRunning] = useState(false);
    const [result, setResult] = useState<InferenceResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [mode, setMode] = useState<'image' | 'video' | 'webcam' | 'batch'>('image');
    const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
    const [videoResult, setVideoResult] = useState<VideoResult | null>(null);
    const [webcamActive, setWebcamActive] = useState(false);
    const [webcamStats, setWebcamStats] = useState<{ detections: number; fps: number } | null>(null);
    const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
    const [batchResults, setBatchResults] = useState<any[] | null>(null);
    const webcamImgRef = useRef<HTMLImageElement>(null);
    const eventSourceRef = useRef<EventSource | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imageRef = useRef<HTMLImageElement | null>(null);

    // Load trained models
    useEffect(() => {
        async function loadModels() {
            if (!projectPath) return;
            try {
                const trainedModels = await getTrainedModels(projectPath);
                setModels(trainedModels);
                if (trainedModels.length > 0) {
                    setSelectedModel(trainedModels[0].path);
                }
            } catch (e) {
                console.error('Failed to load models:', e);
            }
        }
        loadModels();
    }, [projectPath]);

    // Load test images from dataset
    useEffect(() => {
        async function loadImages() {
            if (!projectPath) return;
            try {
                const data = await getDatasetPreview(projectPath, 1, 100);
                setTestImages(data.images);
            } catch (e) {
                console.error('Failed to load images:', e);
            }
        }
        loadImages();
    }, [projectPath]);

    // Draw detections on canvas
    useEffect(() => {
        if (!canvasRef.current || !imageRef.current || !result) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const img = imageRef.current;

        // Set canvas size to match image
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;

        // Draw image
        ctx.drawImage(img, 0, 0);

        // Draw detections
        result.detections.forEach((det) => {
            const classInfo = classes.find(c => c.id === det.class_id);
            const color = classInfo?.color || '#6366f1';

            // Draw box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

            // Draw label background
            ctx.fillStyle = color;
            const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
            const textWidth = ctx.measureText(label).width + 8;
            ctx.fillRect(det.x1, det.y1 - 24, textWidth, 24);

            // Draw label text
            ctx.fillStyle = '#ffffff';
            ctx.font = '14px Inter, sans-serif';
            ctx.fillText(label, det.x1 + 4, det.y1 - 7);
        });
    }, [result, classes]);

    const handleRunInference = async () => {
        if (!projectPath || !selectedModel || !selectedImage) return;

        setIsRunning(true);
        setError(null);
        setResult(null);

        try {
            const inferenceResult = await runImageInference(
                projectPath,
                selectedModel,
                selectedImage,
                confThreshold,
                iouThreshold
            );
            setResult(inferenceResult);
        } catch (e: any) {
            setError(e.message || 'Inference failed');
        } finally {
            setIsRunning(false);
        }
    };

    const handleImageSelect = (imagePath: string) => {
        setSelectedImage(imagePath);
        setResult(null);
        setError(null);
    };

    const handleUploadImage = async () => {
        if (!window.electronAPI) return;
        try {
            const filePaths = await window.electronAPI.openFiles({
                filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'] }],
            });
            if (filePaths.length > 0) {
                setSelectedImage(filePaths[0]);
                setResult(null);
                setError(null);
                setMode('image');
            }
        } catch (e) {
            console.error(e);
        }
    };

    const handleUploadVideo = async () => {
        if (!window.electronAPI) return;
        try {
            const filePaths = await window.electronAPI.openFiles({
                filters: [{ name: 'Videos', extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm'] }],
            });
            if (filePaths.length > 0) {
                setSelectedVideo(filePaths[0]);
                setVideoResult(null);
                setError(null);
                setMode('video');
            }
        } catch (e) {
            console.error(e);
        }
    };

    const handleRunVideoInference = async () => {
        if (!projectPath || !selectedModel || !selectedVideo) return;

        setIsRunning(true);
        setError(null);
        setVideoResult(null);

        // Generate output path
        const outputDir = `${projectPath}/inference_outputs`;
        const videoName = selectedVideo.split(/[\\/]/).pop()?.replace(/\.[^/.]+$/, '') || 'output';
        const outputPath = `${outputDir}/${videoName}_detected.mp4`;

        try {
            const result = await runVideoInference(
                selectedModel,
                selectedVideo,
                outputPath,
                confThreshold,
                iouThreshold
            );
            setVideoResult(result);
        } catch (e: any) {
            setError(e.message || 'Video inference failed');
        } finally {
            setIsRunning(false);
        }
    };

    const handleRunBatchInference = async () => {
        if (!projectPath || !selectedModel || testImages.length === 0) return;

        setIsRunning(true);
        setError(null);
        setBatchResults(null);
        setBatchProgress({ current: 0, total: testImages.length });

        try {
            const results = await runBatchInference(
                projectPath,
                selectedModel,
                testImages,
                confThreshold,
                iouThreshold
            );
            setBatchResults(results);
            setBatchProgress({ current: testImages.length, total: testImages.length });
        } catch (e: any) {
            setError(e.message || 'Batch inference failed');
        } finally {
            setIsRunning(false);
        }
    };

    const handleSaveResult = async () => {
        if (!result || !selectedImage || !projectPath) return;

        try {
            // Save annotated image using canvas
            const canvas = canvasRef.current;
            if (!canvas) return;

            const outputDir = `${projectPath}/inference_outputs`;
            const imageName = selectedImage.split(/[\\/]/).pop()?.replace(/\.[^/.]+$/, '') || 'output';
            const outputPath = `${outputDir}/${imageName}_detected.jpg`;

            // Convert canvas to blob and save via Electron
            canvas.toBlob(async (blob) => {
                if (blob && window.electronAPI) {
                    const buffer = await blob.arrayBuffer();
                    // Ensure directory exists
                    if (window.electronAPI.mkdir) {
                        await window.electronAPI.mkdir(outputDir);
                    }

                    // Use new writeFileBuffer method
                    if (window.electronAPI.writeFileBuffer) {
                        await window.electronAPI.writeFileBuffer(outputPath, new Uint8Array(buffer));
                        alert(`Saved to: ${outputPath}`);
                    } else {
                        // Fallback if method not available yet (shouldn't happen)
                        console.error("writeFileBuffer not available");
                        alert("Error: Save functionality not fully loaded. Please restart app.");
                    }
                }
            }, 'image/jpeg', 0.95);
        } catch (e: any) {
            setError(e.message || 'Failed to save result');
        }
    };

    const startWebcam = async () => {
        if (!selectedModel) return;

        setWebcamActive(true);
        setError(null);

        // Get backend port dynamically
        const port = await window.electronAPI?.getBackendPort() || 8000;
        const backendUrl = `http://localhost:${port}`;

        // Use fetch with streaming for SSE-like behavior
        try {
            const response = await fetch(`${backendUrl}/inference/webcam/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_path: selectedModel,
                    camera_id: 0,
                    conf_threshold: confThreshold,
                    iou_threshold: iouThreshold,
                }),
            });

            const reader = response.body?.getReader();
            if (!reader) return;

            const decoder = new TextDecoder();

            while (webcamActive) {
                const { done, value } = await reader.read();
                if (done) break;

                const text = decoder.decode(value);
                const lines = text.split('\n').filter(l => l.startsWith('data: '));

                for (const line of lines) {
                    try {
                        const data = JSON.parse(line.replace('data: ', '').replace(/'/g, '"'));
                        if (data.error) {
                            setError(data.error);
                            break;
                        }
                        if (data.frame && webcamImgRef.current) {
                            webcamImgRef.current.src = `data:image/jpeg;base64,${data.frame}`;
                            setWebcamStats({ detections: data.detections, fps: Math.round(1000 / data.inference_time_ms) });
                        }
                    } catch (e) {
                        // Parse error, skip
                    }
                }
            }
        } catch (e: any) {
            setError(e.message || 'Webcam connection failed');
        } finally {
            setWebcamActive(false);
        }
    };

    const stopWebcam = () => {
        setWebcamActive(false);
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
    };

    const encodedImagePath = selectedImage
        ? `media://${encodeURIComponent(selectedImage.replace(/\\/g, '/'))}`
        : null;

    return (
        <div className="testing-view">
            {/* Sidebar - Model & Settings */}
            <div className="testing-sidebar">
                <div className="sidebar-section">
                    <h3>Model</h3>
                    {models.length === 0 ? (
                        <div className="empty-models">
                            <p>No trained models found.</p>
                            <p className="text-muted">Train a model first to run inference.</p>
                        </div>
                    ) : (
                        <select
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            className="model-select"
                        >
                            {models.map((m) => (
                                <option key={m.path} value={m.path}>
                                    {m.name}
                                </option>
                            ))}
                        </select>
                    )}
                </div>

                <div className="sidebar-section">
                    <h3>Settings</h3>
                    <div className="setting-item">
                        <label>Confidence Threshold</label>
                        <div className="slider-row">
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={confThreshold}
                                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                            />
                            <span>{(confThreshold * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                    <div className="setting-item">
                        <label>IoU Threshold</label>
                        <div className="slider-row">
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={iouThreshold}
                                onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                            />
                            <span>{(iouThreshold * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                </div>

                <div className="sidebar-section">
                    <h3>Test Images</h3>
                    <button onClick={handleUploadImage} className="btn-secondary full-width">
                        <Upload size={16} /> Upload Image
                    </button>
                    <div className="image-list">
                        {testImages.slice(0, 20).map((img, idx) => (
                            <div
                                key={idx}
                                className={`image-list-item ${selectedImage === img ? 'active' : ''}`}
                                onClick={() => handleImageSelect(img)}
                            >
                                <img
                                    src={`media://${encodeURIComponent(img.replace(/\\/g, '/'))}`}
                                    alt=""
                                    loading="lazy"
                                />
                            </div>
                        ))}
                    </div>
                </div>

                <div className="sidebar-section">
                    <h3>Test Videos</h3>
                    <button onClick={handleUploadVideo} className="btn-secondary full-width">
                        <Video size={16} /> Upload Video
                    </button>
                    {selectedVideo && (
                        <div className="selected-video">
                            <Film size={16} />
                            <span>{selectedVideo.split(/[\\/]/).pop()}</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Main Content */}
            <div className="testing-main">
                {/* Header */}
                <div className="testing-header">
                    <div className="header-left">
                        <h2>Testing & Inference</h2>
                        <div className="mode-toggle">
                            <button
                                className={`mode-btn ${mode === 'image' ? 'active' : ''}`}
                                onClick={() => setMode('image')}
                            >
                                <ImageIcon size={16} /> Image
                            </button>
                            <button
                                className={`mode-btn ${mode === 'video' ? 'active' : ''}`}
                                onClick={() => setMode('video')}
                            >
                                <Video size={16} /> Video
                            </button>
                            <button
                                className={`mode-btn ${mode === 'webcam' ? 'active' : ''}`}
                                onClick={() => setMode('webcam')}
                            >
                                <Camera size={16} /> Webcam
                            </button>
                            <button
                                className={`mode-btn ${mode === 'batch' ? 'active' : ''}`}
                                onClick={() => setMode('batch')}
                            >
                                <FolderOpen size={16} /> Batch
                            </button>
                        </div>
                        {result && mode === 'image' && (
                            <span className="inference-stats">
                                {result.detections.length} detections • {result.inference_time_ms}ms
                            </span>
                        )}
                        {videoResult && mode === 'video' && (
                            <span className="inference-stats">
                                {videoResult.frames_processed} frames • {videoResult.total_detections} detections • {videoResult.avg_fps} FPS
                            </span>
                        )}
                        {webcamStats && mode === 'webcam' && (
                            <span className="inference-stats">
                                {webcamStats.detections} detections • {webcamStats.fps} FPS
                            </span>
                        )}
                        {batchProgress && mode === 'batch' && (
                            <span className="inference-stats">
                                {batchProgress.current}/{batchProgress.total} images processed
                            </span>
                        )}
                    </div>
                    <div className="header-right">
                        {mode === 'image' ? (
                            <>
                                <button
                                    onClick={handleRunInference}
                                    disabled={!selectedModel || !selectedImage || isRunning}
                                    className="btn-primary"
                                >
                                    {isRunning ? (
                                        <><Loader2 size={18} className="animate-spin" /> Running...</>
                                    ) : (
                                        <><Play size={18} /> Run Inference</>
                                    )}
                                </button>
                                {result && (
                                    <button onClick={handleSaveResult} className="btn-secondary">
                                        <Save size={18} /> Save Result
                                    </button>
                                )}
                            </>
                        ) : mode === 'video' ? (
                            <button
                                onClick={handleRunVideoInference}
                                disabled={!selectedModel || !selectedVideo || isRunning}
                                className="btn-primary"
                            >
                                {isRunning ? (
                                    <><Loader2 size={18} className="animate-spin" /> Processing Video...</>
                                ) : (
                                    <><Play size={18} /> Process Video</>
                                )}
                            </button>
                        ) : mode === 'webcam' ? (
                            <button
                                onClick={webcamActive ? stopWebcam : startWebcam}
                                disabled={!selectedModel}
                                className={webcamActive ? "btn-danger" : "btn-primary"}
                            >
                                {webcamActive ? (
                                    <><StopCircle size={18} /> Stop Webcam</>
                                ) : (
                                    <><Camera size={18} /> Start Webcam</>
                                )}
                            </button>
                        ) : (
                            <button
                                onClick={handleRunBatchInference}
                                disabled={!selectedModel || testImages.length === 0 || isRunning}
                                className="btn-primary"
                            >
                                {isRunning ? (
                                    <><Loader2 size={18} className="animate-spin" /> Processing Batch...</>
                                ) : (
                                    <><FolderOpen size={18} /> Run Batch ({testImages.length} images)</>
                                )}
                            </button>
                        )}
                    </div>
                </div>

                {/* Canvas Area */}
                <div className="canvas-area">
                    {mode === 'webcam' ? (
                        <div className="webcam-container">
                            {webcamActive ? (
                                <img ref={webcamImgRef} alt="Webcam feed" className="webcam-feed" />
                            ) : (
                                <div className="empty-state">
                                    <Camera size={48} />
                                    <p>Click "Start Webcam" to begin live detection</p>
                                </div>
                            )}
                        </div>
                    ) : mode === 'batch' ? (
                        <div className="batch-results-container">
                            {batchResults ? (
                                <div className="batch-results-grid">
                                    {batchResults.map((res: any, idx: number) => (
                                        <div key={idx} className="batch-item">
                                            <div className="batch-item-header">
                                                <span className="batch-filename">{res.image_path.split(/[\\/]/).pop()}</span>
                                                <span className="batch-time">{res.inference_time_ms}ms</span>
                                            </div>
                                            <div className="batch-detections">
                                                {res.detections.length} detections
                                            </div>
                                            {res.status === 'success' ? (
                                                <CheckCircle size={16} className="text-green-500" />
                                            ) : (
                                                <XCircle size={16} className="text-red-500" />
                                            )}
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="empty-state">
                                    <FolderOpen size={48} />
                                    <p>Select a model and click "Run Batch" to process all images in the test set.</p>
                                    <p className="text-sm text-muted">Results will be saved to your project folder.</p>
                                </div>
                            )}
                        </div>
                    ) : !selectedImage && mode === 'image' ? (
                        <div className="empty-state">
                            <ImageIcon size={48} />
                            <p>Select an image from the sidebar or upload one to test</p>
                        </div>
                    ) : mode === 'image' ? (
                        <div className="canvas-wrapper">
                            {/* Hidden image for drawing */}
                            <img
                                ref={imageRef}
                                src={encodedImagePath || ''}
                                alt=""
                                style={{ display: 'none' }}
                                onLoad={() => {
                                    // Trigger redraw when image loads
                                    if (result) {
                                        const event = new Event('redraw');
                                        canvasRef.current?.dispatchEvent(event);
                                    }
                                }}
                            />

                            {result ? (
                                <canvas ref={canvasRef} className="result-canvas" />
                            ) : (
                                <img src={encodedImagePath || ''} alt="" className="preview-image" />
                            )}
                        </div>
                    ) : (
                        <div className="empty-state">
                            <Video size={48} />
                            <p>Upload a video to process with object detection</p>
                        </div>
                    )}

                    {error && (
                        <div className="error-banner">
                            <XCircle size={18} />
                            {error}
                        </div>
                    )}
                </div>

                {/* Detections Panel */}
                {result && result.detections.length > 0 && (
                    <div className="detections-panel">
                        <h4>Detections ({result.detections.length})</h4>
                        <div className="detections-list">
                            {result.detections.map((det, idx) => (
                                <div key={idx} className="detection-item">
                                    <span className="det-class">{det.class_name}</span>
                                    <span className="det-conf">{(det.confidence * 100).toFixed(1)}%</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
