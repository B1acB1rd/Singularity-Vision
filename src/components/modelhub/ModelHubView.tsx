import './modelhub.css';
import { useState, useEffect } from 'react';
import { useAppStore } from '../../store/appStore';
import { downloadYoloModel, getProjectModels } from '../../services/api';
import {
    Zap,
    Target,
    Layers,
    Download,
    CheckCircle,
    Info,
    AlertCircle,
} from 'lucide-react';

interface ModelConfig {
    id: string;
    name: string;
    category: 'detection' | 'classification' | 'segmentation';
    size: string;
    params: string;
    mAP: string;
    speed: string;
    description: string;
    downloadUrl: string;
}

const AVAILABLE_MODELS: ModelConfig[] = [
    // Detection Models
    {
        id: 'yolov8n',
        name: 'YOLOv8 Nano',
        category: 'detection',
        size: '6.3 MB',
        params: '3.2M',
        mAP: '37.3%',
        speed: '80.4 ms',
        description: 'Fastest model, ideal for edge devices and real-time applications.',
        downloadUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
    },
    {
        id: 'yolov8s',
        name: 'YOLOv8 Small',
        category: 'detection',
        size: '22.5 MB',
        params: '11.2M',
        mAP: '44.9%',
        speed: '128.4 ms',
        description: 'Good balance of speed and accuracy for general use.',
        downloadUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
    },
    {
        id: 'yolov8m',
        name: 'YOLOv8 Medium',
        category: 'detection',
        size: '52.0 MB',
        params: '25.9M',
        mAP: '50.2%',
        speed: '234.7 ms',
        description: 'Higher accuracy for applications with more compute resources.',
        downloadUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt',
    },
    {
        id: 'yolov8l',
        name: 'YOLOv8 Large',
        category: 'detection',
        size: '87.7 MB',
        params: '43.7M',
        mAP: '52.9%',
        speed: '375.2 ms',
        description: 'High accuracy model for demanding detection tasks.',
        downloadUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt',
    },
    {
        id: 'yolov8x',
        name: 'YOLOv8 XLarge',
        category: 'detection',
        size: '136.7 MB',
        params: '68.2M',
        mAP: '53.9%',
        speed: '479.1 ms',
        description: 'Maximum accuracy, best for offline processing and research.',
        downloadUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt',
    },
    // Classification Models
    {
        id: 'yolov8n-cls',
        name: 'YOLOv8 Nano Classifier',
        category: 'classification',
        size: '5.0 MB',
        params: '2.7M',
        mAP: '66.6%',
        speed: '12.9 ms',
        description: 'Fast classification for edge deployment.',
        downloadUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt',
    },
    {
        id: 'yolov8s-cls',
        name: 'YOLOv8 Small Classifier',
        category: 'classification',
        size: '13.5 MB',
        params: '6.4M',
        mAP: '72.3%',
        speed: '23.4 ms',
        description: 'Balanced classification model.',
        downloadUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-cls.pt',
    },
    // Segmentation Models
    {
        id: 'yolov8n-seg',
        name: 'YOLOv8 Nano Segmentation',
        category: 'segmentation',
        size: '7.0 MB',
        params: '3.4M',
        mAP: '30.5%',
        speed: '96.1 ms',
        description: 'Fast instance segmentation for real-time applications.',
        downloadUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt',
    },
    {
        id: 'yolov8s-seg',
        name: 'YOLOv8 Small Segmentation',
        category: 'segmentation',
        size: '23.8 MB',
        params: '11.8M',
        mAP: '36.8%',
        speed: '155.7 ms',
        description: 'Balanced segmentation model for general use.',
        downloadUrl: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt',
    },
];

type CategoryFilter = 'all' | 'detection' | 'classification' | 'segmentation';

export default function ModelHubView() {
    const { projectPath } = useAppStore();
    const [selectedCategory, setSelectedCategory] = useState<CategoryFilter>('all');
    const [selectedModel, setSelectedModel] = useState<string | null>(null);
    const [downloading, setDownloading] = useState<string | null>(null);
    const [downloadedModels, setDownloadedModels] = useState<string[]>([]);
    const [error, setError] = useState<string | null>(null);

    // Load already-downloaded models on mount
    useEffect(() => {
        if (projectPath) {
            loadProjectModels();
        }
    }, [projectPath]);

    const loadProjectModels = async () => {
        if (!projectPath) return;
        try {
            const models = await getProjectModels(projectPath);
            // Map model filename (e.g., 'yolov8n.pt') to model id (e.g., 'yolov8n')
            const downloaded = models.map(m => m.name.replace('.pt', ''));
            setDownloadedModels(downloaded);
        } catch (e) {
            console.error('Failed to load project models:', e);
        }
    };

    const filteredModels = selectedCategory === 'all'
        ? AVAILABLE_MODELS
        : AVAILABLE_MODELS.filter(m => m.category === selectedCategory);

    const getCategoryIcon = (category: string) => {
        switch (category) {
            case 'detection': return <Target size={16} />;
            case 'classification': return <Layers size={16} />;
            case 'segmentation': return <Zap size={16} />;
            default: return null;
        }
    };

    const handleDownload = async (model: ModelConfig) => {
        if (!projectPath) {
            setError('No project selected');
            return;
        }

        setDownloading(model.id);
        setError(null);

        try {
            await downloadYoloModel(
                `${model.id}.pt`,
                projectPath,
                model.downloadUrl
            );
            setDownloadedModels(prev => [...prev, model.id]);
        } catch (e: any) {
            setError(e.message || 'Download failed');
            console.error('Download failed:', e);
        } finally {
            setDownloading(null);
        }
    };

    const handleSelectModel = (model: ModelConfig) => {
        setSelectedModel(model.id);
    };

    return (
        <div className="modelhub-view">
            {/* Error Display */}
            {error && (
                <div className="p-3 mx-6 mt-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400">
                    <AlertCircle size={18} />
                    <span>{error}</span>
                </div>
            )}

            {/* Header */}
            <div className="modelhub-header">
                <div>
                    <h2>Model Hub</h2>
                    <p>Select a pre-trained model to fine-tune on your dataset</p>
                </div>
            </div>

            {/* Category Filter */}
            <div className="category-filter">
                <button
                    className={`filter-btn ${selectedCategory === 'all' ? 'active' : ''}`}
                    onClick={() => setSelectedCategory('all')}
                >
                    All Models
                </button>
                <button
                    className={`filter-btn ${selectedCategory === 'detection' ? 'active' : ''}`}
                    onClick={() => setSelectedCategory('detection')}
                >
                    <Target size={16} /> Detection
                </button>
                <button
                    className={`filter-btn ${selectedCategory === 'classification' ? 'active' : ''}`}
                    onClick={() => setSelectedCategory('classification')}
                >
                    <Layers size={16} /> Classification
                </button>
                <button
                    className={`filter-btn ${selectedCategory === 'segmentation' ? 'active' : ''}`}
                    onClick={() => setSelectedCategory('segmentation')}
                >
                    <Zap size={16} /> Segmentation
                </button>
            </div>

            {/* Model Grid */}
            <div className="model-grid">
                {filteredModels.map((model) => (
                    <div
                        key={model.id}
                        className={`model-card ${selectedModel === model.id ? 'selected' : ''}`}
                        onClick={() => handleSelectModel(model)}
                    >
                        <div className="model-card-header">
                            <div className="model-category">
                                {getCategoryIcon(model.category)}
                                <span>{model.category}</span>
                            </div>
                            {downloadedModels.includes(model.id) && (
                                <CheckCircle size={18} className="downloaded-icon" />
                            )}
                        </div>

                        <h3 className="model-name">{model.name}</h3>
                        <p className="model-description">{model.description}</p>

                        <div className="model-stats">
                            <div className="stat">
                                <span className="stat-label">Size</span>
                                <span className="stat-value">{model.size}</span>
                            </div>
                            <div className="stat">
                                <span className="stat-label">Params</span>
                                <span className="stat-value">{model.params}</span>
                            </div>
                            <div className="stat">
                                <span className="stat-label">mAP</span>
                                <span className="stat-value">{model.mAP}</span>
                            </div>
                            <div className="stat">
                                <span className="stat-label">Speed</span>
                                <span className="stat-value">{model.speed}</span>
                            </div>
                        </div>

                        <div className="model-actions">
                            {downloadedModels.includes(model.id) ? (
                                <button className="btn-secondary" disabled>
                                    <CheckCircle size={16} /> Downloaded
                                </button>
                            ) : (
                                <button
                                    className="btn-primary"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleDownload(model);
                                    }}
                                    disabled={downloading === model.id}
                                >
                                    {downloading === model.id ? (
                                        'Downloading...'
                                    ) : (
                                        <><Download size={16} /> Download</>
                                    )}
                                </button>
                            )}
                            <button className="btn-icon" title="View Details">
                                <Info size={16} />
                            </button>
                        </div>
                    </div>
                ))}
            </div>

            {/* Selected Model Info */}
            {selectedModel && (
                <div className="selected-model-bar">
                    <span>Selected: <strong>{AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name}</strong></span>
                    <button className="btn-primary">
                        Use for Training
                    </button>
                </div>
            )}
        </div>
    );
}
