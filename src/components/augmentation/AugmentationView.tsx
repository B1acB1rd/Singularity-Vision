import './augmentation.css';
import { useState, useEffect } from 'react';
import { useAppStore } from '../../store/appStore';
import { previewAugmentation, batchAugment, getDatasetPreview } from '../../services/api';
import {
    Wand2,
    RotateCcw,
    Sun,
    Droplets,
    Maximize,
    FlipHorizontal,
    CircleDot,
    Eye,
    Save,
    RefreshCw,
    Loader2,
    Play,
    ImageIcon,
} from 'lucide-react';

interface AugmentationConfig {
    // Preprocessing
    resize: { enabled: boolean; width: number; height: number };
    normalize: boolean;
    grayscale: boolean;
    crop: { enabled: boolean; size: number };

    // Augmentations
    horizontalFlip: { enabled: boolean; prob: number };
    verticalFlip: { enabled: boolean; prob: number };
    rotation: { enabled: boolean; prob: number; maxAngle: number };
    brightness: { enabled: boolean; prob: number; range: number };
    contrast: { enabled: boolean; prob: number; range: number };
    blur: { enabled: boolean; prob: number; maxKernel: number };
    noise: { enabled: boolean; prob: number; intensity: number };
    cutout: { enabled: boolean; prob: number; size: number };
}

const DEFAULT_CONFIG: AugmentationConfig = {
    resize: { enabled: true, width: 640, height: 640 },
    normalize: true,
    grayscale: false,
    crop: { enabled: false, size: 512 },

    horizontalFlip: { enabled: true, prob: 0.5 },
    verticalFlip: { enabled: false, prob: 0.5 },
    rotation: { enabled: true, prob: 0.3, maxAngle: 15 },
    brightness: { enabled: true, prob: 0.3, range: 0.2 },
    contrast: { enabled: true, prob: 0.3, range: 0.2 },
    blur: { enabled: false, prob: 0.1, maxKernel: 5 },
    noise: { enabled: false, prob: 0.1, intensity: 0.02 },
    cutout: { enabled: false, prob: 0.2, size: 50 },
};

export default function AugmentationView() {
    const { projectPath } = useAppStore();
    const [config, setConfig] = useState<AugmentationConfig>(DEFAULT_CONFIG);
    const [images, setImages] = useState<string[]>([]);
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [previewImage, setPreviewImage] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [batchLoading, setBatchLoading] = useState(false);

    // Load images from dataset
    useEffect(() => {
        if (projectPath) {
            getDatasetPreview(projectPath, 1, 20)
                .then((res) => {
                    setImages(res.images || []);
                    if (res.images && res.images.length > 0) {
                        setSelectedImage(res.images[0]);
                    }
                })
                .catch(() => setImages([]));
        }
    }, [projectPath]);

    const updateConfig = <K extends keyof AugmentationConfig>(
        key: K,
        value: AugmentationConfig[K]
    ) => {
        setConfig((prev) => ({ ...prev, [key]: value }));
    };

    // Convert frontend config to backend format
    const toBackendConfig = () => ({
        resize: config.resize,
        normalize: config.normalize,
        grayscale: config.grayscale,
        horizontal_flip: { enabled: config.horizontalFlip.enabled, prob: config.horizontalFlip.prob },
        vertical_flip: { enabled: config.verticalFlip.enabled, prob: config.verticalFlip.prob },
        rotation: { enabled: config.rotation.enabled, prob: config.rotation.prob, max_angle: config.rotation.maxAngle },
        brightness: { enabled: config.brightness.enabled, prob: config.brightness.prob, range: config.brightness.range },
        contrast: { enabled: config.contrast.enabled, prob: config.contrast.prob, range: config.contrast.range },
        blur: { enabled: config.blur.enabled, prob: config.blur.prob, max_kernel: config.blur.maxKernel },
        noise: { enabled: config.noise.enabled, prob: config.noise.prob, intensity: config.noise.intensity },
        cutout: { enabled: config.cutout.enabled, prob: config.cutout.prob, size: config.cutout.size },
    });

    const handlePreview = async () => {
        if (!projectPath || !selectedImage) return;

        setLoading(true);
        try {
            const result = await previewAugmentation(selectedImage, projectPath, toBackendConfig());
            setPreviewImage(result.image);
        } catch (err) {
            console.error('Preview failed:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleBatchAugment = async () => {
        if (!projectPath) return;

        setBatchLoading(true);
        try {
            const result = await batchAugment(projectPath, toBackendConfig(), 3);
            alert(`Created ${result.augmented_count} augmented images in ${result.output_dir}`);
        } catch (err: any) {
            alert(`Batch augmentation failed: ${err.message}`);
        } finally {
            setBatchLoading(false);
        }
    };

    const handleSaveConfig = () => {
        if (projectPath) {
            localStorage.setItem(`aug-config-${projectPath}`, JSON.stringify(config));
            alert('Augmentation config saved!');
        }
    };

    const handleResetConfig = () => {
        setConfig(DEFAULT_CONFIG);
        setPreviewImage(null);
    };

    return (
        <div className="augmentation-view">
            <div className="aug-header">
                <div>
                    <h2><Wand2 size={20} /> Data Augmentation</h2>
                    <p>Configure preprocessing and augmentation pipeline</p>
                </div>
                <div className="aug-actions">
                    <button onClick={handleResetConfig} className="btn-secondary">
                        <RefreshCw size={16} /> Reset
                    </button>
                    <button onClick={handleSaveConfig} className="btn-primary">
                        <Save size={16} /> Save Config
                    </button>
                </div>
            </div>

            <div className="aug-content">
                {/* Preprocessing Section */}
                <section className="aug-section">
                    <h3>Preprocessing</h3>

                    <div className="aug-item">
                        <div className="aug-item-header">
                            <label className="aug-toggle">
                                <input
                                    type="checkbox"
                                    checked={config.resize.enabled}
                                    onChange={(e) =>
                                        updateConfig('resize', { ...config.resize, enabled: e.target.checked })
                                    }
                                />
                                <span className="toggle-slider-small"></span>
                            </label>
                            <span className="aug-label"><Maximize size={16} /> Resize</span>
                        </div>
                        {config.resize.enabled && (
                            <div className="aug-params">
                                <div className="param-group">
                                    <label>Width</label>
                                    <input
                                        type="number"
                                        value={config.resize.width}
                                        onChange={(e) =>
                                            updateConfig('resize', { ...config.resize, width: parseInt(e.target.value) })
                                        }
                                    />
                                </div>
                                <div className="param-group">
                                    <label>Height</label>
                                    <input
                                        type="number"
                                        value={config.resize.height}
                                        onChange={(e) =>
                                            updateConfig('resize', { ...config.resize, height: parseInt(e.target.value) })
                                        }
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="aug-item simple">
                        <label className="aug-toggle">
                            <input
                                type="checkbox"
                                checked={config.normalize}
                                onChange={(e) => updateConfig('normalize', e.target.checked)}
                            />
                            <span className="toggle-slider-small"></span>
                        </label>
                        <span className="aug-label"><CircleDot size={16} /> Normalize (0-1)</span>
                    </div>

                    <div className="aug-item simple">
                        <label className="aug-toggle">
                            <input
                                type="checkbox"
                                checked={config.grayscale}
                                onChange={(e) => updateConfig('grayscale', e.target.checked)}
                            />
                            <span className="toggle-slider-small"></span>
                        </label>
                        <span className="aug-label"><Eye size={16} /> Grayscale</span>
                    </div>
                </section>

                {/* Augmentations Section */}
                <section className="aug-section">
                    <h3>Augmentations</h3>

                    <div className="aug-item">
                        <div className="aug-item-header">
                            <label className="aug-toggle">
                                <input
                                    type="checkbox"
                                    checked={config.horizontalFlip.enabled}
                                    onChange={(e) =>
                                        updateConfig('horizontalFlip', { ...config.horizontalFlip, enabled: e.target.checked })
                                    }
                                />
                                <span className="toggle-slider-small"></span>
                            </label>
                            <span className="aug-label"><FlipHorizontal size={16} /> Horizontal Flip</span>
                        </div>
                        {config.horizontalFlip.enabled && (
                            <div className="aug-params">
                                <div className="param-group full">
                                    <label>Probability</label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.1"
                                        value={config.horizontalFlip.prob}
                                        onChange={(e) =>
                                            updateConfig('horizontalFlip', { ...config.horizontalFlip, prob: parseFloat(e.target.value) })
                                        }
                                    />
                                    <span>{(config.horizontalFlip.prob * 100).toFixed(0)}%</span>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="aug-item">
                        <div className="aug-item-header">
                            <label className="aug-toggle">
                                <input
                                    type="checkbox"
                                    checked={config.rotation.enabled}
                                    onChange={(e) =>
                                        updateConfig('rotation', { ...config.rotation, enabled: e.target.checked })
                                    }
                                />
                                <span className="toggle-slider-small"></span>
                            </label>
                            <span className="aug-label"><RotateCcw size={16} /> Rotation</span>
                        </div>
                        {config.rotation.enabled && (
                            <div className="aug-params">
                                <div className="param-group">
                                    <label>Probability</label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.1"
                                        value={config.rotation.prob}
                                        onChange={(e) =>
                                            updateConfig('rotation', { ...config.rotation, prob: parseFloat(e.target.value) })
                                        }
                                    />
                                    <span>{(config.rotation.prob * 100).toFixed(0)}%</span>
                                </div>
                                <div className="param-group">
                                    <label>Max Angle</label>
                                    <input
                                        type="number"
                                        value={config.rotation.maxAngle}
                                        onChange={(e) =>
                                            updateConfig('rotation', { ...config.rotation, maxAngle: parseInt(e.target.value) })
                                        }
                                    />
                                    <span>°</span>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="aug-item">
                        <div className="aug-item-header">
                            <label className="aug-toggle">
                                <input
                                    type="checkbox"
                                    checked={config.brightness.enabled}
                                    onChange={(e) =>
                                        updateConfig('brightness', { ...config.brightness, enabled: e.target.checked })
                                    }
                                />
                                <span className="toggle-slider-small"></span>
                            </label>
                            <span className="aug-label"><Sun size={16} /> Brightness</span>
                        </div>
                        {config.brightness.enabled && (
                            <div className="aug-params">
                                <div className="param-group full">
                                    <label>Probability</label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.1"
                                        value={config.brightness.prob}
                                        onChange={(e) =>
                                            updateConfig('brightness', { ...config.brightness, prob: parseFloat(e.target.value) })
                                        }
                                    />
                                    <span>{(config.brightness.prob * 100).toFixed(0)}%</span>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="aug-item">
                        <div className="aug-item-header">
                            <label className="aug-toggle">
                                <input
                                    type="checkbox"
                                    checked={config.blur.enabled}
                                    onChange={(e) =>
                                        updateConfig('blur', { ...config.blur, enabled: e.target.checked })
                                    }
                                />
                                <span className="toggle-slider-small"></span>
                            </label>
                            <span className="aug-label"><Droplets size={16} /> Blur</span>
                        </div>
                        {config.blur.enabled && (
                            <div className="aug-params">
                                <div className="param-group full">
                                    <label>Probability</label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.1"
                                        value={config.blur.prob}
                                        onChange={(e) =>
                                            updateConfig('blur', { ...config.blur, prob: parseFloat(e.target.value) })
                                        }
                                    />
                                    <span>{(config.blur.prob * 100).toFixed(0)}%</span>
                                </div>
                            </div>
                        )}
                    </div>
                </section>

                {/* Preview Panel */}
                <section className="aug-preview">
                    <h3><ImageIcon size={16} /> Preview</h3>

                    {/* Image Selector */}
                    <div className="preview-controls">
                        <select
                            className="image-select"
                            value={selectedImage || ''}
                            onChange={(e) => setSelectedImage(e.target.value)}
                            disabled={images.length === 0}
                        >
                            {images.length === 0 ? (
                                <option value="">No images in dataset</option>
                            ) : (
                                images.map((img) => (
                                    <option key={img} value={img}>
                                        {img.split(/[\\/]/).pop()}
                                    </option>
                                ))
                            )}
                        </select>

                        <button
                            className="btn-preview"
                            onClick={handlePreview}
                            disabled={loading || !selectedImage}
                        >
                            {loading ? (
                                <><Loader2 size={14} className="animate-spin" /> Generating...</>
                            ) : (
                                <><Play size={14} /> Preview</>
                            )}
                        </button>
                    </div>

                    {/* Preview Display */}
                    <div className="preview-display">
                        {previewImage ? (
                            <img src={previewImage} alt="Augmented preview" />
                        ) : (
                            <div className="preview-placeholder">
                                <Wand2 size={32} />
                                <p>Click Preview to see augmentation result</p>
                            </div>
                        )}
                    </div>

                    {/* Batch Augment */}
                    <button
                        className="btn-batch"
                        onClick={handleBatchAugment}
                        disabled={batchLoading}
                    >
                        {batchLoading ? (
                            <><Loader2 size={14} className="animate-spin" /> Processing...</>
                        ) : (
                            <><Wand2 size={14} /> Apply to All Images (3x)</>
                        )}
                    </button>
                </section>
            </div>
        </div>
    );
}
