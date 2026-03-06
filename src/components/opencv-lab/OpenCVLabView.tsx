import './opencv-lab.css';
import { useState, useEffect, useRef } from 'react';
import {
    Sparkles,
    Upload,
    Download,
    RotateCcw,
    ChevronDown,
    ChevronRight,
    Loader2,
    ImageIcon,
    Sliders,
    Wand2,
    Scan,
    Palette,
    Move,
    Shapes,
    Contrast,
    ScanLine,
} from 'lucide-react';

// Operation definitions
const CATEGORIES = [
    {
        id: 'filter',
        name: 'Filtering',
        icon: 'blur',
        operations: [
            { id: 'gaussian-blur', name: 'Gaussian Blur', params: { kernel_size: { type: 'range', min: 1, max: 31, step: 2, default: 5 }, sigma_x: { type: 'range', min: 0, max: 10, step: 0.5, default: 0 } } },
            { id: 'median-blur', name: 'Median Blur', params: { kernel_size: { type: 'range', min: 1, max: 31, step: 2, default: 5 } } },
            { id: 'bilateral', name: 'Bilateral Filter', params: { d: { type: 'range', min: 1, max: 15, default: 9 }, sigma_color: { type: 'range', min: 10, max: 150, default: 75 }, sigma_space: { type: 'range', min: 10, max: 150, default: 75 } } },
            { id: 'box-blur', name: 'Box Blur', params: { kernel_size: { type: 'range', min: 1, max: 31, default: 5 } } },
            { id: 'sharpen-kernel', name: 'Sharpen', params: { intensity: { type: 'range', min: 0.1, max: 3, step: 0.1, default: 1 } } },
            { id: 'emboss', name: 'Emboss', params: {} },
        ]
    },
    {
        id: 'edge',
        name: 'Edge Detection',
        icon: 'scan-line',
        operations: [
            { id: 'canny', name: 'Canny Edge', params: { threshold1: { type: 'range', min: 0, max: 300, default: 100 }, threshold2: { type: 'range', min: 0, max: 300, default: 200 } } },
            { id: 'sobel', name: 'Sobel', params: { dx: { type: 'select', options: [0, 1], default: 1 }, dy: { type: 'select', options: [0, 1], default: 0 }, ksize: { type: 'range', min: 1, max: 7, step: 2, default: 3 } } },
            { id: 'laplacian', name: 'Laplacian', params: { ksize: { type: 'range', min: 1, max: 7, step: 2, default: 3 } } },
            { id: 'scharr', name: 'Scharr', params: { dx: { type: 'select', options: [0, 1], default: 1 }, dy: { type: 'select', options: [0, 1], default: 0 } } },
        ]
    },
    {
        id: 'morph',
        name: 'Morphology',
        icon: 'shapes',
        operations: [
            { id: 'erode', name: 'Erosion', params: { kernel_size: { type: 'range', min: 1, max: 21, default: 5 }, iterations: { type: 'range', min: 1, max: 10, default: 1 } } },
            { id: 'dilate', name: 'Dilation', params: { kernel_size: { type: 'range', min: 1, max: 21, default: 5 }, iterations: { type: 'range', min: 1, max: 10, default: 1 } } },
            { id: 'open', name: 'Opening', params: { kernel_size: { type: 'range', min: 1, max: 21, default: 5 } } },
            { id: 'close', name: 'Closing', params: { kernel_size: { type: 'range', min: 1, max: 21, default: 5 } } },
            { id: 'gradient', name: 'Gradient', params: { kernel_size: { type: 'range', min: 1, max: 21, default: 5 } } },
            { id: 'tophat', name: 'Top Hat', params: { kernel_size: { type: 'range', min: 1, max: 21, default: 9 } } },
            { id: 'blackhat', name: 'Black Hat', params: { kernel_size: { type: 'range', min: 1, max: 21, default: 9 } } },
        ]
    },
    {
        id: 'threshold',
        name: 'Thresholding',
        icon: 'contrast',
        operations: [
            { id: 'simple', name: 'Simple Threshold', params: { threshold: { type: 'range', min: 0, max: 255, default: 127 }, threshold_type: { type: 'select', options: ['binary', 'binary_inv', 'trunc', 'tozero'], default: 'binary' } } },
            { id: 'adaptive', name: 'Adaptive Threshold', params: { method: { type: 'select', options: ['gaussian', 'mean'], default: 'gaussian' }, block_size: { type: 'range', min: 3, max: 51, step: 2, default: 11 }, c: { type: 'range', min: -10, max: 20, default: 2 } } },
            { id: 'otsu', name: "Otsu's Threshold", params: {} },
            { id: 'inrange', name: 'Color Range (HSV)', params: { h_min: { type: 'range', min: 0, max: 180, default: 0 }, h_max: { type: 'range', min: 0, max: 180, default: 180 }, s_min: { type: 'range', min: 0, max: 255, default: 0 }, s_max: { type: 'range', min: 0, max: 255, default: 255 }, v_min: { type: 'range', min: 0, max: 255, default: 0 }, v_max: { type: 'range', min: 0, max: 255, default: 255 } } },
        ]
    },
    {
        id: 'color',
        name: 'Color Operations',
        icon: 'palette',
        operations: [
            { id: 'grayscale', name: 'Grayscale', params: {} },
            { id: 'hsv', name: 'Convert to HSV', params: {} },
            { id: 'equalize', name: 'Histogram Equalization', params: {} },
            { id: 'clahe', name: 'CLAHE', params: { clip_limit: { type: 'range', min: 1, max: 10, step: 0.5, default: 2 }, tile_grid_size: { type: 'range', min: 2, max: 16, default: 8 } } },
            { id: 'invert', name: 'Invert Colors', params: {} },
            { id: 'brightness-contrast', name: 'Brightness/Contrast', params: { brightness: { type: 'range', min: -100, max: 100, default: 0 }, contrast: { type: 'range', min: 0.5, max: 3, step: 0.1, default: 1 } } },
            { id: 'gamma', name: 'Gamma Correction', params: { gamma: { type: 'range', min: 0.1, max: 3, step: 0.1, default: 1 } } },
        ]
    },
    {
        id: 'transform',
        name: 'Transforms',
        icon: 'move',
        operations: [
            { id: 'resize', name: 'Resize', params: { scale: { type: 'range', min: 0.1, max: 3, step: 0.1, default: 1 } } },
            { id: 'rotate', name: 'Rotate', params: { angle: { type: 'range', min: -180, max: 180, default: 0 } } },
            { id: 'flip', name: 'Flip', params: { mode: { type: 'select', options: ['horizontal', 'vertical', 'both'], default: 'horizontal' } } },
            { id: 'crop', name: 'Crop', params: { x: { type: 'range', min: 0, max: 500, default: 0 }, y: { type: 'range', min: 0, max: 500, default: 0 }, width: { type: 'range', min: 50, max: 1000, default: 200 }, height: { type: 'range', min: 50, max: 1000, default: 200 } } },
        ]
    },
    {
        id: 'features',
        name: 'Feature Detection',
        icon: 'scan',
        operations: [
            { id: 'contours', name: 'Find Contours', params: { threshold: { type: 'range', min: 0, max: 255, default: 127 } } },
            { id: 'harris', name: 'Harris Corners', params: { block_size: { type: 'range', min: 2, max: 10, default: 2 }, k: { type: 'range', min: 0.01, max: 0.1, step: 0.01, default: 0.04 } } },
            { id: 'hough-lines', name: 'Hough Lines', params: { threshold: { type: 'range', min: 50, max: 200, default: 100 }, min_line_length: { type: 'range', min: 10, max: 200, default: 50 } } },
            { id: 'hough-circles', name: 'Hough Circles', params: { min_dist: { type: 'range', min: 10, max: 200, default: 50 }, param1: { type: 'range', min: 50, max: 200, default: 100 }, param2: { type: 'range', min: 10, max: 100, default: 30 } } },
        ]
    },
    {
        id: 'advanced',
        name: 'Advanced',
        icon: 'sparkles',
        operations: [
            { id: 'sharpen', name: 'Unsharp Mask', params: { strength: { type: 'range', min: 0.1, max: 3, step: 0.1, default: 1 } } },
            { id: 'denoise', name: 'Denoise (NLM)', params: { h: { type: 'range', min: 1, max: 30, default: 10 } } },
            { id: 'dft', name: 'DFT Spectrum', params: {} },
            { id: 'histogram', name: 'Histogram', params: { channel: { type: 'select', options: ['all', 'red', 'green', 'blue', 'gray'], default: 'all' } } },
            { id: 'watershed', name: 'Watershed', params: { marker_threshold: { type: 'range', min: 10, max: 90, default: 50 } } },
            { id: 'distance-transform', name: 'Distance Transform', params: { distance_type: { type: 'select', options: ['l1', 'l2', 'c'], default: 'l2' } } },
            { id: 'skeleton', name: 'Skeletonize', params: {} },
            { id: 'inpaint', name: 'Inpainting', params: { radius: { type: 'range', min: 1, max: 10, default: 3 }, method: { type: 'select', options: ['telea', 'ns'], default: 'telea' } } },
            { id: 'convex-hull', name: 'Convex Hull', params: { threshold: { type: 'range', min: 0, max: 255, default: 127 } } },
            { id: 'bounding-rects', name: 'Bounding Rectangles', params: { threshold: { type: 'range', min: 0, max: 255, default: 127 } } },
        ]
    }
];

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
    'filter': <Wand2 size={16} />,
    'edge': <ScanLine size={16} />,
    'morph': <Shapes size={16} />,
    'threshold': <Contrast size={16} />,
    'color': <Palette size={16} />,
    'transform': <Move size={16} />,
    'features': <Scan size={16} />,
    'advanced': <Sparkles size={16} />,
};

export default function OpenCVLabView() {
    const [originalImage, setOriginalImage] = useState<File | null>(null);
    const [originalUrl, setOriginalUrl] = useState<string | null>(null);
    const [resultUrl, setResultUrl] = useState<string | null>(null);
    const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['filter']));
    const [selectedOp, setSelectedOp] = useState<{ category: string; op: any } | null>(null);
    const [params, setParams] = useState<Record<string, any>>({});
    const [processing, setProcessing] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Initialize params when operation changes
    useEffect(() => {
        if (selectedOp) {
            const defaults: Record<string, any> = {};
            Object.entries(selectedOp.op.params).forEach(([key, config]: [string, any]) => {
                defaults[key] = config.default;
            });
            setParams(defaults);
        }
    }, [selectedOp]);

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setOriginalImage(file);
            const url = URL.createObjectURL(file);
            setOriginalUrl(url);
            setResultUrl(null);
        }
    };

    const toggleCategory = (id: string) => {
        setExpandedCategories(prev => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    const selectOperation = (category: typeof CATEGORIES[0], op: any) => {
        setSelectedOp({ category: category.id, op });
    };

    const applyOperation = async () => {
        if (!originalImage || !selectedOp) return;

        setProcessing(true);
        try {
            const formData = new FormData();
            formData.append('image', originalImage);

            Object.entries(params).forEach(([key, value]) => {
                formData.append(key, String(value));
            });

            const port = await (window as any).electronAPI?.getBackendPort?.() || 8765;
            const url = `http://127.0.0.1:${port}/opencv/${selectedOp.category}/${selectedOp.op.id}`;

            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Failed: ${response.statusText}`);
            }

            const blob = await response.blob();
            const resultObjectUrl = URL.createObjectURL(blob);
            setResultUrl(resultObjectUrl);

        } catch (error) {
            console.error('Operation failed:', error);
            alert('Operation failed. Check console for details.');
        } finally {
            setProcessing(false);
        }
    };

    const downloadResult = () => {
        if (!resultUrl) return;
        const a = document.createElement('a');
        a.href = resultUrl;
        a.download = `opencv_result_${selectedOp?.op.id || 'image'}.png`;
        a.click();
    };

    const resetAll = () => {
        setResultUrl(null);
        if (selectedOp) {
            const defaults: Record<string, any> = {};
            Object.entries(selectedOp.op.params).forEach(([key, config]: [string, any]) => {
                defaults[key] = config.default;
            });
            setParams(defaults);
        }
    };

    return (
        <div className="opencv-lab">
            {/* Operations Sidebar */}
            <div className="opencv-sidebar">
                <div className="opencv-sidebar-header">
                    <h2><Sparkles size={20} /> OpenCV Lab</h2>
                    <p>Visual image processing playground</p>
                </div>

                <div className="opencv-categories">
                    {CATEGORIES.map(category => (
                        <div key={category.id} className="opencv-category">
                            <div
                                className={`opencv-category-header ${expandedCategories.has(category.id) ? 'active' : ''}`}
                                onClick={() => toggleCategory(category.id)}
                            >
                                <div className="opencv-category-icon">
                                    {CATEGORY_ICONS[category.id]}
                                </div>
                                {category.name}
                                <span className="ml-auto">
                                    {expandedCategories.has(category.id) ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                                </span>
                            </div>

                            {expandedCategories.has(category.id) && (
                                <div className="opencv-category-ops">
                                    {category.operations.map(op => (
                                        <button
                                            key={op.id}
                                            className={`opencv-op-btn ${selectedOp?.op.id === op.id ? 'active' : ''}`}
                                            onClick={() => selectOperation(category, op)}
                                        >
                                            {op.name}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Main Content */}
            <div className="opencv-main">
                {/* Toolbar */}
                <div className="opencv-toolbar">
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileUpload}
                        accept="image/*"
                        className="hidden"
                        style={{ display: 'none' }}
                    />
                    <button
                        className="opencv-toolbar-btn opencv-toolbar-btn-primary"
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <Upload size={18} /> Upload Image
                    </button>

                    {resultUrl && (
                        <>
                            <button
                                className="opencv-toolbar-btn opencv-toolbar-btn-secondary"
                                onClick={downloadResult}
                            >
                                <Download size={18} /> Download Result
                            </button>
                            <button
                                className="opencv-toolbar-btn opencv-toolbar-btn-secondary"
                                onClick={resetAll}
                            >
                                <RotateCcw size={18} /> Reset
                            </button>
                        </>
                    )}
                </div>

                {/* Workspace */}
                <div className="opencv-workspace">
                    {/* Original Image */}
                    <div className="opencv-image-panel">
                        <div className="opencv-image-panel-header">
                            Original
                        </div>
                        <div className="opencv-image-container">
                            {originalUrl ? (
                                <img src={originalUrl} alt="Original" />
                            ) : (
                                <div className="opencv-image-placeholder">
                                    <ImageIcon />
                                    <span>Upload an image to get started</span>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Result Image */}
                    <div className="opencv-image-panel">
                        <div className="opencv-image-panel-header">
                            Result
                            {processing && <Loader2 size={14} className="animate-spin" />}
                        </div>
                        <div className="opencv-image-container" style={{ position: 'relative' }}>
                            {resultUrl ? (
                                <img src={resultUrl} alt="Result" />
                            ) : (
                                <div className="opencv-image-placeholder">
                                    <Wand2 />
                                    <span>Select an operation and click Apply</span>
                                </div>
                            )}
                            {processing && (
                                <div className="opencv-processing">
                                    <Loader2 size={40} className="animate-spin text-indigo-500" />
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Parameters Panel */}
                    <div className="opencv-params-panel">
                        <div className="opencv-params-header">
                            <h3><Sliders size={16} /> Parameters</h3>
                        </div>

                        {selectedOp ? (
                            <>
                                <div className="opencv-params-list">
                                    <div className="mb-4 pb-4 border-b border-white/10">
                                        <div className="text-xs text-gray-500 uppercase tracking-wider">Operation</div>
                                        <div className="text-lg font-bold text-indigo-400">{selectedOp.op.name}</div>
                                    </div>

                                    {Object.entries(selectedOp.op.params).length === 0 ? (
                                        <div className="text-sm text-gray-500 text-center py-4">
                                            No parameters needed
                                        </div>
                                    ) : (
                                        Object.entries(selectedOp.op.params).map(([key, config]: [string, any]) => (
                                            <div key={key} className="opencv-param">
                                                <div className="opencv-param-value">
                                                    <label>{key.replace(/_/g, ' ')}</label>
                                                    <span>{params[key]}</span>
                                                </div>

                                                {config.type === 'range' ? (
                                                    <input
                                                        type="range"
                                                        min={config.min}
                                                        max={config.max}
                                                        step={config.step || 1}
                                                        value={params[key] ?? config.default}
                                                        onChange={(e) => setParams(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                                                    />
                                                ) : config.type === 'select' ? (
                                                    <select
                                                        value={params[key] ?? config.default}
                                                        onChange={(e) => setParams(p => ({ ...p, [key]: e.target.value }))}
                                                    >
                                                        {config.options.map((opt: any) => (
                                                            <option key={opt} value={opt}>{opt}</option>
                                                        ))}
                                                    </select>
                                                ) : (
                                                    <input
                                                        type="number"
                                                        value={params[key] ?? config.default}
                                                        onChange={(e) => setParams(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                                                    />
                                                )}
                                            </div>
                                        ))
                                    )}
                                </div>

                                <div className="opencv-params-footer">
                                    <button
                                        className="opencv-apply-btn"
                                        onClick={applyOperation}
                                        disabled={!originalImage || processing}
                                    >
                                        {processing ? (
                                            <><Loader2 size={18} className="animate-spin" /> Processing...</>
                                        ) : (
                                            <><Sparkles size={18} /> Apply Operation</>
                                        )}
                                    </button>
                                </div>
                            </>
                        ) : (
                            <div className="opencv-no-op">
                                <Sliders />
                                <div className="text-sm">
                                    Select an operation from the sidebar
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
