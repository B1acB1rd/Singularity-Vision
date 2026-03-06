import './annotate.css';
import { useState, useEffect } from 'react';
import { useAppStore } from '../../store/appStore';
import { getDatasetPreview, loadAnnotations, saveAnnotations } from '../../services/api';
import type { BoundingBox, Polygon } from '../../types/project';
import Canvas, { type AnnotationTool } from './Canvas';
import {
    MousePointer2,
    Square,
    Hexagon,
    ZoomIn,
    ZoomOut,
    ArrowLeft,
    ArrowRight,
    Plus,
    Loader2,
    X,
    Trash2,
} from 'lucide-react';

// Color palette for classes
const CLASS_COLORS = [
    '#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6',
    '#06b6d4', '#3b82f6', '#6366f1', '#8b5cf6', '#d946ef',
    '#ec4899', '#f43f5e'
];

export default function AnnotateView() {
    const { projectPath, classes, addClass, removeClass } = useAppStore();

    // State
    const [images, setImages] = useState<string[]>([]);
    const [currentImageIdx, setCurrentImageIdx] = useState(0);
    const [scale, setScale] = useState(1);
    const [selectedTool, setSelectedTool] = useState<AnnotationTool>('box');
    const [loading, setLoading] = useState(false);

    // Annotation State
    const [boxes, setBoxes] = useState<BoundingBox[]>([]);
    const [polygons, setPolygons] = useState<Polygon[]>([]);
    const [currentClassId, setCurrentClassId] = useState<number>(0);
    const [saving, setSaving] = useState(false);

    // Class creation modal
    const [showAddClass, setShowAddClass] = useState(false);
    const [newClassName, setNewClassName] = useState('');

    // Initialize Default Class if none
    useEffect(() => {
        if (classes.length > 0 && !currentClassId) {
            setCurrentClassId(classes[0].id);
        }
    }, [classes, currentClassId]);

    // Load images
    useEffect(() => {
        async function init() {
            if (!projectPath) return;
            setLoading(true);
            try {
                const data = await getDatasetPreview(projectPath, 1, 1000);
                setImages(data.images);
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        }
        init();
    }, [projectPath]);

    const currentImage = images[currentImageIdx];

    // Load annotations when image changes
    useEffect(() => {
        async function loadAnns() {
            if (!projectPath || !currentImage) return;
            try {
                const imageId = currentImage.split(/[\\/]/).pop() || '';
                const data = await loadAnnotations(projectPath, imageId);
                if (data && data.annotations) {
                    const loadedBoxes: BoundingBox[] = data.annotations.map((ann: any) => ({
                        id: crypto.randomUUID(),
                        x: ann.x - (ann.w / 2),
                        y: ann.y - (ann.h / 2),
                        width: ann.w,
                        height: ann.h,
                        classId: ann.cls_idx,
                        className: classes.find(c => c.id === ann.cls_idx)?.name || 'Unknown'
                    }));
                    setBoxes(loadedBoxes);
                } else {
                    setBoxes([]);
                }
                setPolygons([]);
            } catch (e) {
                console.error(e);
                setBoxes([]);
                setPolygons([]);
            }
        }
        loadAnns();
    }, [projectPath, currentImage, classes]);

    const handleNext = () => {
        if (currentImageIdx < images.length - 1) setCurrentImageIdx(p => p + 1);
    };

    const handlePrev = () => {
        if (currentImageIdx > 0) setCurrentImageIdx(p => p - 1);
    };

    const handleBoxesChange = async (newBoxes: BoundingBox[]) => {
        setBoxes(newBoxes);
        await saveCurrentAnnotations(newBoxes, polygons);
    };

    const handlePolygonsChange = async (newPolygons: Polygon[]) => {
        setPolygons(newPolygons);
        await saveCurrentAnnotations(boxes, newPolygons);
    };

    const saveCurrentAnnotations = async (currentBoxes: BoundingBox[], currentPolygons: Polygon[]) => {
        if (!projectPath || !currentImage) return;

        const imageId = currentImage.split(/[\\/]/).pop() || '';

        const yoloAnns = currentBoxes.map(box => ({
            type: 'box',
            cls_idx: box.classId,
            x: box.x + (box.width / 2),
            y: box.y + (box.height / 2),
            w: box.width,
            h: box.height
        }));

        const polyAnns = currentPolygons.map(poly => ({
            type: 'polygon',
            cls_idx: poly.classId,
            points: poly.points,
            closed: poly.closed
        }));

        try {
            setSaving(true);
            await saveAnnotations(projectPath, imageId, [...yoloAnns, ...polyAnns]);
        } catch (e) {
            console.error("Failed to save:", e);
        } finally {
            setSaving(false);
        }
    };

    const handleAddClass = () => {
        if (!newClassName.trim()) return;

        const newId = classes.length > 0 ? Math.max(...classes.map(c => c.id)) + 1 : 0;
        const color = CLASS_COLORS[newId % CLASS_COLORS.length];

        addClass({
            id: newId,
            name: newClassName.trim(),
            color: color,
            count: 0
        });

        setNewClassName('');
        setShowAddClass(false);
        setCurrentClassId(newId);
    };

    const handleDeleteClass = (id: number) => {
        if (confirm('Delete this class? Annotations using this class will need to be updated.')) {
            removeClass(id);
            if (currentClassId === id && classes.length > 1) {
                const remaining = classes.filter(c => c.id !== id);
                if (remaining.length > 0) setCurrentClassId(remaining[0].id);
            }
        }
    };

    return (
        <div className="annotate-container">
            {/* Left Sidebar - Image List */}
            <div className="annotate-sidebar-left">
                <div className="p-4 border-b border-gray-800">
                    <h3 className="font-bold text-gray-400 text-xs uppercase tracking-wider">Images ({images.length})</h3>
                </div>
                <div className="flex-1 overflow-y-auto">
                    {images.map((img, idx) => (
                        <div
                            key={idx}
                            className={`image-list-item ${idx === currentImageIdx ? 'active' : ''}`}
                            onClick={() => setCurrentImageIdx(idx)}
                        >
                            <img src={`media://${encodeURIComponent(img.replace(/\\/g, '/'))}`} className="image-list-thumb" loading="lazy" alt="" />
                            <span className="text-sm truncate text-gray-300">
                                {img.split(/[\\/]/).pop()}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Main Area */}
            <div className="annotate-main">
                {/* Toolbar */}
                <div className="tools-bar">
                    <div className="flex bg-white/5 rounded-lg p-1 gap-1">
                        <button
                            className={`p-2 rounded ${selectedTool === 'move' ? 'bg-indigo-600' : 'hover:bg-white/10'}`}
                            onClick={() => setSelectedTool('move')}
                            title="Move/Pan (V)"
                        >
                            <MousePointer2 size={18} />
                        </button>
                        <button
                            className={`p-2 rounded ${selectedTool === 'box' ? 'bg-indigo-600' : 'hover:bg-white/10'}`}
                            onClick={() => setSelectedTool('box')}
                            title="Draw Box (B)"
                        >
                            <Square size={18} />
                        </button>
                        <button
                            className={`p-2 rounded ${selectedTool === 'polygon' ? 'bg-green-600' : 'hover:bg-white/10'}`}
                            onClick={() => setSelectedTool('polygon')}
                            title="Draw Polygon (P) - for segmentation"
                        >
                            <Hexagon size={18} />
                        </button>
                    </div>

                    <div className="h-6 w-px bg-white/10 mx-2"></div>

                    <div className="flex items-center gap-2">
                        <button className="p-2 hover:bg-white/10 rounded" onClick={() => setScale(s => Math.max(0.1, s - 0.1))}><ZoomOut size={18} /></button>
                        <span className="text-sm w-12 text-center">{Math.round(scale * 100)}%</span>
                        <button className="p-2 hover:bg-white/10 rounded" onClick={() => setScale(s => s + 0.1)}><ZoomIn size={18} /></button>
                    </div>

                    <div className="ml-4">
                        {saving && <span className="text-xs text-gray-400 flex items-center gap-1"><Loader2 size={12} className="animate-spin" /> Saving...</span>}
                    </div>

                    <div className="flex-1"></div>

                    <div className="flex items-center gap-2">
                        <button className="p-2 hover:bg-white/10 rounded" onClick={handlePrev} disabled={currentImageIdx === 0}>
                            <ArrowLeft size={18} />
                        </button>
                        <span className="text-sm text-gray-400">
                            {currentImageIdx + 1} / {images.length}
                        </span>
                        <button className="p-2 hover:bg-white/10 rounded" onClick={handleNext} disabled={currentImageIdx === images.length - 1}>
                            <ArrowRight size={18} />
                        </button>
                    </div>
                </div>

                {/* Canvas Area */}
                <div className="canvas-wrapper bg-[#121215]">
                    {loading ? (
                        <div className="text-gray-500 flex items-center gap-2">
                            <Loader2 className="animate-spin" /> Loading images...
                        </div>
                    ) : currentImage ? (
                        <Canvas
                            imageUrl={`media://${encodeURIComponent(currentImage.replace(/\\/g, '/'))}`}
                            boxes={boxes}
                            polygons={polygons}
                            classes={classes}
                            currentClassId={currentClassId}
                            scale={scale}
                            tool={selectedTool}
                            onBoxesChange={handleBoxesChange}
                            onPolygonsChange={handlePolygonsChange}
                        />
                    ) : (
                        <div className="text-gray-500">No images. Add images in the Dataset tab first.</div>
                    )}
                </div>
            </div>

            {/* Right Sidebar - Classes & Attributes */}
            <div className="annotate-sidebar-right">
                <div className="p-4 border-b border-gray-800 flex justify-between items-center">
                    <h3 className="font-bold text-gray-400 text-xs uppercase tracking-wider">Classes</h3>
                    <button
                        className="p-1.5 hover:bg-white/10 rounded bg-indigo-600/20 text-indigo-400"
                        onClick={() => setShowAddClass(true)}
                        title="Add new class"
                    >
                        <Plus size={16} />
                    </button>
                </div>
                <div className="class-list">
                    {classes.length === 0 ? (
                        <div className="p-4 text-sm text-gray-500 text-center">
                            <p className="mb-3">No classes defined.</p>
                            <button
                                onClick={() => setShowAddClass(true)}
                                className="px-3 py-1.5 bg-indigo-600 rounded text-white text-xs"
                            >
                                + Add First Class
                            </button>
                        </div>
                    ) : (
                        classes.map(cls => (
                            <div
                                key={cls.id}
                                className={`class-item ${currentClassId === cls.id ? 'active' : ''}`}
                                onClick={() => setCurrentClassId(cls.id)}
                            >
                                <div className="class-color" style={{ backgroundColor: cls.color }}></div>
                                <span className="text-sm flex-1">{cls.name}</span>
                                <span className="text-xs text-gray-500 mr-2">{cls.count || 0}</span>
                                <button
                                    onClick={(e) => { e.stopPropagation(); handleDeleteClass(cls.id); }}
                                    className="p-1 hover:bg-red-500/20 rounded text-gray-500 hover:text-red-400"
                                >
                                    <Trash2 size={12} />
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Add Class Modal */}
            {showAddClass && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                    <div className="bg-[#18181b] border border-gray-800 rounded-xl shadow-2xl p-6 w-[400px]">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-lg font-bold">Add New Class</h3>
                            <button onClick={() => setShowAddClass(false)} className="p-1 hover:bg-white/10 rounded">
                                <X size={18} />
                            </button>
                        </div>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Class Name</label>
                                <input
                                    type="text"
                                    value={newClassName}
                                    onChange={(e) => setNewClassName(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleAddClass()}
                                    placeholder="e.g., car, person, dog..."
                                    className="w-full bg-white/5 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-indigo-500 outline-none"
                                    autoFocus
                                />
                            </div>
                            <div className="flex gap-3">
                                <button
                                    onClick={() => setShowAddClass(false)}
                                    className="flex-1 py-2 rounded-lg border border-gray-700 hover:bg-white/5"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleAddClass}
                                    disabled={!newClassName.trim()}
                                    className="flex-1 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50"
                                >
                                    Add Class
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

