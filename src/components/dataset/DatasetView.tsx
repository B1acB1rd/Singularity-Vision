import './dataset.css';
import { useState, useEffect, useCallback } from 'react';
import { useAppStore } from '../../store/appStore';
import { getDatasetPreview, importImages, splitDataset, deleteImage } from '../../services/api';
import {
    Upload,
    Video,
    Image as ImageIcon,
    MoreVertical,
    Plus,
    Loader2,
    ChevronLeft,
    ChevronRight,
    Scissors,
    AlertCircle,
    CheckCircle2,
    Trash2
} from 'lucide-react';

export default function DatasetView() {
    const { projectPath } = useAppStore();
    const [images, setImages] = useState<string[]>([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(1);
    const [loading, setLoading] = useState(false);
    const [importing, setImporting] = useState(false);
    const [pageSize] = useState(50);
    const [showSplitModal, setShowSplitModal] = useState(false);

    const loadImages = useCallback(async () => {
        if (!projectPath) return;
        try {
            setLoading(true);
            const data = await getDatasetPreview(projectPath, page, pageSize);
            setImages(data.images);
            setTotal(data.total);
        } catch (error) {
            console.error('Failed to load dataset:', error);
        } finally {
            setLoading(false);
        }
    }, [projectPath, page, pageSize]);

    useEffect(() => {
        loadImages();
    }, [loadImages]);

    const handleImportImages = async () => {
        if (!window.electronAPI || !projectPath) return;

        try {
            const filePaths = await window.electronAPI.openFiles({
                filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'] }],
            });

            if (filePaths.length > 0) {
                setImporting(true);
                await importImages(projectPath, filePaths);
                await loadImages(); // Refresh
            }
        } catch (error) {
            console.error('Import failed:', error);
        } finally {
            setImporting(false);
        }
    };

    return (
        <div className="dataset-view relative">
            {/* Toolbar */}
            <div className="dataset-toolbar">
                <div className="toolbar-title">
                    <h2>
                        <DatabaseIcon /> Dataset
                        <span className="image-count">
                            {total} images
                        </span>
                    </h2>
                    <p className="text-gray-400 text-sm mt-1">
                        Manage your training data. Supported formats: JPG, PNG, BMP, WEBP.
                    </p>
                </div>
                <div className="toolbar-actions">
                    <button
                        onClick={handleImportImages}
                        disabled={importing}
                        className="btn-primary"
                    >
                        {importing ? <Loader2 className="animate-spin" size={20} /> : <Upload size={20} />}
                        Import Images
                    </button>
                    <button
                        onClick={() => setShowSplitModal(true)}
                        disabled={total === 0}
                        className="btn-secondary"
                    >
                        <Scissors size={20} />
                        Split Dataset
                    </button>
                    <button
                        disabled
                        className="btn-secondary opacity-50 cursor-not-allowed"
                        title="Video import coming soon"
                    >
                        <Video size={20} />
                        Import Video
                    </button>
                </div>
            </div>

            {/* Content */}
            <div className="dataset-grid-container">
                {loading && images.length === 0 ? (
                    <div className="flex items-center justify-center h-full">
                        <Loader2 className="animate-spin text-indigo-500" size={48} />
                    </div>
                ) : images.length === 0 ? (
                    <EmptyState onImport={handleImportImages} />
                ) : (
                    <div className="image-grid">
                        {images.map((img, idx) => (
                            <ImageCard
                                key={idx}
                                path={img}
                                onDelete={async (path) => {
                                    try {
                                        await deleteImage(projectPath || '', path);
                                        await loadImages();
                                    } catch (e) {
                                        console.error('Delete failed:', e);
                                    }
                                }}
                            />
                        ))}
                    </div>
                )}
            </div>

            {/* Pagination */}
            {total > pageSize && (
                <div className="pagination">
                    <span className="text-sm text-gray-400">
                        Showing {((page - 1) * pageSize) + 1} - {Math.min(page * pageSize, total)} of {total}
                    </span>
                    <div className="page-controls">
                        <button
                            onClick={() => setPage((p) => Math.max(1, p - 1))}
                            disabled={page === 1}
                            className="btn-icon"
                        >
                            <ChevronLeft size={20} />
                        </button>
                        <span className="flex items-center px-4 bg-white/5 rounded-lg text-sm font-medium">
                            Page {page}
                        </span>
                        <button
                            onClick={() => setPage((p) => p + 1)}
                            disabled={page * pageSize >= total}
                            className="btn-icon"
                        >
                            <ChevronRight size={20} />
                        </button>
                    </div>
                </div>
            )}

            {/* Split Modal */}
            {showSplitModal && (
                <SplitModal
                    totalImages={total}
                    onClose={() => setShowSplitModal(false)}
                    projectPath={projectPath || ''}
                />
            )}
        </div>
    );
}

function ImageCard({ path, onDelete }: { path: string; onDelete: (path: string) => void }) {
    const [showMenu, setShowMenu] = useState(false);
    const normalizedPath = path.replace(/\\/g, '/');
    const src = `media://${encodeURIComponent(normalizedPath)}`;

    const handleDelete = async () => {
        if (confirm('Delete this image? This action cannot be undone.')) {
            onDelete(path);
        }
        setShowMenu(false);
    };

    return (
        <div className="image-card group" onMouseLeave={() => setShowMenu(false)}>
            <img
                src={src}
                alt=""
                loading="lazy"
            />
            <div className="image-overlay">
                <div className="relative">
                    <button
                        className="p-2 bg-white/20 rounded-full hover:bg-white/30 backdrop-blur-sm text-white"
                        onClick={() => setShowMenu(!showMenu)}
                    >
                        <MoreVertical size={20} />
                    </button>
                    {showMenu && (
                        <div className="absolute right-0 top-full mt-1 bg-gray-900 border border-gray-700 rounded-lg shadow-xl overflow-hidden z-10">
                            <button
                                onClick={handleDelete}
                                className="flex items-center gap-2 px-4 py-2 text-sm text-red-400 hover:bg-red-500/20 w-full"
                            >
                                <Trash2 size={14} /> Delete
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function EmptyState({ onImport }: { onImport: () => void }) {
    return (
        <div className="empty-state">
            <div className="empty-icon-circle">
                <ImageIcon size={48} className="text-gray-500" />
            </div>
            <h3 className="text-xl font-bold mb-2">No Images Yet</h3>
            <p className="text-gray-400 max-w-sm mb-8">
                Import images to get started. You can upload individual files or select an entire folder.
            </p>
            <button
                onClick={onImport}
                className="btn-primary"
            >
                <Plus size={20} />
                Import Images
            </button>
        </div>
    );
}

function DatabaseIcon() {
    return (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
            <path d="M3 5v14c0 1.66 4 3 9 3s 9-1.34 9-3V5"></path>
        </svg>
    );
}

// Split Modal Component
function SplitModal({ totalImages, onClose, projectPath }: { totalImages: number, onClose: () => void, projectPath: string }) {
    const [train, setTrain] = useState(70);
    const [val, setVal] = useState(20);
    const [test, setTest] = useState(10);
    const [processing, setProcessing] = useState(false);
    const [result, setResult] = useState<{ train: number, val: number, test: number } | null>(null);

    // Auto-adjust logic could be added here, but for simplicity relying on sliders
    const handleSliderChange = (type: 'train' | 'val' | 'test', value: number) => {
        // Enforce sum = 100 logic roughly or just let user ensure it? 
        // Simple approach: Adjust others to fit? NO, too complex for now.
        // Let's just update the specific one and show warning if sum != 100

        if (type === 'train') setTrain(value);
        if (type === 'val') setVal(value);
        if (type === 'test') setTest(value);
    };

    const total = train + val + test;
    const isValid = total === 100;

    const handleSplit = async () => {
        if (!isValid) return;
        setProcessing(true);
        try {
            const res = await splitDataset(projectPath, train / 100, val / 100, test / 100);
            setResult(res);
        } catch (e) {
            console.error(e);
        } finally {
            setProcessing(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="bg-[#18181b] border border-gray-800 rounded-xl shadow-2xl p-6 w-[500px]">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <Scissors size={20} /> Split Dataset
                </h3>

                {!result ? (
                    <>
                        <p className="text-gray-400 text-sm mb-6">
                            Randomly split your {totalImages} images into Train, Validation, and Test sets.
                        </p>

                        <div className="space-y-6 mb-8">
                            {/* Sliders */}
                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-green-400 font-medium">Train</span>
                                    <span>{train}%</span>
                                </div>
                                <input
                                    type="range" min="0" max="100" value={train}
                                    onChange={(e) => handleSliderChange('train', parseInt(e.target.value))}
                                    className="w-full accent-green-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                                />
                            </div>

                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-blue-400 font-medium">Validation</span>
                                    <span>{val}%</span>
                                </div>
                                <input
                                    type="range" min="0" max="100" value={val}
                                    onChange={(e) => handleSliderChange('val', parseInt(e.target.value))}
                                    className="w-full accent-blue-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                                />
                            </div>

                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-yellow-400 font-medium">Test</span>
                                    <span>{test}%</span>
                                </div>
                                <input
                                    type="range" min="0" max="100" value={test}
                                    onChange={(e) => handleSliderChange('test', parseInt(e.target.value))}
                                    className="w-full accent-yellow-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                                />
                            </div>
                        </div>

                        {/* Summary */}
                        <div className={`p-4 rounded-lg bg-white/5 border ${isValid ? 'border-gray-700' : 'border-red-500/50'}`}>
                            <div className="flex justify-between mb-2">
                                <span className="text-sm text-gray-400">Total Images</span>
                                <span className="font-bold">{totalImages}</span>
                            </div>
                            <div className="h-2 w-full flex rounded-full overflow-hidden">
                                <div style={{ width: `${train}%` }} className="bg-green-500" />
                                <div style={{ width: `${val}%` }} className="bg-blue-500" />
                                <div style={{ width: `${test}%` }} className="bg-yellow-500" />
                            </div>
                            <div className="mt-2 text-xs text-center text-gray-400 flex justify-between">
                                <span>~{Math.round(totalImages * (train / 100))} Train</span>
                                <span>~{Math.round(totalImages * (val / 100))} Val</span>
                                <span>~{Math.round(totalImages * (test / 100))} Test</span>
                            </div>

                            {!isValid && (
                                <div className="mt-4 flex items-center gap-2 text-red-400 text-sm">
                                    <AlertCircle size={16} /> Total must be 100% (Current: {total}%)
                                </div>
                            )}
                        </div>

                        <div className="flex justify-end gap-3 mt-8">
                            <button onClick={onClose} className="px-4 py-2 rounded text-sm hover:bg-white/10">Cancel</button>
                            <button
                                onClick={handleSplit}
                                disabled={!isValid || processing}
                                className="px-4 py-2 rounded text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                            >
                                {processing ? <Loader2 className="animate-spin" size={16} /> : <Scissors size={16} />}
                                Split Now
                            </button>
                        </div>
                    </>
                ) : (
                    <div className="text-center py-8">
                        <div className="inline-flex p-4 rounded-full bg-green-500/20 text-green-500 mb-4">
                            <CheckCircle2 size={40} />
                        </div>
                        <h4 className="text-xl font-bold mb-2">Split Complete!</h4>
                        <p className="text-gray-400 mb-8">
                            Your dataset has been partitioned successfully.
                        </p>

                        <div className="grid grid-cols-3 gap-4 mb-8">
                            <div className="p-4 bg-white/5 rounded-lg border-t-2 border-green-500">
                                <div className="text-2xl font-bold">{result.train}</div>
                                <div className="text-xs text-gray-400 uppercase">Train</div>
                            </div>
                            <div className="p-4 bg-white/5 rounded-lg border-t-2 border-blue-500">
                                <div className="text-2xl font-bold">{result.val}</div>
                                <div className="text-xs text-gray-400 uppercase">Val</div>
                            </div>
                            <div className="p-4 bg-white/5 rounded-lg border-t-2 border-yellow-500">
                                <div className="text-2xl font-bold">{result.test}</div>
                                <div className="text-xs text-gray-400 uppercase">Test</div>
                            </div>
                        </div>

                        <button
                            onClick={onClose}
                            className="w-full py-2 rounded bg-white/10 hover:bg-white/20"
                        >
                            Close
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
