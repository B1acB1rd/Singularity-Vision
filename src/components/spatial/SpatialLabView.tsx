import { useState } from 'react';
import {
    MapPin,
    Globe,
    Layers,
    Grid3X3,
    FileJson,
    Upload,
    RefreshCw,
    CheckCircle,
    Loader2,
} from 'lucide-react';
import './spatial.css';

interface GeoImage {
    path: string;
    coordinate: {
        latitude: number;
        longitude: number;
        altitude?: number;
        timestamp?: string;
    } | null;
}

interface SpatialStats {
    totalImages: number;
    geoTagged: number;
    coverageArea?: string;
}

export default function SpatialLabView() {
    const [activeTab, setActiveTab] = useState<'data' | 'mapping' | 'analysis' | 'export'>('data');
    const [stats, setStats] = useState<SpatialStats>({ totalImages: 0, geoTagged: 0 });
    const [geoImages, setGeoImages] = useState<GeoImage[]>([]);
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState<'idle' | 'indexing' | 'success' | 'error'>('idle');

    const tabs = [
        { id: 'data', icon: Layers, label: 'Spatial Data' },
        { id: 'mapping', icon: Globe, label: 'Vision on Maps' },
        { id: 'analysis', icon: Grid3X3, label: 'Analysis' },
        { id: 'export', icon: FileJson, label: 'Export' },
    ];

    const handleIndexDataset = async () => {
        setLoading(true);
        setStatus('indexing');

        // This would call the backend API
        setTimeout(() => {
            setStats({ totalImages: 150, geoTagged: 42 });
            setGeoImages([
                { path: 'image_001.jpg', coordinate: { latitude: 40.7128, longitude: -74.0060 } },
                { path: 'image_002.jpg', coordinate: { latitude: 40.7589, longitude: -73.9851 } },
                { path: 'image_003.jpg', coordinate: null },
            ]);
            setStatus('success');
            setLoading(false);
        }, 1500);
    };

    const renderContent = () => {
        switch (activeTab) {
            case 'data':
                return (
                    <div className="spatial-content">
                        <div className="spatial-intro">
                            <Globe size={48} className="intro-icon" />
                            <h3>Spatial Data Management</h3>
                            <p>Import, index, and manage geo-referenced imagery including satellite, drone, and GPS-tagged photos.</p>
                        </div>

                        <div className="spatial-stats">
                            <div className="stat-card">
                                <Layers size={24} />
                                <div className="stat-info">
                                    <span className="stat-value">{stats.totalImages}</span>
                                    <span className="stat-label">Total Images</span>
                                </div>
                            </div>
                            <div className="stat-card">
                                <MapPin size={24} />
                                <div className="stat-info">
                                    <span className="stat-value">{stats.geoTagged}</span>
                                    <span className="stat-label">Geo-tagged</span>
                                </div>
                            </div>
                            <div className="stat-card">
                                <CheckCircle size={24} />
                                <div className="stat-info">
                                    <span className="stat-value">
                                        {stats.totalImages > 0
                                            ? Math.round((stats.geoTagged / stats.totalImages) * 100)
                                            : 0}%
                                    </span>
                                    <span className="stat-label">Coverage</span>
                                </div>
                            </div>
                        </div>

                        <div className="spatial-actions">
                            <button className="btn btn-primary" onClick={handleIndexDataset} disabled={loading}>
                                {loading ? (
                                    <>
                                        <Loader2 size={18} className="spin" />
                                        Indexing...
                                    </>
                                ) : (
                                    <>
                                        <RefreshCw size={18} />
                                        Index Dataset
                                    </>
                                )}
                            </button>
                            <button className="btn btn-secondary">
                                <Upload size={18} />
                                Import GeoTIFF
                            </button>
                        </div>

                        {status === 'success' && geoImages.length > 0 && (
                            <div className="geo-images-list">
                                <h4>Geo-tagged Images</h4>
                                <div className="geo-images-grid">
                                    {geoImages.filter(img => img.coordinate).map((img, i) => (
                                        <div key={i} className="geo-image-card">
                                            <MapPin size={16} />
                                            <span className="geo-image-name">{img.path}</span>
                                            <span className="geo-image-coords">
                                                {img.coordinate?.latitude.toFixed(4)}, {img.coordinate?.longitude.toFixed(4)}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                );

            case 'mapping':
                return (
                    <div className="spatial-content">
                        <div className="spatial-placeholder">
                            <Globe size={64} className="placeholder-icon" />
                            <h3>Vision on Maps</h3>
                            <p>Run computer vision models on map imagery and overlay predictions geographically.</p>
                            <span className="coming-badge">Coming in Phase 2</span>
                        </div>
                    </div>
                );

            case 'analysis':
                return (
                    <div className="spatial-content">
                        <div className="spatial-placeholder">
                            <Grid3X3 size={64} className="placeholder-icon" />
                            <h3>Spatial Analysis</h3>
                            <p>Change detection, temporal comparisons, and area statistics for geo-referenced data.</p>
                            <span className="coming-badge">Coming in Phase 2</span>
                        </div>
                    </div>
                );

            case 'export':
                return (
                    <div className="spatial-content">
                        <div className="spatial-placeholder">
                            <FileJson size={64} className="placeholder-icon" />
                            <h3>Spatial Export</h3>
                            <p>Export predictions as GeoJSON, GeoTIFF, or integrate with GIS tools.</p>
                            <span className="coming-badge">Coming in Phase 2</span>
                        </div>
                    </div>
                );

            default:
                return null;
        }
    };

    return (
        <div className="spatial-view">
            <div className="spatial-header">
                <div className="header-title">
                    <Globe size={24} />
                    <h2>Spatial Vision Lab</h2>
                </div>
                <span className="header-badge">Flagship Module</span>
            </div>

            <div className="spatial-tabs">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        className={`spatial-tab ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id as any)}
                    >
                        <tab.icon size={18} />
                        {tab.label}
                    </button>
                ))}
            </div>

            {renderContent()}
        </div>
    );
}
