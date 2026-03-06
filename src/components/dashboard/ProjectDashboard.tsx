import './dashboard.css';
import { useAppStore } from '../../store/appStore';
import DatasetView from '../dataset/DatasetView';
import AnnotateView from '../annotate/AnnotateView';
import TrainingView from '../training/TrainingView';
import TestingView from '../testing/TestingView';
import ExportView from '../export/ExportView';
import ModelHubView from '../modelhub/ModelHubView';
import SettingsView from '../settings/SettingsView';
import AugmentationView from '../augmentation/AugmentationView';
import EvaluationView from '../evaluation/EvaluationView';
import SpatialLabView from '../spatial/SpatialLabView';
import OpenCVLabView from '../opencv-lab/OpenCVLabView';
import ThreeDView from '../3d/ThreeDView';
import {
    LayoutDashboard,
    Database,
    Pencil,
    Brain,
    PlayCircle,
    Share,
    Settings,
    ChevronLeft,
    Menu,
    Wand2,
    BarChart3,
    Globe,
    Sparkles,
    FolderOpen,
    Home,
    Box,
} from 'lucide-react';

interface MenuItem {
    id: string;
    icon: any;
    label: string;
    accent?: boolean;
}

interface MenuGroup {
    id: string;
    label: string;
    items: MenuItem[];
}

// Grouped navigation structure
const MENU_GROUPS: MenuGroup[] = [
    {
        id: 'data',
        label: 'Data',
        items: [
            { id: 'dataset', icon: Database, label: 'Dataset' },
            { id: 'annotate', icon: Pencil, label: 'Annotate' },
            { id: 'augmentation', icon: Wand2, label: 'Augmentation' },
        ]
    },
    {
        id: 'model',
        label: 'Model',
        items: [
            { id: 'models', icon: Brain, label: 'Model Hub' },
            { id: 'training', icon: LayoutDashboard, label: 'Training' },
            { id: 'testing', icon: PlayCircle, label: 'Testing' },
            { id: 'evaluation', icon: BarChart3, label: 'Evaluation' },
        ]
    },
    {
        id: 'tools',
        label: 'Tools',
        items: [
            { id: 'opencv-lab', icon: Sparkles, label: 'OpenCV Lab', accent: true },
            { id: 'spatial', icon: Globe, label: 'Spatial Lab', accent: true },
            { id: '3d-lab', icon: Box, label: '3D Lab', accent: true },
        ]
    },
    {
        id: 'output',
        label: 'Output',
        items: [
            { id: 'export', icon: Share, label: 'Export' },
        ]
    }
];

export default function ProjectDashboard() {
    const { currentProject, activeTab, setActiveTab, sidebarCollapsed, toggleSidebar, setCurrentProject } =
        useAppStore();

    const handleCloseProject = () => {
        setCurrentProject(null, null);
    };

    const renderContent = () => {
        switch (activeTab) {
            case 'dataset':
                return <DatasetView />;
            case 'annotate':
                return <AnnotateView />;
            case 'training':
                return <TrainingView />;
            case 'testing':
                return <TestingView />;
            case 'export':
                return <ExportView />;
            case 'models':
                return <ModelHubView />;
            case 'augmentation':
                return <AugmentationView />;
            case 'evaluation':
                return <EvaluationView />;
            case 'spatial':
                return <SpatialLabView />;
            case 'opencv-lab':
                return <OpenCVLabView />;
            case '3d-lab':
                return <ThreeDView />;
            case 'settings':
                return <SettingsView />;
            default:
                return (
                    <div className="flex-1 p-8 overflow-auto">
                        <h1 className="text-3xl font-bold mb-6 capitalize">{activeTab}</h1>
                        <div className="p-12 border-2 border-dashed border-white/10 rounded-xl flex flex-col items-center justify-center text-gray-500 min-h-[400px]">
                            <p className="mb-4">Component for {activeTab} is under construction</p>
                            <div className="animate-pulse w-16 h-16 bg-white/5 rounded-full"></div>
                        </div>
                    </div>
                );
        }
    };

    return (
        <div className="dashboard-container">
            {/* Sidebar */}
            <aside className={`dashboard-sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
                {/* Sidebar Header */}
                <div className="sidebar-header">
                    {!sidebarCollapsed && (
                        <div className="flex items-center gap-2 min-w-0">
                            <FolderOpen size={18} className="text-indigo-400 flex-shrink-0" />
                            <span className="font-bold truncate">{currentProject?.name}</span>
                        </div>
                    )}
                    <button
                        onClick={toggleSidebar}
                        className="p-1.5 rounded-md hover:bg-white/5 text-gray-400 hover:text-white flex-shrink-0"
                    >
                        {sidebarCollapsed ? <Menu size={20} /> : <ChevronLeft size={20} />}
                    </button>
                </div>

                {/* Grouped Navigation */}
                <nav className="sidebar-nav">
                    {MENU_GROUPS.map((group) => (
                        <div key={group.id} className="nav-group">
                            {!sidebarCollapsed && (
                                <div className="nav-group-label">{group.label}</div>
                            )}
                            {group.items.map((item) => (
                                <button
                                    key={item.id}
                                    onClick={() => setActiveTab(item.id as any)}
                                    className={`nav-item ${activeTab === item.id ? 'active' : ''} ${item.accent ? 'accent' : ''}`}
                                >
                                    <item.icon size={20} />
                                    {!sidebarCollapsed && (
                                        <span className="nav-label">{item.label}</span>
                                    )}
                                </button>
                            ))}
                        </div>
                    ))}
                </nav>

                {/* Sidebar Footer */}
                <div className="sidebar-footer">
                    <button
                        onClick={() => setActiveTab('settings')}
                        className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
                    >
                        <Settings size={20} />
                        {!sidebarCollapsed && (
                            <span className="nav-label">Settings</span>
                        )}
                    </button>
                    <button
                        onClick={handleCloseProject}
                        className="nav-item home-btn"
                    >
                        <Home size={20} />
                        {!sidebarCollapsed && (
                            <span className="nav-label">Close Project</span>
                        )}
                    </button>
                </div>
            </aside>

            {/* Main Content Area */}
            <main className="dashboard-main">
                {renderContent()}
            </main>
        </div>
    );
}
