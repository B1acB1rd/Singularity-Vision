import { useState } from 'react';
import { useAppStore } from '../../store/appStore';
import { createProject, loadProject } from '../../services/api';
import {
    FolderPlus,
    FolderOpen,
    Clock,
    Sparkles,
    Box,
    ScanSearch,
    Layers,
    ChevronRight,
    Trash2,
    Settings,
    HardDrive,
    Cloud,
    Wifi,
} from 'lucide-react';
import type { ProjectConfig, IndustryProfile, SecurityMode, TaskType } from '../../types/project';
import IndustrySelector from './IndustrySelector';
import styles from './HomeScreen.module.css';

export default function HomeScreen() {
    const { recentProjects, addRecentProject, removeRecentProject, setCurrentProject } =
        useAppStore();
    const [showNewProjectModal, setShowNewProjectModal] = useState(false);
    const [loading, setLoading] = useState(false);

    const handleOpenProject = async () => {
        if (!window.electronAPI) return;

        const path = await window.electronAPI.openDirectory();
        if (!path) return;

        setLoading(true);
        try {
            const project = await loadProject(path);
            setCurrentProject(project, path);
            addRecentProject({
                id: project.id,
                name: project.name,
                path,
                lastOpened: new Date().toISOString(),
                taskType: project.taskType,
            });
        } catch (error) {
            console.error('Failed to load project:', error);
            // TODO: Show error toast
        } finally {
            setLoading(false);
        }
    };

    const handleOpenRecentProject = async (path: string) => {
        setLoading(true);
        try {
            const project = await loadProject(path);
            setCurrentProject(project, path);
            addRecentProject({
                id: project.id,
                name: project.name,
                path,
                lastOpened: new Date().toISOString(),
                taskType: project.taskType,
            });
        } catch (error) {
            console.error('Failed to load project:', error);
            // TODO: Show error toast
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={styles.homeScreen}>
            {/* Background Effects */}
            <div className={styles.backgroundEffects}>
                <div className={styles.gradientOrb1}></div>
                <div className={styles.gradientOrb2}></div>
                <div className={styles.gridPattern}></div>
            </div>

            {/* Header */}
            <header className={styles.header}>
                <div className={styles.logo}>
                    <Sparkles className={styles.logoIcon} />
                    <span>Singularity Vision</span>
                </div>
                <button className="btn btn-ghost btn-icon" data-tooltip="Settings">
                    <Settings size={20} />
                </button>
            </header>

            {/* Main Content */}
            <main className={styles.main}>
                {/* Hero Section */}
                <section className={styles.hero}>
                    <h1 className={styles.title}>
                        Build. Train. Deploy.{' '}
                        <span className={styles.titleAccent}>Locally.</span>
                    </h1>
                    <p className={styles.subtitle}>
                        The no-code computer vision platform that runs entirely on your machine.
                        Train models, annotate data, and deploy — all without writing a single line
                        of code.
                    </p>
                </section>

                {/* Action Cards */}
                <section className={styles.actions}>
                    <button
                        className={styles.actionCard}
                        onClick={() => setShowNewProjectModal(true)}
                        disabled={loading}
                    >
                        <div className={styles.actionIcon}>
                            <FolderPlus size={28} />
                        </div>
                        <div className={styles.actionContent}>
                            <h3>New Project</h3>
                            <p>Start a new computer vision project</p>
                        </div>
                        <ChevronRight className={styles.actionArrow} />
                    </button>

                    <button
                        className={styles.actionCard}
                        onClick={handleOpenProject}
                        disabled={loading}
                    >
                        <div className={styles.actionIcon}>
                            <FolderOpen size={28} />
                        </div>
                        <div className={styles.actionContent}>
                            <h3>Open Project</h3>
                            <p>Continue working on an existing project</p>
                        </div>
                        <ChevronRight className={styles.actionArrow} />
                    </button>
                </section>

                {/* Recent Projects */}
                {recentProjects.length > 0 && (
                    <section className={styles.recentSection}>
                        <div className={styles.sectionHeader}>
                            <Clock size={18} />
                            <h2>Recent Projects</h2>
                        </div>
                        <div className={styles.recentGrid}>
                            {recentProjects.map((project) => (
                                <div key={project.id} className={styles.recentCard}>
                                    <button
                                        className={styles.recentContent}
                                        onClick={() => handleOpenRecentProject(project.path)}
                                        disabled={loading}
                                    >
                                        <div className={styles.recentIcon}>
                                            {project.taskType === 'classification' && <Layers size={24} />}
                                            {project.taskType === 'detection' && <Box size={24} />}
                                            {project.taskType === 'segmentation' && <ScanSearch size={24} />}
                                        </div>
                                        <div className={styles.recentInfo}>
                                            <h4>{project.name}</h4>
                                            <p>{project.taskType}</p>
                                            <span className={styles.recentDate}>
                                                {formatDate(project.lastOpened)}
                                            </span>
                                        </div>
                                    </button>
                                    <button
                                        className={styles.recentDelete}
                                        onClick={() => removeRecentProject(project.id)}
                                        data-tooltip="Remove from recent"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* Task Type Cards - Only show when no recent projects */}
                {recentProjects.length === 0 && (
                    <section className={styles.taskTypes}>
                        <h2>What will you build today?</h2>
                        <div className={styles.taskGrid}>
                            <div className={styles.taskCard}>
                                <Layers size={32} />
                                <h3>Image Classification</h3>
                                <p>Categorize images into predefined classes</p>
                            </div>
                            <div className={styles.taskCard}>
                                <Box size={32} />
                                <h3>Object Detection</h3>
                                <p>Locate and identify objects in images</p>
                            </div>
                            <div className={styles.taskCard}>
                                <ScanSearch size={32} />
                                <h3>Segmentation</h3>
                                <p>Pixel-level object recognition</p>
                                <span className={styles.comingSoon}>Coming Soon</span>
                            </div>
                        </div>
                    </section>
                )}
            </main>

            {/* Footer */}
            <footer className={styles.footer}>
                <p>
                    Singularity Vision v1.0.0 • Built with ❤️ for the CV community
                </p>
            </footer>

            {/* New Project Modal */}
            {showNewProjectModal && (
                <NewProjectModal
                    onClose={() => setShowNewProjectModal(false)}
                    onCreated={(project, path) => {
                        setCurrentProject(project, path);
                        addRecentProject({
                            id: project.id,
                            name: project.name,
                            path,
                            lastOpened: new Date().toISOString(),
                            taskType: project.taskType,
                        });
                        setShowNewProjectModal(false);
                    }}
                />
            )}
        </div>
    );
}

interface NewProjectModalProps {
    onClose: () => void;
    onCreated: (project: ProjectConfig, path: string) => void;
}

function NewProjectModal({ onClose, onCreated }: NewProjectModalProps) {
    const [name, setName] = useState('');
    const [path, setPath] = useState('');
    const [taskType, setTaskType] = useState<TaskType>('detection');
    const [framework, setFramework] = useState<ProjectConfig['framework']>('pytorch');
    const [industryProfile, setIndustryProfile] = useState<IndustryProfile>('general');
    const [securityMode, setSecurityMode] = useState<SecurityMode>('local');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSelectPath = async () => {
        if (!window.electronAPI) return;
        const selected = await window.electronAPI.openDirectory();
        if (selected) {
            setPath(selected);
        }
    };

    const handleCreate = async () => {
        if (!name.trim()) {
            setError('Please enter a project name');
            return;
        }
        if (!path) {
            setError('Please select a project location');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const fullPath = `${path}/${name.replace(/[^a-zA-Z0-9-_]/g, '_')}`;
            const project = await createProject(name, fullPath, taskType, framework, industryProfile, securityMode);
            onCreated(project, fullPath);
        } catch (err) {
            setError('Failed to create project. Please try again.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h2 className="modal-title">Create New Project</h2>
                    <button className="modal-close" onClick={onClose}>
                        ✕
                    </button>
                </div>

                <div className={styles.formGroup}>
                    <label>Project Name</label>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="My CV Project"
                        autoFocus
                    />
                </div>

                <div className={styles.formGroup}>
                    <label>Location</label>
                    <div className={styles.pathInput}>
                        <input
                            type="text"
                            value={path}
                            onChange={(e) => setPath(e.target.value)}
                            placeholder="Select folder..."
                            readOnly
                        />
                        <button className="btn btn-secondary" onClick={handleSelectPath}>
                            Browse
                        </button>
                    </div>
                </div>

                <div className={styles.formGroup}>
                    <label>Industry Profile</label>
                    <IndustrySelector value={industryProfile} onChange={setIndustryProfile} />
                </div>

                <div className={styles.formGroup}>
                    <label>Task Type</label>
                    <div className={styles.optionGrid}>
                        <button
                            type="button"
                            className={`${styles.optionCard} ${taskType === 'classification' ? styles.selected : ''}`}
                            onClick={() => setTaskType('classification')}
                        >
                            <Layers size={24} />
                            <span>Classification</span>
                        </button>
                        <button
                            type="button"
                            className={`${styles.optionCard} ${taskType === 'detection' ? styles.selected : ''}`}
                            onClick={() => setTaskType('detection')}
                        >
                            <Box size={24} />
                            <span>Detection</span>
                        </button>
                    </div>
                </div>

                <div className={styles.formGroup}>
                    <label>Security Mode</label>
                    <div className={styles.securityModeGrid}>
                        <button
                            type="button"
                            className={`${styles.securityCard} ${securityMode === 'local' ? styles.securityCardSelected : ''}`}
                            onClick={() => setSecurityMode('local')}
                        >
                            <HardDrive size={24} />
                            <span>Local Only</span>
                        </button>
                        <button
                            type="button"
                            className={`${styles.securityCard} ${securityMode === 'hybrid' ? styles.securityCardSelected : ''}`}
                            onClick={() => setSecurityMode('hybrid')}
                        >
                            <Wifi size={24} />
                            <span>Hybrid</span>
                        </button>
                        <button
                            type="button"
                            className={`${styles.securityCard} ${securityMode === 'cloud' ? styles.securityCardSelected : ''}`}
                            onClick={() => setSecurityMode('cloud')}
                        >
                            <Cloud size={24} />
                            <span>Cloud</span>
                        </button>
                    </div>
                </div>

                <div className={styles.formGroup}>
                    <label>Framework</label>
                    <div className={styles.optionGrid}>
                        <button
                            type="button"
                            className={`${styles.optionCard} ${framework === 'pytorch' ? styles.selected : ''}`}
                            onClick={() => setFramework('pytorch')}
                        >
                            🔥
                            <span>PyTorch</span>
                        </button>
                        <button
                            type="button"
                            className={`${styles.optionCard} ${framework === 'tensorflow' ? styles.selected : ''}`}
                            onClick={() => setFramework('tensorflow')}
                        >
                            🧠
                            <span>TensorFlow</span>
                        </button>
                    </div>
                </div>

                {error && <p className={styles.error}>{error}</p>}

                <div className={styles.modalActions}>
                    <button className="btn btn-secondary" onClick={onClose}>
                        Cancel
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={handleCreate}
                        disabled={loading}
                    >
                        {loading ? 'Creating...' : 'Create Project'}
                    </button>
                </div>
            </div>
        </div>
    );
}

function formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
}
