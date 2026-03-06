import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ProjectConfig, TrainingStatus, ClassInfo, TaskType } from '../types/project';

interface RecentProject {
    id: string;
    name: string;
    path: string;
    lastOpened: string;
    taskType: TaskType;
}

interface AppState {
    // Current project
    currentProject: ProjectConfig | null;
    projectPath: string | null;

    // Recent projects
    recentProjects: RecentProject[];

    // UI state
    activeTab: 'dataset' | 'annotate' | 'augmentation' | 'opencv-lab' | 'models' | 'training' | 'testing' | 'evaluation' | 'spatial' | '3d-lab' | 'export' | 'settings';
    sidebarCollapsed: boolean;
    darkMode: boolean;

    // Training state
    trainingStatus: TrainingStatus;

    // Dataset state
    selectedImages: string[];
    currentImageIndex: number;
    classes: ClassInfo[];

    // Actions
    setCurrentProject: (project: ProjectConfig | null, path: string | null) => void;
    updateProject: (updates: Partial<ProjectConfig>) => void;
    addRecentProject: (project: RecentProject) => void;
    removeRecentProject: (id: string) => void;
    setActiveTab: (tab: AppState['activeTab']) => void;
    toggleSidebar: () => void;
    toggleDarkMode: () => void;
    setTrainingStatus: (status: TrainingStatus) => void;
    setSelectedImages: (images: string[]) => void;
    setCurrentImageIndex: (index: number) => void;
    setClasses: (classes: ClassInfo[]) => void;
    addClass: (classInfo: ClassInfo) => void;
    updateClass: (id: number, updates: Partial<ClassInfo>) => void;
    removeClass: (id: number) => void;
    reset: () => void;
}

const initialTrainingStatus: TrainingStatus = {
    status: 'idle',
    current_epoch: 0,
    total_epochs: 0,
    progress: 0,
    logs: [],
    metrics: undefined,
};

export const useAppStore = create<AppState>()(
    persist(
        (set) => ({
            // Initial state
            currentProject: null,
            projectPath: null,
            recentProjects: [],
            activeTab: 'dataset',
            sidebarCollapsed: false,
            darkMode: true,
            trainingStatus: initialTrainingStatus,
            selectedImages: [],
            currentImageIndex: 0,
            classes: [],

            // Actions
            setCurrentProject: (project, path) =>
                set({ currentProject: project, projectPath: path }),

            updateProject: (updates) =>
                set((state) => ({
                    currentProject: state.currentProject
                        ? { ...state.currentProject, ...updates, updatedAt: new Date().toISOString() }
                        : null,
                })),

            addRecentProject: (project) =>
                set((state) => {
                    const filtered = state.recentProjects.filter((p) => p.id !== project.id);
                    return {
                        recentProjects: [project, ...filtered].slice(0, 10),
                    };
                }),

            removeRecentProject: (id) =>
                set((state) => ({
                    recentProjects: state.recentProjects.filter((p) => p.id !== id),
                })),

            setActiveTab: (tab) => set({ activeTab: tab }),

            toggleSidebar: () =>
                set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

            toggleDarkMode: () =>
                set((state) => ({ darkMode: !state.darkMode })),

            setTrainingStatus: (status) => set({ trainingStatus: status }),

            setSelectedImages: (images) => set({ selectedImages: images }),

            setCurrentImageIndex: (index) => set({ currentImageIndex: index }),

            setClasses: (classes) => set({ classes }),

            addClass: (classInfo) =>
                set((state) => ({ classes: [...state.classes, classInfo] })),

            updateClass: (id, updates) =>
                set((state) => ({
                    classes: state.classes.map((c) =>
                        c.id === id ? { ...c, ...updates } : c
                    ),
                })),

            removeClass: (id) =>
                set((state) => ({
                    classes: state.classes.filter((c) => c.id !== id),
                })),

            reset: () =>
                set({
                    currentProject: null,
                    projectPath: null,
                    activeTab: 'dataset',
                    trainingStatus: initialTrainingStatus,
                    selectedImages: [],
                    currentImageIndex: 0,
                    classes: [],
                }),
        }),
        {
            name: 'singularity-vision-storage',
            partialize: (state) => ({
                recentProjects: state.recentProjects,
                darkMode: state.darkMode,
                sidebarCollapsed: state.sidebarCollapsed,
            }),
        }
    )
);
