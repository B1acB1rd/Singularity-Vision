import axios from 'axios';
import type {
    ProjectConfig,
    IndustryProfile,
    SecurityMode,
    TaskType,
} from '../types/project';

// API client - will connect to Python backend
const getApiClient = async () => {
    let port = 8765;
    if (window.electronAPI) {
        port = await window.electronAPI.getBackendPort();
    }
    return axios.create({
        baseURL: `http://127.0.0.1:${port}`,
        timeout: 30000,
        headers: {
            'Content-Type': 'application/json',
        },
    });
};

// Health check
export const checkBackendHealth = async (): Promise<boolean> => {
    try {
        const api = await getApiClient();
        const response = await api.get('/health');
        return response.data.status === 'ok';
    } catch {
        return false;
    }
};

// System info
export const getSystemInfo = async () => {
    const api = await getApiClient();
    const response = await api.get('/system/info');
    return response.data;
};

// Project operations
export const createProject = async (
    name: string,
    path: string,
    taskType: TaskType,
    framework: ProjectConfig['framework'],
    industryProfile: IndustryProfile = 'general',
    securityMode: SecurityMode = 'local'
): Promise<ProjectConfig> => {
    const api = await getApiClient();
    const response = await api.post('/projects', {
        name,
        path,
        task_type: taskType,
        framework,
        industry_profile: industryProfile,
        security_mode: securityMode,
    });
    return response.data;
};

export const loadProject = async (path: string): Promise<ProjectConfig> => {
    const api = await getApiClient();
    const response = await api.get('/projects/load', { params: { path } });
    return response.data;
};

export const saveProject = async (project: ProjectConfig, path: string): Promise<void> => {
    const api = await getApiClient();
    await api.post('/projects/save', { project, path });
};

// Dataset operations
export const importImages = async (
    projectPath: string,
    imagePaths: string[]
): Promise<{ imported: number; failed: string[] }> => {
    const api = await getApiClient();
    const response = await api.post('/datasets/import', {
        project_path: projectPath,
        image_paths: imagePaths,
    });
    return response.data;
};

export const importVideo = async (
    projectPath: string,
    videoPath: string,
    frameInterval: number = 1
): Promise<{ extracted: number }> => {
    const api = await getApiClient();
    const response = await api.post('/datasets/import-video', {
        project_path: projectPath,
        video_path: videoPath,
        frame_interval: frameInterval,
    });
    return response.data;
};

export const getDatasetPreview = async (
    projectPath: string,
    page: number = 1,
    pageSize: number = 50
): Promise<{ images: string[]; total: number }> => {
    const api = await getApiClient();
    const response = await api.get('/datasets/preview', {
        params: { project_path: projectPath, page, page_size: pageSize },
    });
    return response.data;
};

export const deleteImage = async (
    projectPath: string,
    imagePath: string
): Promise<{ success: boolean }> => {
    const api = await getApiClient();
    const response = await api.delete('/datasets/image', {
        params: { project_path: projectPath, image_path: imagePath },
    });
    return response.data;
};

export const extractVideoFrames = async (
    projectPath: string,
    videoPath: string,
    frameInterval: number = 30,
    maxFrames: number | null = null
): Promise<any> => {
    const api = await getApiClient();
    const response = await api.post('/datasets/extract-frames', {
        project_path: projectPath,
        video_path: videoPath,
        frame_interval: frameInterval,
        max_frames: maxFrames,
    });
    return response.data;
};

export const validateDataset = async (projectPath: string) => {
    const api = await getApiClient();
    const response = await api.post('/datasets/validate', {
        project_path: projectPath,
    });
    return response.data;
};

export const splitDataset = async (
    projectPath: string,
    trainRatio: number,
    valRatio: number,
    testRatio: number
): Promise<{ train: number; val: number; test: number }> => {
    const api = await getApiClient();
    const response = await api.post('/datasets/split', {
        project_path: projectPath,
        train_ratio: trainRatio,
        val_ratio: valRatio,
        test_ratio: testRatio,
    });
    return response.data;
};

// Annotation operations
export const saveAnnotations = async (
    projectPath: string,
    imageId: string,
    annotations: unknown
): Promise<void> => {
    const api = await getApiClient();
    await api.post('/annotations/save', {
        project_path: projectPath,
        image_id: imageId,
        annotations,
    });
};

export const loadAnnotations = async (projectPath: string, imageId: string) => {
    const api = await getApiClient();
    const response = await api.get('/annotations/load', {
        params: { project_path: projectPath, image_id: imageId },
    });
    return response.data;
};

export const exportAnnotations = async (
    projectPath: string,
    format: 'coco' | 'yolo' | 'voc'
): Promise<string> => {
    const api = await getApiClient();
    const response = await api.post('/annotations/export', {
        project_path: projectPath,
        format,
    });
    return response.data.output_path;
};

// Model operations
export const getAvailableModels = async (taskType: ProjectConfig['taskType']) => {
    const api = await getApiClient();
    const response = await api.get('/training/available', {
        params: { task_type: taskType },
    });
    return response.data;
};

// Training operations
export const startTraining = async (
    projectPath: string,
    modelName: string,
    config: any
): Promise<string> => {
    const api = await getApiClient();
    const response = await api.post('/training/start', {
        project_path: projectPath,
        model_name: modelName,
        config,
    });
    return response.data.job_id;
};

export const stopTraining = async (jobId: string): Promise<void> => {
    const api = await getApiClient();
    await api.post(`/training/${jobId}/stop`);
};

export const getTrainingStatus = async (jobId: string) => {
    const api = await getApiClient();
    const response = await api.get(`/training/${jobId}/status`);
    return response.data;
};

// Inference operations
export const getTrainedModels = async (projectPath: string) => {
    const api = await getApiClient();
    const response = await api.get('/inference/models', {
        params: { project_path: projectPath },
    });
    return response.data.models;
};

export const runImageInference = async (
    projectPath: string,
    modelPath: string,
    imagePath: string,
    confThreshold: number = 0.25,
    iouThreshold: number = 0.45
): Promise<any> => {
    const api = await getApiClient();
    const response = await api.post('/inference/run', {
        project_path: projectPath,
        model_path: modelPath,
        image_path: imagePath,
        conf_threshold: confThreshold,
        iou_threshold: iouThreshold,
    });
    return response.data;
};

export const runBatchInference = async (
    projectPath: string,
    modelPath: string,
    imagePaths: string[],
    confThreshold: number = 0.25,
    iouThreshold: number = 0.45
): Promise<any> => {
    const api = await getApiClient();
    const response = await api.post('/inference/batch', {
        project_path: projectPath,
        model_path: modelPath,
        image_paths: imagePaths,
        conf_threshold: confThreshold,
        iou_threshold: iouThreshold,
    });
    return response.data.results;
};

export const runVideoInference = async (
    modelPath: string,
    videoPath: string,
    outputPath: string | null = null,
    confThreshold: number = 0.25,
    iouThreshold: number = 0.45
): Promise<any> => {
    const api = await getApiClient();
    const response = await api.post('/inference/video', {
        model_path: modelPath,
        video_path: videoPath,
        output_path: outputPath,
        conf_threshold: confThreshold,
        iou_threshold: iouThreshold,
    });
    return response.data;
};

// Export operations
export const getExportableModels = async (projectPath: string) => {
    const api = await getApiClient();
    const response = await api.get('/export/models', {
        params: { project_path: projectPath },
    });
    return response.data.models;
};

export const getExportFormats = async () => {
    const api = await getApiClient();
    const response = await api.get('/export/formats');
    return response.data.formats;
};

export const exportModel = async (
    modelPath: string,
    outputDir: string,
    format: string = 'onnx',
    imgsz: number = 640,
    half: boolean = false,
    dynamic: boolean = false
): Promise<any> => {
    const api = await getApiClient();
    const response = await api.post('/export/run', {
        model_path: modelPath,
        output_dir: outputDir,
        format,
        imgsz,
        half,
        dynamic,
    });
    return response.data;
};

// Model Hub - Download YOLO model to project folder
export const downloadYoloModel = async (
    modelName: string,
    projectPath: string,
    downloadUrl: string
): Promise<{ status: string; model_name: string; path: string }> => {
    const api = await getApiClient();
    const response = await api.post('/model-hub/download-yolo', {
        model_name: modelName,
        project_path: projectPath,
        download_url: downloadUrl,
    }, { timeout: 300000 }); // 5 min timeout for large models
    return response.data;
};

// Get models downloaded to a project
export const getProjectModels = async (projectPath: string): Promise<{ name: string; path: string; size: number }[]> => {
    const api = await getApiClient();
    const response = await api.get('/model-hub/project-models', {
        params: { project_path: projectPath },
    });
    return response.data.models;
};

// ============================================================
// AUGMENTATION API
// ============================================================

export interface AugmentationConfig {
    resize?: { enabled: boolean; width: number; height: number };
    normalize?: boolean;
    grayscale?: boolean;
    horizontal_flip?: { enabled: boolean; prob: number };
    vertical_flip?: { enabled: boolean; prob: number };
    rotation?: { enabled: boolean; prob: number; max_angle: number };
    brightness?: { enabled: boolean; prob: number; range: number };
    contrast?: { enabled: boolean; prob: number; range: number };
    blur?: { enabled: boolean; prob: number; max_kernel: number };
    noise?: { enabled: boolean; prob: number; intensity: number };
    cutout?: { enabled: boolean; prob: number; size: number };
}

export const previewAugmentation = async (
    imagePath: string,
    projectPath: string,
    config: AugmentationConfig
): Promise<{ success: boolean; image: string; width: number; height: number }> => {
    const api = await getApiClient();
    const response = await api.post('/augmentation/preview-from-path', {
        project_path: projectPath,
        image_path: imagePath,
        config,
    });
    return response.data;
};

export const batchAugment = async (
    projectPath: string,
    config: AugmentationConfig,
    numAugmentedPerImage: number = 3
): Promise<{ success: boolean; original_count: number; augmented_count: number; output_dir: string }> => {
    const api = await getApiClient();
    const response = await api.post('/augmentation/batch', {
        project_path: projectPath,
        config,
        num_augmented_per_image: numAugmentedPerImage,
    });
    return response.data;
};

export const saveAugmentationConfig = async (
    projectPath: string,
    config: AugmentationConfig
): Promise<{ success: boolean; path: string }> => {
    const api = await getApiClient();
    const response = await api.post('/augmentation/save-config', null, {
        params: { project_path: projectPath },
        data: config,
    });
    return response.data;
};

export const loadAugmentationConfig = async (
    projectPath: string
): Promise<{ config: AugmentationConfig | null }> => {
    const api = await getApiClient();
    const response = await api.get('/augmentation/load-config', {
        params: { project_path: projectPath },
    });
    return response.data;
};

// ============================================================
// EVALUATION API
// ============================================================

export interface EvaluationMetrics {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    mAP: number;
    mAP_50: number;
    mAP_50_95: number;
    class_metrics: { [key: string]: { precision: number; recall: number; ap: number } };
    confusion_matrix: number[][];
    total_predictions: number;
    total_ground_truth: number;
    class_names: string[];
}

export const runEvaluation = async (
    projectPath: string,
    modelPath: string,
    datasetSplit: 'val' | 'test' = 'val',
    confThreshold: number = 0.25,
    iouThreshold: number = 0.5
): Promise<EvaluationMetrics> => {
    const api = await getApiClient();
    const response = await api.post('/evaluation/run', {
        project_path: projectPath,
        model_path: modelPath,
        dataset_split: datasetSplit,
        conf_threshold: confThreshold,
        iou_threshold: iouThreshold,
    });
    return response.data;
};

export const getEvaluationHistory = async (
    projectPath: string,
    limit: number = 10
): Promise<{ evaluations: any[]; count: number }> => {
    const api = await getApiClient();
    const response = await api.get('/evaluation/history', {
        params: { project_path: projectPath, limit },
    });
    return response.data;
};

export const exportEvaluationReport = async (
    projectPath: string,
    format: 'json' | 'csv' = 'json'
): Promise<{ success: boolean; format: string; output_path: string }> => {
    const api = await getApiClient();
    const response = await api.post('/evaluation/export-report', null, {
        params: { project_path: projectPath, format },
    });
    return response.data;
};

// ============================================================
// 3D RECONSTRUCTION API
// ============================================================

export interface ReconstructionConfig {
    feature_detector: 'ORB' | 'SIFT';
    max_features: number;
    match_ratio: number;
    min_matches: number;
}

export const startReconstruction = async (
    imagesDir: string,
    outputDir: string,
    projectPath?: string,
    config?: ReconstructionConfig
): Promise<{ job_id: string; status: string; message: string }> => {
    const api = await getApiClient();
    const response = await api.post('/3d/start', {
        images_dir: imagesDir,
        output_dir: outputDir,
        project_path: projectPath,
        config,
    });
    return response.data;
};

export const getReconstructionStatus = async (
    jobId: string
): Promise<any> => {
    const api = await getApiClient();
    const response = await api.get(`/3d/status/${jobId}`);
    return response.data;
};

export const getPointCloud = async (
    jobId: string,
    format: 'json' | 'file' = 'json'
): Promise<{ num_points: number; points: any[] }> => {
    const api = await getApiClient();
    const response = await api.get(`/3d/point-cloud/${jobId}`, {
        params: { format },
    });
    return response.data;
};

export const measureDistance3D = async (
    pointA: { x: number; y: number; z: number },
    pointB: { x: number; y: number; z: number }
): Promise<{ distance: number; unit: string }> => {
    const api = await getApiClient();
    const response = await api.post('/3d/measure/distance', {
        point_a: pointA,
        point_b: pointB,
    });
    return response.data;
};

// ============================================================
// DATASET ANALYTICS API
// ============================================================

export const getDatasetStats = async (projectPath: string): Promise<{
    total_images: number;
    total_annotations: number;
    class_distribution: { [key: string]: number };
    avg_annotations_per_image: number;
    dataset_size_mb: number;
}> => {
    const api = await getApiClient();
    const response = await api.get('/datasets/stats', {
        params: { project_path: projectPath },
    });
    return response.data;
};
