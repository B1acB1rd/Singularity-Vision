// Industry Profiles for specialized workflows
export type IndustryProfile =
    | 'general'
    | 'defense'
    | 'aviation'
    | 'mining'
    | 'sports'
    | 'health';

// Security modes for data handling
export type SecurityMode = 'local' | 'hybrid' | 'cloud';

// Task types supported by the platform
export type TaskType =
    | 'classification'
    | 'detection'
    | 'segmentation'
    | 'change_detection'
    | 'tracking'
    | 'pose';

// Industry profile metadata
export interface IndustryProfileInfo {
    profile: IndustryProfile;
    name: string;
    description: string;
    icon: string;
    features: string[];
}

// Available industry profiles
export const INDUSTRY_PROFILES: IndustryProfileInfo[] = [
    {
        profile: 'general',
        name: 'General',
        description: 'Standard computer vision workflows',
        icon: 'Layers',
        features: ['Object Detection', 'Classification', 'Segmentation']
    },
    {
        profile: 'defense',
        name: 'Defense & Security',
        description: 'Surveillance, change detection, secure data handling',
        icon: 'Shield',
        features: ['Offline-first', 'Audit Logs', 'Change Detection', 'Secure Handling']
    },
    {
        profile: 'aviation',
        name: 'Aviation',
        description: 'Runway inspection, aircraft analysis',
        icon: 'Plane',
        features: ['High-res Imagery', 'Surface Analysis', 'Runway Inspection']
    },
    {
        profile: 'mining',
        name: 'Mining',
        description: 'Pit mapping, volume estimation, terrain analysis',
        icon: 'Mountain',
        features: ['Volume Estimation', 'Terrain Analysis', 'Equipment Tracking']
    },
    {
        profile: 'sports',
        name: 'Sports',
        description: 'Stadium mapping, player tracking, crowd analysis',
        icon: 'Trophy',
        features: ['Player Tracking', 'Crowd Flow', 'Camera Optimization']
    },
    {
        profile: 'health',
        name: 'Health & Emergency',
        description: 'Disaster mapping, damage assessment',
        icon: 'Heart',
        features: ['Privacy-first', 'Edge Deployment', 'Damage Assessment']
    }
];

// Project template configuration
export interface ProjectTemplate {
    id: string;
    name: string;
    description: string;
    icon: string;
    category: 'cv' | 'spatial' | 'industry' | 'quickstart';
    taskType: TaskType;
    industryProfile: IndustryProfile;
    securityMode: SecurityMode;
    suggestedClasses?: string[];
    defaultModel?: string;
}

// Pre-configured project templates
export const PROJECT_TEMPLATES: ProjectTemplate[] = [
    // CV Basics
    {
        id: 'detection-basic',
        name: 'Object Detection',
        description: 'Train models to detect and locate objects in images',
        icon: 'ScanSearch',
        category: 'cv',
        taskType: 'detection',
        industryProfile: 'general',
        securityMode: 'local',
        defaultModel: 'yolov8n'
    },
    {
        id: 'classification-basic',
        name: 'Image Classification',
        description: 'Classify images into categories',
        icon: 'Tags',
        category: 'cv',
        taskType: 'classification',
        industryProfile: 'general',
        securityMode: 'local',
        defaultModel: 'yolov8n-cls'
    },
    {
        id: 'segmentation-basic',
        name: 'Instance Segmentation',
        description: 'Pixel-level object detection with masks',
        icon: 'Shapes',
        category: 'cv',
        taskType: 'segmentation',
        industryProfile: 'general',
        securityMode: 'local',
        defaultModel: 'yolov8n-seg'
    },
    // Spatial / Mapping
    {
        id: 'satellite-mapping',
        name: 'Satellite Imagery',
        description: 'Process geo-referenced satellite images',
        icon: 'Globe',
        category: 'spatial',
        taskType: 'detection',
        industryProfile: 'general',
        securityMode: 'local',
        suggestedClasses: ['building', 'road', 'vegetation', 'water']
    },
    {
        id: 'drone-inspection',
        name: 'Drone Inspection',
        description: 'Analyze drone footage for infrastructure',
        icon: 'Plane',
        category: 'spatial',
        taskType: 'detection',
        industryProfile: 'aviation',
        securityMode: 'local',
        suggestedClasses: ['damage', 'crack', 'rust', 'debris']
    },
    {
        id: 'change-detection',
        name: 'Change Detection',
        description: 'Compare before/after imagery for changes',
        icon: 'GitCompare',
        category: 'spatial',
        taskType: 'change_detection',
        industryProfile: 'defense',
        securityMode: 'local'
    },
    // Industry Quick Starts
    {
        id: 'security-surveillance',
        name: 'Security Surveillance',
        description: 'Person/vehicle detection for security',
        icon: 'Shield',
        category: 'industry',
        taskType: 'detection',
        industryProfile: 'defense',
        securityMode: 'local',
        suggestedClasses: ['person', 'vehicle', 'weapon']
    },
    {
        id: 'sports-tracking',
        name: 'Sports Analytics',
        description: 'Player and ball tracking for sports',
        icon: 'Trophy',
        category: 'industry',
        taskType: 'tracking',
        industryProfile: 'sports',
        securityMode: 'local',
        suggestedClasses: ['player', 'ball', 'referee']
    }
];

// Project configuration schema
export interface ProjectConfig {
    id: string;
    name: string;
    description: string;
    createdAt: string;
    updatedAt: string;
    taskType: TaskType;
    framework: 'pytorch' | 'tensorflow';
    industryProfile: IndustryProfile;
    securityMode: SecurityMode;
    datasetInfo: DatasetInfo | null;
    modelInfo: ModelInfo | null;
    trainingConfig: TrainingConfig | null;
}

export interface DatasetInfo {
    totalImages: number;
    classes: ClassInfo[];
    trainCount: number;
    valCount: number;
    testCount: number;
    imageSize: { width: number; height: number } | null;
    hasLabels: boolean;
}

export interface ClassInfo {
    id: number;
    name: string;
    color: string;
    count: number;
}

export interface ModelInfo {
    architecture: string;
    pretrained: boolean;
    inputSize: [number, number];
    numClasses: number;
    selectedModel: string;
}

export interface TrainingConfig {
    epochs: number;
    batchSize: number;
    learningRate: number;
    optimizer: 'adam' | 'sgd' | 'adamw';
    trainSplit: number;
    valSplit: number;
    testSplit: number;
    augmentations: AugmentationConfig;
}

export interface AugmentationConfig {
    horizontalFlip: boolean;
    verticalFlip: boolean;
    rotation: number;
    brightness: number;
    contrast: number;
    saturation: number;
    blur: boolean;
    noise: boolean;
}

// Annotation types
export interface TrainingStatus {
    status: string;
    progress: number;
    current_epoch: number;
    total_epochs: number;
    logs: string[];
    metrics?: {
        map50: number;
        map5095: number;
        loss: number;
    };
    error?: string;
}

export interface BoundingBox {
    id: string;
    x: number;
    y: number;
    width: number;
    height: number;
    classId: number;
    className: string;
}

// Point for polygon vertices (normalized 0-1 coordinates)
export interface Point {
    x: number;
    y: number;
}

// Polygon annotation for segmentation
export interface Polygon {
    id: string;
    points: Point[];  // Array of vertices
    classId: number;
    className: string;
    closed: boolean;  // Whether the polygon is complete
}

export interface ImageAnnotation {
    imageId: string;
    imagePath: string;
    width: number;
    height: number;
    boxes: BoundingBox[];
    polygons?: Polygon[];  // Optional for backward compatibility
}

// Inference results
export interface DetectionResult {
    boxes: {
        x1: number;
        y1: number;
        x2: number;
        y2: number;
        classId: number;
        className: string;
        confidence: number;
    }[];
    inferenceTime: number;
}

export interface ClassificationResult {
    predictions: {
        classId: number;
        className: string;
        confidence: number;
    }[];
    inferenceTime: number;
}

// Export options
export interface ExportOptions {
    format: 'pytorch' | 'tensorflow' | 'onnx';
    includeScript: boolean;
    includeMetadata: boolean;
    outputPath: string;
}
