"""
3D Reconstruction Engine for Singularity Vision

3D is a CORE DIFFERENTIATOR, not an optional feature.
v1.0 includes basic local CPU-based Structure-from-Motion.
Future: online acceleration as upgrade, not replacement.

Key Features:
- Feature extraction and matching
- Camera pose estimation  
- Sparse point cloud generation
- Detection projection to 3D
- Distance/volume measurement

Use Cases:
- Mining: Volume estimation, terrain analysis
- Construction: Progress monitoring
- Defense: 3D mapping, surveillance
"""

import os
import cv2
import numpy as np
import json
import uuid
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReconstructionStatus(Enum):
    """Status of a reconstruction job."""
    PENDING = "pending"
    EXTRACTING_FEATURES = "extracting_features"
    MATCHING_FEATURES = "matching_features"
    ESTIMATING_POSES = "estimating_poses"
    GENERATING_POINTCLOUD = "generating_pointcloud"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReconstructionConfig:
    """Configuration for 3D reconstruction."""
    # Feature detection
    feature_detector: str = "ORB"  # ORB (fast) or SIFT (accurate but slow)
    max_features: int = 5000
    
    # Matching
    match_ratio: float = 0.75  # Lowe's ratio test threshold
    min_matches: int = 50  # Minimum matches to consider pair
    
    # Reconstruction
    min_triangulation_angle: float = 2.0  # degrees
    
    # Output
    output_format: str = "ply"  # Point cloud format
    
    def to_dict(self) -> Dict:
        return {
            "feature_detector": self.feature_detector,
            "max_features": self.max_features,
            "match_ratio": self.match_ratio,
            "min_matches": self.min_matches,
            "min_triangulation_angle": self.min_triangulation_angle,
            "output_format": self.output_format
        }


@dataclass
class Point3D:
    """A point in 3D space."""
    x: float
    y: float
    z: float
    color: Optional[Tuple[int, int, int]] = None  # RGB
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def to_dict(self) -> Dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "color": self.color
        }


@dataclass
class CameraPose:
    """Camera pose for an image."""
    image_name: str
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector
    focal_length: float
    principal_point: Tuple[float, float]
    
    def to_dict(self) -> Dict:
        return {
            "image_name": self.image_name,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "focal_length": self.focal_length,
            "principal_point": self.principal_point
        }
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get 3x4 projection matrix."""
        K = np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])
        RT = np.hstack([self.rotation, self.translation])
        return K @ RT


@dataclass
class ReconstructionJob:
    """A 3D reconstruction job."""
    job_id: str
    project_path: str
    images_dir: str
    output_dir: str
    config: ReconstructionConfig
    status: ReconstructionStatus
    progress: float = 0.0
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "result": self.result
        }


class ReconstructionEngine:
    """
    3D reconstruction engine using Structure-from-Motion.
    
    v1.0: CPU-based using OpenCV
    Future: Optional GPU acceleration or cloud offload
    
    This is a CORE DIFFERENTIATOR - not optional.
    """
    
    def __init__(self, project_path: Optional[str] = None):
        self.project_path = project_path
        self._jobs: Dict[str, ReconstructionJob] = {}
        self._job_threads: Dict[str, threading.Thread] = {}
    
    def start_reconstruction(
        self,
        images_dir: str,
        output_dir: str,
        config: Optional[ReconstructionConfig] = None,
        project_path: Optional[str] = None
    ) -> str:
        """
        Start a 3D reconstruction job.
        
        Args:
            images_dir: Directory containing input images
            output_dir: Where to save reconstruction results
            config: Reconstruction configuration
            project_path: Project path for metadata storage
        
        Returns:
            job_id: Unique identifier for this job
        """
        job_id = str(uuid.uuid4())[:8]
        config = config or ReconstructionConfig()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        job = ReconstructionJob(
            job_id=job_id,
            project_path=project_path or self.project_path,
            images_dir=images_dir,
            output_dir=output_dir,
            config=config,
            status=ReconstructionStatus.PENDING,
            progress=0.0,
            message="Initializing...",
            started_at=datetime.now().isoformat()
        )
        
        self._jobs[job_id] = job
        
        # Start reconstruction in background thread
        thread = threading.Thread(
            target=self._run_reconstruction,
            args=(job_id,),
            daemon=True
        )
        self._job_threads[job_id] = thread
        thread.start()
        
        return job_id
    
    def get_status(self, job_id: str) -> Optional[Dict]:
        """Get the status of a reconstruction job."""
        job = self._jobs.get(job_id)
        if job:
            return job.to_dict()
        return None
    
    def list_jobs(self) -> List[Dict]:
        """List all reconstruction jobs."""
        return [job.to_dict() for job in self._jobs.values()]
    
    def _run_reconstruction(self, job_id: str):
        """Run the actual reconstruction process."""
        job = self._jobs[job_id]
        
        try:
            # Load images
            job.status = ReconstructionStatus.EXTRACTING_FEATURES
            job.message = "Loading and analyzing images..."
            job.progress = 0.1
            
            images, image_names = self._load_images(job.images_dir)
            if len(images) < 2:
                raise ValueError(f"Need at least 2 images, found {len(images)}")
            
            logger.info(f"Loaded {len(images)} images for reconstruction")
            
            # Extract features
            job.message = f"Extracting features from {len(images)} images..."
            job.progress = 0.2
            
            keypoints_list, descriptors_list = self._extract_features(
                images, job.config
            )
            
            # Match features between image pairs
            job.status = ReconstructionStatus.MATCHING_FEATURES
            job.message = "Matching features between images..."
            job.progress = 0.4
            
            matches_dict = self._match_features(
                descriptors_list, image_names, job.config
            )
            
            if not matches_dict:
                raise ValueError("No sufficient matches found between images")
            
            # Estimate camera poses
            job.status = ReconstructionStatus.ESTIMATING_POSES
            job.message = "Estimating camera poses..."
            job.progress = 0.6
            
            camera_poses = self._estimate_poses(
                images, keypoints_list, matches_dict, image_names
            )
            
            # Generate point cloud
            job.status = ReconstructionStatus.GENERATING_POINTCLOUD
            job.message = "Generating 3D point cloud..."
            job.progress = 0.8
            
            points_3d, colors = self._triangulate_points(
                images, keypoints_list, matches_dict, camera_poses, image_names
            )
            
            # Save results
            output_path = self._save_point_cloud(
                points_3d, colors, job.output_dir, job.config.output_format
            )
            
            # Save camera poses
            poses_path = os.path.join(job.output_dir, "camera_poses.json")
            with open(poses_path, 'w') as f:
                json.dump([p.to_dict() for p in camera_poses], f, indent=2)
            
            # Complete
            job.status = ReconstructionStatus.COMPLETED
            job.progress = 1.0
            job.message = "Reconstruction complete!"
            job.completed_at = datetime.now().isoformat()
            job.result = {
                "point_cloud_path": output_path,
                "poses_path": poses_path,
                "num_points": len(points_3d),
                "num_cameras": len(camera_poses),
                "num_images_used": len(images)
            }
            
            logger.info(f"Reconstruction {job_id} completed: {len(points_3d)} points")
            
        except Exception as e:
            logger.error(f"Reconstruction {job_id} failed: {e}")
            job.status = ReconstructionStatus.FAILED
            job.error = str(e)
            job.message = f"Failed: {e}"
    
    def _load_images(self, images_dir: str) -> Tuple[List[np.ndarray], List[str]]:
        """Load all images from directory."""
        supported = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        images = []
        names = []
        
        for filename in sorted(os.listdir(images_dir)):
            if os.path.splitext(filename)[1].lower() in supported:
                path = os.path.join(images_dir, filename)
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    names.append(filename)
        
        return images, names
    
    def _extract_features(
        self, 
        images: List[np.ndarray],
        config: ReconstructionConfig
    ) -> Tuple[List, List]:
        """Extract keypoints and descriptors from images."""
        # Create feature detector
        if config.feature_detector == "SIFT":
            detector = cv2.SIFT_create(nfeatures=config.max_features)
        else:  # ORB (default - faster)
            detector = cv2.ORB_create(nfeatures=config.max_features)
        
        keypoints_list = []
        descriptors_list = []
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = detector.detectAndCompute(gray, None)
            keypoints_list.append(kp)
            descriptors_list.append(desc)
        
        return keypoints_list, descriptors_list
    
    def _match_features(
        self,
        descriptors_list: List,
        image_names: List[str],
        config: ReconstructionConfig
    ) -> Dict[Tuple[int, int], List]:
        """Match features between all image pairs."""
        n = len(descriptors_list)
        matches_dict = {}
        
        # Use BFMatcher for ORB, FLANN for SIFT
        if config.feature_detector == "SIFT":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        for i in range(n):
            for j in range(i + 1, n):
                if descriptors_list[i] is None or descriptors_list[j] is None:
                    continue
                
                try:
                    matches = matcher.knnMatch(
                        descriptors_list[i], 
                        descriptors_list[j], 
                        k=2
                    )
                    
                    # Apply Lowe's ratio test
                    good_matches = []
                    for m_n in matches:
                        if len(m_n) == 2:
                            m, n = m_n
                            if m.distance < config.match_ratio * n.distance:
                                good_matches.append(m)
                    
                    if len(good_matches) >= config.min_matches:
                        matches_dict[(i, j)] = good_matches
                        logger.debug(
                            f"Matched {image_names[i]} <-> {image_names[j]}: "
                            f"{len(good_matches)} matches"
                        )
                except Exception as e:
                    logger.warning(f"Failed to match {i} <-> {j}: {e}")
        
        return matches_dict
    
    def _estimate_poses(
        self,
        images: List[np.ndarray],
        keypoints_list: List,
        matches_dict: Dict,
        image_names: List[str]
    ) -> List[CameraPose]:
        """Estimate camera poses from matched features."""
        camera_poses = []
        
        if not matches_dict:
            return camera_poses
        
        # Estimate intrinsic camera matrix (assume principal point at center)
        h, w = images[0].shape[:2]
        focal_length = max(h, w)  # Rough estimate
        cx, cy = w / 2, h / 2
        
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Initialize first camera at origin
        camera_poses.append(CameraPose(
            image_name=image_names[0],
            rotation=np.eye(3),
            translation=np.zeros((3, 1)),
            focal_length=focal_length,
            principal_point=(cx, cy)
        ))
        
        # Find best pair for initialization
        best_pair = None
        best_matches = 0
        for (i, j), matches in matches_dict.items():
            if 0 in (i, j) and len(matches) > best_matches:
                best_pair = (i, j)
                best_matches = len(matches)
        
        if best_pair and best_matches > 0:
            i, j = best_pair
            other_idx = j if i == 0 else i
            matches = matches_dict[best_pair]
            
            # Get matched points
            pts1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches])
            pts2 = np.float32([keypoints_list[j][m.trainIdx].pt for m in matches])
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
                
                camera_poses.append(CameraPose(
                    image_name=image_names[other_idx],
                    rotation=R,
                    translation=t,
                    focal_length=focal_length,
                    principal_point=(cx, cy)
                ))
        
        return camera_poses
    
    def _triangulate_points(
        self,
        images: List[np.ndarray],
        keypoints_list: List,
        matches_dict: Dict,
        camera_poses: List[CameraPose],
        image_names: List[str]
    ) -> Tuple[List[Point3D], List[Tuple[int, int, int]]]:
        """Triangulate 3D points from matched 2D points and camera poses."""
        points_3d = []
        colors = []
        
        if len(camera_poses) < 2:
            return points_3d, colors
        
        # Get projection matrices
        P1 = camera_poses[0].get_projection_matrix()
        P2 = camera_poses[1].get_projection_matrix()
        
        # Find the matching pair
        idx1 = image_names.index(camera_poses[0].image_name)
        idx2 = image_names.index(camera_poses[1].image_name)
        
        pair = (min(idx1, idx2), max(idx1, idx2))
        if pair not in matches_dict:
            return points_3d, colors
        
        matches = matches_dict[pair]
        
        # Get matched points
        pts1 = np.float32([keypoints_list[idx1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints_list[idx2][m.trainIdx].pt for m in matches])
        
        # Triangulate
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = pts4d[:3] / pts4d[3]  # Convert from homogeneous
        
        # Get colors from first image
        img1 = images[idx1]
        
        for i, pt2d in enumerate(pts1):
            x, y, z = pts3d[:, i]
            
            # Filter invalid points
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                # Get color from image
                px, py = int(pt2d[0]), int(pt2d[1])
                if 0 <= px < img1.shape[1] and 0 <= py < img1.shape[0]:
                    bgr = img1[py, px]
                    color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # BGR -> RGB
                else:
                    color = (128, 128, 128)
                
                points_3d.append(Point3D(x=float(x), y=float(y), z=float(z), color=color))
                colors.append(color)
        
        return points_3d, colors
    
    def _save_point_cloud(
        self,
        points: List[Point3D],
        colors: List[Tuple[int, int, int]],
        output_dir: str,
        format: str = "ply"
    ) -> str:
        """Save point cloud to file."""
        output_path = os.path.join(output_dir, f"point_cloud.{format}")
        
        if format == "ply":
            with open(output_path, 'w') as f:
                # PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                # Points
                for point in points:
                    color = point.color or (128, 128, 128)
                    f.write(f"{point.x} {point.y} {point.z} {color[0]} {color[1]} {color[2]}\n")
        
        return output_path
    
    # =========================================================================
    # MEASUREMENT TOOLS
    # =========================================================================
    
    def measure_distance(self, point_a: Point3D, point_b: Point3D) -> float:
        """
        Calculate Euclidean distance between two 3D points.
        
        Note: Units depend on the scale of reconstruction.
        For real-world measurements, scale factor must be calibrated.
        """
        a = point_a.to_array()
        b = point_b.to_array()
        return float(np.linalg.norm(a - b))
    
    def calculate_bounding_box(self, points: List[Point3D]) -> Dict:
        """Calculate bounding box of point cloud."""
        if not points:
            return {}
        
        coords = np.array([[p.x, p.y, p.z] for p in points])
        min_pt = coords.min(axis=0)
        max_pt = coords.max(axis=0)
        dimensions = max_pt - min_pt
        
        return {
            "min": {"x": float(min_pt[0]), "y": float(min_pt[1]), "z": float(min_pt[2])},
            "max": {"x": float(max_pt[0]), "y": float(max_pt[1]), "z": float(max_pt[2])},
            "dimensions": {
                "width": float(dimensions[0]),
                "height": float(dimensions[1]),
                "depth": float(dimensions[2])
            },
            "volume_estimate": float(np.prod(dimensions))
        }
    
    def calculate_volume(self, points: List[Point3D]) -> float:
        """
        Estimate volume of convex hull of points.
        
        Useful for mining volume estimation.
        """
        if len(points) < 4:
            return 0.0
        
        try:
            from scipy.spatial import ConvexHull
            coords = np.array([[p.x, p.y, p.z] for p in points])
            hull = ConvexHull(coords)
            return float(hull.volume)
        except ImportError:
            # Fallback to bounding box volume
            bbox = self.calculate_bounding_box(points)
            return bbox.get("volume_estimate", 0.0)
        except Exception:
            return 0.0
    
    def load_point_cloud(self, path: str) -> List[Point3D]:
        """Load a PLY point cloud file."""
        points = []
        
        if not os.path.exists(path):
            return points
        
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # Find end of header
        header_end = 0
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                header_end = i + 1
                break
        
        # Parse points
        for line in lines[header_end:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                color = None
                if len(parts) >= 6:
                    color = (int(parts[3]), int(parts[4]), int(parts[5]))
                points.append(Point3D(x=x, y=y, z=z, color=color))
        
        return points


# Singleton instance
reconstruction_engine = ReconstructionEngine()
