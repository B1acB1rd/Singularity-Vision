"""
Change Detection Engine for Singularity Vision
Compares before/after images to detect and highlight differences.
Useful for surveillance, infrastructure monitoring, and temporal analysis.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ChangeType(str, Enum):
    """Types of detected changes."""
    ADDITION = "addition"      # New objects appeared
    REMOVAL = "removal"        # Objects disappeared
    MODIFICATION = "modification"  # Objects changed
    NO_CHANGE = "no_change"


@dataclass
class ChangeRegion:
    """Represents a detected change region."""
    x: int
    y: int
    width: int
    height: int
    change_type: ChangeType
    confidence: float
    area_percentage: float  # Percentage of total image area


@dataclass
class ChangeDetectionResult:
    """Result of change detection analysis."""
    has_changes: bool
    change_percentage: float  # Total percentage of image that changed
    regions: List[ChangeRegion]
    diff_image_path: Optional[str]
    overlay_image_path: Optional[str]
    

class ChangeDetectionEngine:
    """
    Engine for detecting changes between two images.
    
    Algorithms supported:
    - Simple difference (fast, good for controlled environments)
    - Structural similarity (SSIM) (robust to lighting changes)
    - Background subtraction (for video sequences)
    """
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        
    def _align_images(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two images using feature matching.
        Useful when camera position may have shifted slightly.
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2
        
        # Resize if different sizes
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # Detect ORB features
        orb = cv2.ORB_create(nfeatures=500)
        kp1, desc1 = orb.detectAndCompute(gray1, None)
        kp2, desc2 = orb.detectAndCompute(gray2, None)
        
        if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
            return image1, image2
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)[:50]
        
        if len(matches) < 4:
            return image1, image2
        
        # Get matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        if H is not None:
            aligned = cv2.warpPerspective(image2, H, (image1.shape[1], image1.shape[0]))
            return image1, aligned
        
        return image1, image2
    
    def detect_changes_simple(
        self,
        before_path: str,
        after_path: str,
        threshold: int = 30,
        min_area: int = 100,
        align: bool = False
    ) -> ChangeDetectionResult:
        """
        Detect changes using simple pixel difference.
        
        Args:
            before_path: Path to the "before" image
            after_path: Path to the "after" image
            threshold: Pixel difference threshold (0-255)
            min_area: Minimum contour area to consider as change
            align: Whether to align images before comparison
            
        Returns:
            ChangeDetectionResult with detected regions
        """
        # Load images
        before = cv2.imread(before_path)
        after = cv2.imread(after_path)
        
        if before is None or after is None:
            return ChangeDetectionResult(
                has_changes=False,
                change_percentage=0.0,
                regions=[],
                diff_image_path=None,
                overlay_image_path=None
            )
        
        # Align if requested
        if align:
            before, after = self._align_images(before, after)
        
        # Ensure same size
        if before.shape != after.shape:
            after = cv2.resize(after, (before.shape[1], before.shape[0]))
        
        # Convert to grayscale
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        before_blur = cv2.GaussianBlur(before_gray, (5, 5), 0)
        after_blur = cv2.GaussianBlur(after_gray, (5, 5), 0)
        
        # Compute absolute difference
        diff = cv2.absdiff(before_blur, after_blur)
        
        # Threshold
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process regions
        total_area = before.shape[0] * before.shape[1]
        change_area = 0
        regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            change_area += area
            
            # Determine change type by comparing mean intensities
            before_region = before_gray[y:y+h, x:x+w]
            after_region = after_gray[y:y+h, x:x+w]
            
            before_mean = np.mean(before_region)
            after_mean = np.mean(after_region)
            
            if after_mean > before_mean + 20:
                change_type = ChangeType.ADDITION
            elif after_mean < before_mean - 20:
                change_type = ChangeType.REMOVAL
            else:
                change_type = ChangeType.MODIFICATION
            
            regions.append(ChangeRegion(
                x=x, y=y, width=w, height=h,
                change_type=change_type,
                confidence=min(1.0, area / min_area * 0.1),
                area_percentage=round((area / total_area) * 100, 2)
            ))
        
        change_percentage = (change_area / total_area) * 100
        
        # Generate output images
        diff_path = None
        overlay_path = None
        
        if self.output_dir and regions:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Diff heatmap
            diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            diff_path = os.path.join(self.output_dir, "diff_heatmap.jpg")
            cv2.imwrite(diff_path, diff_color)
            
            # Overlay with bounding boxes
            overlay = after.copy()
            for region in regions:
                color = (0, 255, 0) if region.change_type == ChangeType.ADDITION else \
                        (0, 0, 255) if region.change_type == ChangeType.REMOVAL else \
                        (0, 255, 255)
                cv2.rectangle(overlay, 
                             (region.x, region.y), 
                             (region.x + region.width, region.y + region.height),
                             color, 2)
            overlay_path = os.path.join(self.output_dir, "change_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)
        
        return ChangeDetectionResult(
            has_changes=len(regions) > 0,
            change_percentage=round(change_percentage, 2),
            regions=regions,
            diff_image_path=diff_path,
            overlay_image_path=overlay_path
        )
    
    def detect_changes_ssim(
        self,
        before_path: str,
        after_path: str,
        threshold: float = 0.9,
        min_area: int = 100
    ) -> ChangeDetectionResult:
        """
        Detect changes using Structural Similarity Index (SSIM).
        More robust to lighting and exposure changes.
        
        Args:
            before_path: Path to the "before" image
            after_path: Path to the "after" image
            threshold: SSIM threshold (0-1, higher = more similar)
            min_area: Minimum contour area to consider
            
        Returns:
            ChangeDetectionResult with detected regions
        """
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            print("scikit-image not installed. Using simple detection.")
            return self.detect_changes_simple(before_path, after_path)
        
        # Load images
        before = cv2.imread(before_path)
        after = cv2.imread(after_path)
        
        if before is None or after is None:
            return ChangeDetectionResult(
                has_changes=False,
                change_percentage=0.0,
                regions=[],
                diff_image_path=None,
                overlay_image_path=None
            )
        
        # Ensure same size
        if before.shape != after.shape:
            after = cv2.resize(after, (before.shape[1], before.shape[0]))
        
        # Convert to grayscale
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM
        score, diff = ssim(before_gray, after_gray, full=True)
        diff = (diff * 255).astype("uint8")
        diff = cv2.bitwise_not(diff)  # Invert so changes are bright
        
        # Threshold the difference
        _, thresh = cv2.threshold(diff, int((1 - threshold) * 255), 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area = before.shape[0] * before.shape[1]
        change_area = 0
        regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            change_area += area
            
            regions.append(ChangeRegion(
                x=x, y=y, width=w, height=h,
                change_type=ChangeType.MODIFICATION,
                confidence=1 - score,
                area_percentage=round((area / total_area) * 100, 2)
            ))
        
        change_percentage = (change_area / total_area) * 100
        
        # Generate output images
        diff_path = None
        overlay_path = None
        
        if self.output_dir and regions:
            os.makedirs(self.output_dir, exist_ok=True)
            
            diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            diff_path = os.path.join(self.output_dir, "ssim_diff.jpg")
            cv2.imwrite(diff_path, diff_color)
            
            overlay = after.copy()
            for region in regions:
                cv2.rectangle(overlay,
                             (region.x, region.y),
                             (region.x + region.width, region.y + region.height),
                             (0, 255, 255), 2)
            overlay_path = os.path.join(self.output_dir, "ssim_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)
        
        return ChangeDetectionResult(
            has_changes=len(regions) > 0,
            change_percentage=round(change_percentage, 2),
            regions=regions,
            diff_image_path=diff_path,
            overlay_image_path=overlay_path
        )
