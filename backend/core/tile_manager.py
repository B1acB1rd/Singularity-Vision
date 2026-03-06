"""
Tile Merging and Geo-Overlay for Singularity Vision

Handles:
- Breaking large images into tiles (for performance)
- Geo-aligned tile boundaries
- Merging inference results across tiles
- Accurate geo-overlay for predictions

Design Principles:
- REPRODUCIBLE: Same image + config = same tiles
- ACCURATE: No lost predictions at tile boundaries
- GEO-ALIGNED: Tiles maintain geographic coordinates
"""

import os
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class TileConfig:
    """Configuration for image tiling."""
    tile_size: int = 1024  # Pixels
    overlap: int = 128  # Overlap pixels between tiles
    min_tile_size: int = 256  # Minimum tile dimension
    
    def to_dict(self) -> Dict:
        return {
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "min_tile_size": self.min_tile_size
        }


@dataclass
class Tile:
    """A single tile from a larger image."""
    tile_id: str
    source_image: str
    row: int
    col: int
    x_offset: int  # Pixel offset in source image
    y_offset: int
    width: int
    height: int
    
    # Geographic bounds (if available)
    geo_bounds: Optional[Tuple[float, float, float, float]] = None  # (min_lon, min_lat, max_lon, max_lat)
    
    def to_dict(self) -> Dict:
        return {
            "tile_id": self.tile_id,
            "source_image": self.source_image,
            "row": self.row,
            "col": self.col,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "width": self.width,
            "height": self.height,
            "geo_bounds": self.geo_bounds
        }


@dataclass
class Detection:
    """A detection result from inference."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    mask: Optional[np.ndarray] = None
    
    # Geographic coordinates (if geo-referenced)
    geo_bbox: Optional[Tuple[float, float, float, float]] = None  # (min_lon, min_lat, max_lon, max_lat)
    
    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "geo_bbox": self.geo_bbox
        }


@dataclass
class MergedResult:
    """Merged detection results from all tiles."""
    source_image: str
    detections: List[Detection]
    tile_count: int
    image_size: Tuple[int, int]  # (width, height)
    geo_bounds: Optional[Tuple[float, float, float, float]] = None
    
    def to_dict(self) -> Dict:
        return {
            "source_image": self.source_image,
            "detections": [d.to_dict() for d in self.detections],
            "tile_count": self.tile_count,
            "image_size": self.image_size,
            "geo_bounds": self.geo_bounds
        }


class TileManager:
    """
    Manages image tiling for large image processing.
    
    Key Features:
    - Deterministic tiling: Same image + config = same tiles
    - Overlap handling: Prevents missed detections at boundaries
    - Geo-alignment: Tiles maintain geographic coordinates
    """
    
    def __init__(self, config: Optional[TileConfig] = None):
        self.config = config or TileConfig()
    
    def needs_tiling(self, image_path: str, max_dimension: int = 4096) -> bool:
        """Check if an image needs to be tiled based on size."""
        width, height = self._get_image_size(image_path)
        return width > max_dimension or height > max_dimension
    
    def create_tiles(
        self, 
        image_path: str,
        output_dir: str,
        geo_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> List[Tile]:
        """
        Tile an image into smaller pieces.
        
        DETERMINISTIC: Same image + config always produces same tiles.
        
        Args:
            image_path: Path to source image
            output_dir: Directory to save tiles
            geo_bounds: Geographic bounds (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            List of Tile objects
        """
        os.makedirs(output_dir, exist_ok=True)
        
        width, height = self._get_image_size(image_path)
        
        # Calculate tile grid
        step = self.config.tile_size - self.config.overlap
        cols = math.ceil(max(1, (width - self.config.overlap) / step))
        rows = math.ceil(max(1, (height - self.config.overlap) / step))
        
        tiles = []
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Load image
        img = self._load_image(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return []
        
        for row in range(rows):
            for col in range(cols):
                # Calculate tile bounds
                x_offset = col * step
                y_offset = row * step
                
                # Clamp to image bounds
                tile_width = min(self.config.tile_size, width - x_offset)
                tile_height = min(self.config.tile_size, height - y_offset)
                
                # Skip if tile is too small
                if tile_width < self.config.min_tile_size or tile_height < self.config.min_tile_size:
                    continue
                
                # Extract tile
                tile_img = img[y_offset:y_offset+tile_height, x_offset:x_offset+tile_width]
                
                # Generate deterministic tile ID
                tile_id = f"{image_name}_r{row:03d}_c{col:03d}"
                
                # Calculate geo bounds for this tile
                tile_geo = None
                if geo_bounds:
                    tile_geo = self._calculate_tile_geo_bounds(
                        geo_bounds, width, height,
                        x_offset, y_offset, tile_width, tile_height
                    )
                
                # Save tile
                tile_path = os.path.join(output_dir, f"{tile_id}.jpg")
                self._save_image(tile_img, tile_path)
                
                tile = Tile(
                    tile_id=tile_id,
                    source_image=image_path,
                    row=row,
                    col=col,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    width=tile_width,
                    height=tile_height,
                    geo_bounds=tile_geo
                )
                
                tiles.append(tile)
        
        logger.info(f"Created {len(tiles)} tiles from {image_path} ({rows}x{cols} grid)")
        return tiles
    
    def _calculate_tile_geo_bounds(
        self,
        image_geo: Tuple[float, float, float, float],
        img_width: int,
        img_height: int,
        x_offset: int,
        y_offset: int,
        tile_width: int,
        tile_height: int
    ) -> Tuple[float, float, float, float]:
        """Calculate geographic bounds for a tile."""
        min_lon, min_lat, max_lon, max_lat = image_geo
        
        # Calculate pixel to geo conversion
        lon_per_pixel = (max_lon - min_lon) / img_width
        lat_per_pixel = (max_lat - min_lat) / img_height
        
        # Note: Image y-axis is typically inverted (0 at top)
        tile_min_lon = min_lon + (x_offset * lon_per_pixel)
        tile_max_lon = min_lon + ((x_offset + tile_width) * lon_per_pixel)
        tile_max_lat = max_lat - (y_offset * lat_per_pixel)
        tile_min_lat = max_lat - ((y_offset + tile_height) * lat_per_pixel)
        
        return (tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat)
    
    def _get_image_size(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions without loading full image."""
        try:
            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    return img.size  # (width, height)
            elif CV2_AVAILABLE:
                img = cv2.imread(image_path)
                return (img.shape[1], img.shape[0])
        except Exception as e:
            logger.error(f"Could not get image size: {e}")
        return (0, 0)
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image as numpy array."""
        try:
            if CV2_AVAILABLE:
                return cv2.imread(image_path)
            elif PIL_AVAILABLE:
                img = Image.open(image_path)
                return np.array(img)
        except Exception as e:
            logger.error(f"Could not load image: {e}")
        return None
    
    def _save_image(self, img: np.ndarray, path: str) -> bool:
        """Save image to disk."""
        try:
            if CV2_AVAILABLE:
                cv2.imwrite(path, img)
                return True
            elif PIL_AVAILABLE:
                Image.fromarray(img).save(path)
                return True
        except Exception as e:
            logger.error(f"Could not save image: {e}")
        return False


class ResultMerger:
    """
    Merges detection results from multiple tiles.
    
    Handles:
    - Duplicate detection removal (objects at tile boundaries)
    - Coordinate transformation to source image space
    - Geographic coordinate assignment
    """
    
    def __init__(self, nms_threshold: float = 0.5):
        self.nms_threshold = nms_threshold
    
    def merge_tile_results(
        self,
        tile_results: Dict[str, List[Detection]],  # tile_id -> detections
        tiles: List[Tile],
        source_image: str,
        image_size: Tuple[int, int],
        geo_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> MergedResult:
        """
        Merge detection results from all tiles.
        
        ACCURATE: Handles boundary duplicates with NMS.
        GEO-ALIGNED: Assigns geographic coordinates to detections.
        
        Args:
            tile_results: Detection results per tile
            tiles: Tile metadata
            source_image: Path to source image
            image_size: (width, height) of source image
            geo_bounds: Geographic bounds of source image
            
        Returns:
            MergedResult with all detections in source image coordinates
        """
        # Create tile lookup
        tile_map = {t.tile_id: t for t in tiles}
        
        # Transform all detections to source image coordinates
        all_detections: List[Detection] = []
        
        for tile_id, detections in tile_results.items():
            tile = tile_map.get(tile_id)
            if not tile:
                continue
            
            for det in detections:
                # Transform bbox to source image coordinates
                x1 = det.bbox[0] + tile.x_offset
                y1 = det.bbox[1] + tile.y_offset
                x2 = det.bbox[2] + tile.x_offset
                y2 = det.bbox[3] + tile.y_offset
                
                # Calculate geo coordinates
                geo_bbox = None
                if geo_bounds:
                    geo_bbox = self._pixel_to_geo(
                        (x1, y1, x2, y2),
                        image_size,
                        geo_bounds
                    )
                
                transformed = Detection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=(x1, y1, x2, y2),
                    mask=det.mask,
                    geo_bbox=geo_bbox
                )
                
                all_detections.append(transformed)
        
        # Apply NMS to remove duplicates at tile boundaries
        merged_detections = self._apply_nms(all_detections)
        
        return MergedResult(
            source_image=source_image,
            detections=merged_detections,
            tile_count=len(tiles),
            image_size=image_size,
            geo_bounds=geo_bounds
        )
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Groups by class and removes overlapping boxes.
        """
        if not detections:
            return []
        
        # Group by class
        by_class: Dict[int, List[Detection]] = {}
        for det in detections:
            if det.class_id not in by_class:
                by_class[det.class_id] = []
            by_class[det.class_id].append(det)
        
        result: List[Detection] = []
        
        for class_id, class_dets in by_class.items():
            # Sort by confidence (descending)
            class_dets.sort(key=lambda x: x.confidence, reverse=True)
            
            keep = []
            while class_dets:
                best = class_dets.pop(0)
                keep.append(best)
                
                # Remove overlapping detections
                class_dets = [
                    d for d in class_dets
                    if self._iou(best.bbox, d.bbox) < self.nms_threshold
                ]
            
            result.extend(keep)
        
        return result
    
    def _iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _pixel_to_geo(
        self,
        pixel_bbox: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        geo_bounds: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Convert pixel coordinates to geographic coordinates."""
        x1, y1, x2, y2 = pixel_bbox
        width, height = image_size
        min_lon, min_lat, max_lon, max_lat = geo_bounds
        
        lon_per_pixel = (max_lon - min_lon) / width
        lat_per_pixel = (max_lat - min_lat) / height
        
        geo_x1 = min_lon + (x1 * lon_per_pixel)
        geo_x2 = min_lon + (x2 * lon_per_pixel)
        geo_y2 = max_lat - (y1 * lat_per_pixel)  # Note: y-axis inverted
        geo_y1 = max_lat - (y2 * lat_per_pixel)
        
        return (geo_x1, geo_y1, geo_x2, geo_y2)


class GeoOverlay:
    """
    Overlay predictions on geographic maps.
    
    Converts detection results to GeoJSON for visualization
    and integration with mapping tools.
    """
    
    def detections_to_geojson(
        self,
        detections: List[Detection],
        properties: Optional[Dict] = None
    ) -> Dict:
        """
        Convert detections to GeoJSON format.
        
        Only includes detections with geo_bbox set.
        """
        features = []
        
        for det in detections:
            if not det.geo_bbox:
                continue
            
            min_lon, min_lat, max_lon, max_lat = det.geo_bbox
            
            # Create polygon from bbox
            coordinates = [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat]  # Close polygon
            ]]
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                },
                "properties": {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    **(properties or {})
                }
            }
            
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def tiles_to_geojson(self, tiles: List[Tile]) -> Dict:
        """Convert tile boundaries to GeoJSON for visualization."""
        features = []
        
        for tile in tiles:
            if not tile.geo_bounds:
                continue
            
            min_lon, min_lat, max_lon, max_lat = tile.geo_bounds
            
            coordinates = [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat]
            ]]
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                },
                "properties": {
                    "tile_id": tile.tile_id,
                    "row": tile.row,
                    "col": tile.col
                }
            }
            
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def merge_geojson(self, *geojson_collections: Dict) -> Dict:
        """Merge multiple GeoJSON feature collections."""
        all_features = []
        
        for collection in geojson_collections:
            if collection and "features" in collection:
                all_features.extend(collection["features"])
        
        return {
            "type": "FeatureCollection",
            "features": all_features
        }


# Convenience instances
tile_manager = TileManager()
result_merger = ResultMerger()
geo_overlay = GeoOverlay()
