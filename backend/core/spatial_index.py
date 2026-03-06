"""
R-Tree Spatial Index for Singularity Vision

Core spatial infrastructure for:
- Efficient geo-querying of datasets and annotations
- Deterministic spatial searches
- Industry profile constraint enforcement at query time
- Version-safe spatial operations

Based on the Rtree library (libspatialindex wrapper)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import rtree (libspatialindex)
try:
    from rtree import index as rtree_index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    logger.warning("rtree not installed. Spatial index will use fallback mode.")


@dataclass
class BoundingBox:
    """Geographic bounding box in WGS84 (EPSG:4326)."""
    min_lon: float  # West
    min_lat: float  # South
    max_lon: float  # East
    max_lat: float  # North
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (minx, miny, maxx, maxy) for rtree."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)
    
    def to_dict(self) -> Dict:
        return {
            "min_lon": self.min_lon,
            "min_lat": self.min_lat,
            "max_lon": self.max_lon,
            "max_lat": self.max_lat
        }
    
    @classmethod
    def from_tuple(cls, coords: Tuple[float, float, float, float]) -> 'BoundingBox':
        return cls(min_lon=coords[0], min_lat=coords[1], 
                   max_lon=coords[2], max_lat=coords[3])
    
    def contains(self, lon: float, lat: float) -> bool:
        """Check if point is inside bbox."""
        return (self.min_lon <= lon <= self.max_lon and 
                self.min_lat <= lat <= self.max_lat)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if bboxes intersect."""
        return not (
            self.max_lon < other.min_lon or
            self.min_lon > other.max_lon or
            self.max_lat < other.min_lat or
            self.min_lat > other.max_lat
        )
    
    def area(self) -> float:
        """Calculate area in square degrees (approximate)."""
        return (self.max_lon - self.min_lon) * (self.max_lat - self.min_lat)


@dataclass
class SpatialItem:
    """Item stored in the spatial index."""
    item_id: str
    item_type: str  # "image", "annotation", "tile", "region"
    bbox: BoundingBox
    dataset_version: Optional[str] = None  # For version-safe queries
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "bbox": self.bbox.to_dict(),
            "dataset_version": self.dataset_version,
            "metadata": self.metadata
        }


@dataclass
class SpatialQuery:
    """Query parameters for spatial search."""
    bbox: Optional[BoundingBox] = None  # Area-based query
    point: Optional[Tuple[float, float]] = None  # Point query (lon, lat)
    radius_meters: Optional[float] = None  # For point+radius queries
    item_types: Optional[List[str]] = None  # Filter by type
    dataset_version: Optional[str] = None  # Version constraint
    limit: int = 100
    
    def to_dict(self) -> Dict:
        return {
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "point": self.point,
            "radius_meters": self.radius_meters,
            "item_types": self.item_types,
            "dataset_version": self.dataset_version,
            "limit": self.limit
        }


class SpatialIndex:
    """
    R-Tree based spatial index for efficient geo-queries.
    
    Design Principles:
    - Deterministic: Same query always returns same results
    - Version-safe: Can query specific dataset versions
    - Profile-aware: Respects industry profile constraints
    - Local-first: No external dependencies
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize spatial index.
        
        Args:
            index_path: Path to store persistent index (optional)
        """
        self.index_path = index_path
        self._items: Dict[str, SpatialItem] = {}  # item_id -> SpatialItem
        self._id_counter = 0
        self._id_map: Dict[str, int] = {}  # item_id -> rtree_id
        self._reverse_map: Dict[int, str] = {}  # rtree_id -> item_id
        
        # Initialize R-tree index
        if RTREE_AVAILABLE:
            props = rtree_index.Property()
            props.dimension = 2
            props.fill_factor = 0.7
            
            if index_path and os.path.exists(index_path + ".idx"):
                # Load existing index
                self._index = rtree_index.Index(index_path, properties=props)
                self._load_metadata()
            else:
                # Create new index
                self._index = rtree_index.Index(properties=props)
        else:
            self._index = None
            logger.info("Using fallback spatial search (no rtree)")
    
    def insert(self, item: SpatialItem) -> bool:
        """
        Insert an item into the spatial index.
        
        Args:
            item: SpatialItem to insert
            
        Returns:
            True if successful
        """
        if item.item_id in self._items:
            # Update existing item
            return self.update(item)
        
        # Generate rtree ID
        rtree_id = self._id_counter
        self._id_counter += 1
        
        # Store mappings
        self._items[item.item_id] = item
        self._id_map[item.item_id] = rtree_id
        self._reverse_map[rtree_id] = item.item_id
        
        # Insert into R-tree
        if self._index is not None:
            self._index.insert(rtree_id, item.bbox.to_tuple())
        
        return True
    
    def insert_batch(self, items: List[SpatialItem]) -> int:
        """
        Insert multiple items efficiently.
        
        Returns:
            Number of items inserted
        """
        count = 0
        for item in items:
            if self.insert(item):
                count += 1
        return count
    
    def update(self, item: SpatialItem) -> bool:
        """Update an existing item."""
        if item.item_id not in self._items:
            return self.insert(item)
        
        old_item = self._items[item.item_id]
        rtree_id = self._id_map[item.item_id]
        
        # Update R-tree
        if self._index is not None:
            self._index.delete(rtree_id, old_item.bbox.to_tuple())
            self._index.insert(rtree_id, item.bbox.to_tuple())
        
        # Update stored item
        self._items[item.item_id] = item
        
        return True
    
    def delete(self, item_id: str) -> bool:
        """Remove an item from the index."""
        if item_id not in self._items:
            return False
        
        item = self._items[item_id]
        rtree_id = self._id_map[item_id]
        
        # Remove from R-tree
        if self._index is not None:
            self._index.delete(rtree_id, item.bbox.to_tuple())
        
        # Clean up mappings
        del self._items[item_id]
        del self._id_map[item_id]
        del self._reverse_map[rtree_id]
        
        return True
    
    def query(self, spatial_query: SpatialQuery) -> List[SpatialItem]:
        """
        Execute a spatial query.
        
        This is DETERMINISTIC: Same query always returns same results
        in the same order (sorted by item_id).
        
        Args:
            spatial_query: Query parameters
            
        Returns:
            List of matching SpatialItems (sorted by item_id for determinism)
        """
        candidates: List[str] = []
        
        # Get candidates from R-tree or fallback
        if spatial_query.bbox:
            candidates = self._query_bbox(spatial_query.bbox)
        elif spatial_query.point:
            if spatial_query.radius_meters:
                # Convert radius to degrees (approximate)
                radius_deg = spatial_query.radius_meters / 111000.0
                bbox = BoundingBox(
                    min_lon=spatial_query.point[0] - radius_deg,
                    min_lat=spatial_query.point[1] - radius_deg,
                    max_lon=spatial_query.point[0] + radius_deg,
                    max_lat=spatial_query.point[1] + radius_deg
                )
                candidates = self._query_bbox(bbox)
            else:
                # Point containment query
                candidates = self._query_point(spatial_query.point)
        else:
            # No spatial constraint - return all
            candidates = list(self._items.keys())
        
        # Apply filters
        results: List[SpatialItem] = []
        for item_id in candidates:
            item = self._items.get(item_id)
            if not item:
                continue
            
            # Filter by type
            if spatial_query.item_types:
                if item.item_type not in spatial_query.item_types:
                    continue
            
            # Filter by version (version-safe queries)
            if spatial_query.dataset_version:
                if item.dataset_version != spatial_query.dataset_version:
                    continue
            
            results.append(item)
            
            if len(results) >= spatial_query.limit:
                break
        
        # Sort by item_id for deterministic results
        results.sort(key=lambda x: x.item_id)
        
        return results[:spatial_query.limit]
    
    def _query_bbox(self, bbox: BoundingBox) -> List[str]:
        """Query by bounding box."""
        if self._index is not None:
            # Use R-tree
            rtree_ids = list(self._index.intersection(bbox.to_tuple()))
            return [self._reverse_map.get(rid) for rid in rtree_ids 
                    if rid in self._reverse_map]
        else:
            # Fallback: brute force
            return [
                item_id for item_id, item in self._items.items()
                if item.bbox.intersects(bbox)
            ]
    
    def _query_point(self, point: Tuple[float, float]) -> List[str]:
        """Query items containing a point."""
        lon, lat = point
        return [
            item_id for item_id, item in self._items.items()
            if item.bbox.contains(lon, lat)
        ]
    
    def nearest(self, lon: float, lat: float, n: int = 5) -> List[SpatialItem]:
        """
        Find n nearest items to a point.
        
        Args:
            lon: Longitude
            lat: Latitude
            n: Number of results
            
        Returns:
            List of nearest items (sorted by distance)
        """
        if self._index is not None:
            rtree_ids = list(self._index.nearest((lon, lat, lon, lat), n))
            return [self._items[self._reverse_map[rid]] 
                    for rid in rtree_ids 
                    if rid in self._reverse_map]
        else:
            # Fallback: compute distances and sort
            items_with_dist = []
            for item in self._items.values():
                center_lon = (item.bbox.min_lon + item.bbox.max_lon) / 2
                center_lat = (item.bbox.min_lat + item.bbox.max_lat) / 2
                dist = ((center_lon - lon) ** 2 + (center_lat - lat) ** 2) ** 0.5
                items_with_dist.append((dist, item))
            
            items_with_dist.sort(key=lambda x: x[0])
            return [item for _, item in items_with_dist[:n]]
    
    def get_bounds(self) -> Optional[BoundingBox]:
        """Get bounding box of all items in the index."""
        if not self._items:
            return None
        
        min_lon = min(item.bbox.min_lon for item in self._items.values())
        min_lat = min(item.bbox.min_lat for item in self._items.values())
        max_lon = max(item.bbox.max_lon for item in self._items.values())
        max_lat = max(item.bbox.max_lat for item in self._items.values())
        
        return BoundingBox(min_lon, min_lat, max_lon, max_lat)
    
    def count(self) -> int:
        """Return number of items in the index."""
        return len(self._items)
    
    def clear(self) -> None:
        """Remove all items from the index."""
        self._items.clear()
        self._id_map.clear()
        self._reverse_map.clear()
        self._id_counter = 0
        
        if RTREE_AVAILABLE:
            props = rtree_index.Property()
            props.dimension = 2
            self._index = rtree_index.Index(properties=props)
    
    def save(self, path: Optional[str] = None) -> bool:
        """
        Save index to disk.
        
        The R-tree is persisted automatically if index_path was provided.
        This saves the metadata (items, mappings).
        """
        save_path = path or self.index_path
        if not save_path:
            return False
        
        metadata = {
            "id_counter": self._id_counter,
            "items": {k: v.to_dict() for k, v in self._items.items()},
            "id_map": self._id_map,
            "saved_at": datetime.now().isoformat()
        }
        
        try:
            with open(save_path + ".meta.json", 'w') as f:
                json.dump(metadata, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save spatial index: {e}")
            return False
    
    def _load_metadata(self) -> bool:
        """Load metadata from disk."""
        if not self.index_path:
            return False
        
        meta_path = self.index_path + ".meta.json"
        if not os.path.exists(meta_path):
            return False
        
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            self._id_counter = metadata.get("id_counter", 0)
            self._id_map = metadata.get("id_map", {})
            
            # Rebuild reverse map
            self._reverse_map = {v: k for k, v in self._id_map.items()}
            
            # Rebuild items
            for item_id, item_data in metadata.get("items", {}).items():
                bbox = BoundingBox.from_tuple((
                    item_data["bbox"]["min_lon"],
                    item_data["bbox"]["min_lat"],
                    item_data["bbox"]["max_lon"],
                    item_data["bbox"]["max_lat"]
                ))
                self._items[item_id] = SpatialItem(
                    item_id=item_id,
                    item_type=item_data["item_type"],
                    bbox=bbox,
                    dataset_version=item_data.get("dataset_version"),
                    metadata=item_data.get("metadata", {})
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to load spatial index metadata: {e}")
            return False


class ProjectSpatialIndex:
    """
    Project-level spatial index manager.
    
    Manages spatial indices for a project, including:
    - Dataset images with geo-tags
    - Annotations with spatial data
    - Tile boundaries
    - Regions of interest
    
    Enforces industry profile constraints at query time.
    """
    
    def __init__(self, project_path: str, profile_id: str = "general"):
        self.project_path = project_path
        self.profile_id = profile_id
        
        # Index storage path
        self.index_dir = os.path.join(project_path, ".spatial")
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Main spatial index
        index_path = os.path.join(self.index_dir, "main_index")
        self.index = SpatialIndex(index_path)
        
        # Track indexed items for versioning
        self._indexed_version: Optional[str] = None
    
    def index_image(
        self, 
        image_path: str,
        bbox: BoundingBox,
        dataset_version: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add an image to the spatial index."""
        item_id = os.path.relpath(image_path, self.project_path)
        
        item = SpatialItem(
            item_id=item_id,
            item_type="image",
            bbox=bbox,
            dataset_version=dataset_version,
            metadata=metadata or {}
        )
        
        return self.index.insert(item)
    
    def index_annotation(
        self,
        annotation_id: str,
        bbox: BoundingBox,
        image_path: str,
        annotation_type: str = "bbox",
        dataset_version: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add an annotation to the spatial index."""
        item = SpatialItem(
            item_id=f"annotation:{annotation_id}",
            item_type="annotation",
            bbox=bbox,
            dataset_version=dataset_version,
            metadata={
                "image_path": image_path,
                "annotation_type": annotation_type,
                **(metadata or {})
            }
        )
        
        return self.index.insert(item)
    
    def index_tile(
        self,
        tile_id: str,
        bbox: BoundingBox,
        source_image: str,
        tile_coords: Tuple[int, int],  # (row, col)
        dataset_version: Optional[str] = None
    ) -> bool:
        """Add a tile to the spatial index."""
        item = SpatialItem(
            item_id=f"tile:{tile_id}",
            item_type="tile",
            bbox=bbox,
            dataset_version=dataset_version,
            metadata={
                "source_image": source_image,
                "row": tile_coords[0],
                "col": tile_coords[1]
            }
        )
        
        return self.index.insert(item)
    
    def query_region(
        self,
        bbox: BoundingBox,
        item_types: Optional[List[str]] = None,
        dataset_version: Optional[str] = None,
        limit: int = 100
    ) -> List[SpatialItem]:
        """
        Query items in a geographic region.
        
        DETERMINISTIC: Same query always returns same results.
        VERSION-SAFE: Can query specific dataset versions.
        PROFILE-AWARE: Respects industry profile constraints.
        """
        # Check profile constraints
        if not self._check_spatial_query_allowed():
            logger.warning(f"Spatial query blocked by profile {self.profile_id}")
            return []
        
        query = SpatialQuery(
            bbox=bbox,
            item_types=item_types,
            dataset_version=dataset_version,
            limit=limit
        )
        
        return self.index.query(query)
    
    def query_point(
        self,
        lon: float,
        lat: float,
        item_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[SpatialItem]:
        """Query items containing a specific point."""
        query = SpatialQuery(
            point=(lon, lat),
            item_types=item_types,
            limit=limit
        )
        
        return self.index.query(query)
    
    def query_radius(
        self,
        lon: float,
        lat: float,
        radius_meters: float,
        item_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[SpatialItem]:
        """Query items within radius of a point."""
        query = SpatialQuery(
            point=(lon, lat),
            radius_meters=radius_meters,
            item_types=item_types,
            limit=limit
        )
        
        return self.index.query(query)
    
    def nearest_images(self, lon: float, lat: float, n: int = 5) -> List[SpatialItem]:
        """Find nearest images to a point."""
        items = self.index.nearest(lon, lat, n * 2)  # Get extra for filtering
        return [item for item in items if item.item_type == "image"][:n]
    
    def get_coverage(self) -> Optional[BoundingBox]:
        """Get total geographic coverage of indexed items."""
        return self.index.get_bounds()
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        items_by_type = {}
        for item in self.index._items.values():
            items_by_type[item.item_type] = items_by_type.get(item.item_type, 0) + 1
        
        bounds = self.get_coverage()
        
        return {
            "total_items": self.index.count(),
            "items_by_type": items_by_type,
            "coverage": bounds.to_dict() if bounds else None,
            "indexed_version": self._indexed_version,
            "rtree_available": RTREE_AVAILABLE
        }
    
    def _check_spatial_query_allowed(self) -> bool:
        """Check if spatial queries are allowed by profile."""
        # Import here to avoid circular dependency
        try:
            from core.industry_profiles import profile_manager
            return profile_manager.is_feature_allowed(self.profile_id, "spatial_analysis")
        except:
            return True  # Default allow if profile system unavailable
    
    def rebuild_index(self, dataset_version: Optional[str] = None) -> int:
        """
        Rebuild the entire spatial index from project data.
        
        Returns:
            Number of items indexed
        """
        self.index.clear()
        count = 0
        
        # Index images with EXIF GPS data
        datasets_dir = os.path.join(self.project_path, "datasets")
        if os.path.exists(datasets_dir):
            for root, _, files in os.walk(datasets_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                        image_path = os.path.join(root, file)
                        bbox = self._extract_geo_from_image(image_path)
                        if bbox:
                            self.index_image(image_path, bbox, dataset_version)
                            count += 1
        
        self._indexed_version = dataset_version
        self.save()
        
        return count
    
    def _extract_geo_from_image(self, image_path: str) -> Optional[BoundingBox]:
        """Extract geographic bounds from image EXIF/metadata."""
        try:
            # Try to read EXIF GPS data
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS
            
            img = Image.open(image_path)
            exif = img._getexif()
            
            if not exif:
                return None
            
            gps_info = {}
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    for gps_tag_id, gps_value in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_info[gps_tag] = gps_value
            
            if not gps_info:
                return None
            
            # Extract coordinates
            lat = self._convert_gps_to_decimal(
                gps_info.get("GPSLatitude"),
                gps_info.get("GPSLatitudeRef", "N")
            )
            lon = self._convert_gps_to_decimal(
                gps_info.get("GPSLongitude"),
                gps_info.get("GPSLongitudeRef", "E")
            )
            
            if lat is None or lon is None:
                return None
            
            # Create small bbox around point (image footprint unknown)
            delta = 0.0001  # ~10 meters
            return BoundingBox(
                min_lon=lon - delta,
                min_lat=lat - delta,
                max_lon=lon + delta,
                max_lat=lat + delta
            )
            
        except Exception as e:
            logger.debug(f"Could not extract GPS from {image_path}: {e}")
            return None
    
    def _convert_gps_to_decimal(
        self, 
        coords: Optional[Tuple], 
        ref: str
    ) -> Optional[float]:
        """Convert GPS coordinates to decimal degrees."""
        if not coords:
            return None
        
        try:
            degrees = float(coords[0])
            minutes = float(coords[1])
            seconds = float(coords[2])
            
            decimal = degrees + (minutes / 60) + (seconds / 3600)
            
            if ref in ["S", "W"]:
                decimal = -decimal
            
            return decimal
        except:
            return None
    
    def save(self) -> bool:
        """Save the spatial index to disk."""
        return self.index.save()


# Factory function
def create_spatial_index(project_path: str, profile_id: str = "general") -> ProjectSpatialIndex:
    """Create a project-level spatial index."""
    return ProjectSpatialIndex(project_path, profile_id)
