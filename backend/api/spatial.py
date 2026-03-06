"""
Spatial Data API endpoints for Singularity Vision.

Enhanced with R-tree spatial index for efficient geo-queries.
Supports:
- Deterministic geo-querying
- Version-safe spatial operations
- Profile-aware constraint enforcement
- Tile management for large images
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os

router = APIRouter(prefix="/spatial", tags=["spatial"])


# Request/Response Models
class BBoxRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class SpatialQueryRequest(BaseModel):
    project_path: str
    bbox: Optional[BBoxRequest] = None
    point: Optional[Tuple[float, float]] = None
    radius_meters: Optional[float] = None
    item_types: Optional[List[str]] = None
    dataset_version: Optional[str] = None
    limit: int = 100


class IndexImageRequest(BaseModel):
    project_path: str
    image_path: str
    bbox: BBoxRequest
    dataset_version: Optional[str] = None
    metadata: Optional[Dict] = None


class TileRequest(BaseModel):
    image_path: str
    output_dir: str
    tile_size: int = 1024
    overlap: int = 128
    geo_bounds: Optional[List[float]] = None


class RebuildIndexRequest(BaseModel):
    project_path: str
    dataset_version: Optional[str] = None


class ExifGpsRequest(BaseModel):
    image_path: str


# Cached spatial indices
_spatial_indices: Dict[str, Any] = {}


def _get_spatial_index(project_path: str, profile_id: str = "general"):
    """Get or create spatial index for project."""
    from core.spatial_index import ProjectSpatialIndex
    
    cache_key = f"{project_path}:{profile_id}"
    
    if cache_key not in _spatial_indices:
        _spatial_indices[cache_key] = ProjectSpatialIndex(project_path, profile_id)
    
    return _spatial_indices[cache_key]


@router.post("/query")
async def query_spatial(request: SpatialQueryRequest):
    """
    Query spatial index for items in a region.
    
    DETERMINISTIC: Same query always returns same results.
    VERSION-SAFE: Can query specific dataset versions.
    """
    try:
        index = _get_spatial_index(request.project_path)
        
        from core.spatial_index import BoundingBox, SpatialQuery
        
        bbox = None
        if request.bbox:
            bbox = BoundingBox(
                min_lon=request.bbox.min_lon,
                min_lat=request.bbox.min_lat,
                max_lon=request.bbox.max_lon,
                max_lat=request.bbox.max_lat
            )
        
        query = SpatialQuery(
            bbox=bbox,
            point=request.point,
            radius_meters=request.radius_meters,
            item_types=request.item_types,
            dataset_version=request.dataset_version,
            limit=request.limit
        )
        
        results = index.index.query(query)
        
        return {
            "success": True,
            "count": len(results),
            "items": [r.to_dict() for r in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/region")
async def query_region(request: SpatialQueryRequest):
    """Query items within a bounding box region."""
    try:
        index = _get_spatial_index(request.project_path)
        
        if not request.bbox:
            raise HTTPException(status_code=400, detail="bbox is required for region query")
        
        from core.spatial_index import BoundingBox
        
        bbox = BoundingBox(
            min_lon=request.bbox.min_lon,
            min_lat=request.bbox.min_lat,
            max_lon=request.bbox.max_lon,
            max_lat=request.bbox.max_lat
        )
        
        results = index.query_region(
            bbox=bbox,
            item_types=request.item_types,
            dataset_version=request.dataset_version,
            limit=request.limit
        )
        
        return {
            "success": True,
            "count": len(results),
            "items": [r.to_dict() for r in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/point")
async def query_point(
    project_path: str,
    lon: float,
    lat: float,
    item_types: Optional[str] = None,
    limit: int = 100
):
    """Query items containing a specific point."""
    try:
        index = _get_spatial_index(project_path)
        
        types = item_types.split(",") if item_types else None
        
        results = index.query_point(
            lon=lon,
            lat=lat,
            item_types=types,
            limit=limit
        )
        
        return {
            "success": True,
            "count": len(results),
            "items": [r.to_dict() for r in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nearest")
async def query_nearest(
    project_path: str,
    lon: float,
    lat: float,
    n: int = 5
):
    """Find nearest images to a point."""
    try:
        index = _get_spatial_index(project_path)
        
        results = index.nearest_images(lon=lon, lat=lat, n=n)
        
        return {
            "success": True,
            "count": len(results),
            "items": [r.to_dict() for r in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/image")
async def index_image(request: IndexImageRequest):
    """Add an image to the spatial index."""
    try:
        index = _get_spatial_index(request.project_path)
        
        from core.spatial_index import BoundingBox
        
        bbox = BoundingBox(
            min_lon=request.bbox.min_lon,
            min_lat=request.bbox.min_lat,
            max_lon=request.bbox.max_lon,
            max_lat=request.bbox.max_lat
        )
        
        success = index.index_image(
            image_path=request.image_path,
            bbox=bbox,
            dataset_version=request.dataset_version,
            metadata=request.metadata
        )
        
        return {"success": success}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/rebuild")
async def rebuild_index(request: RebuildIndexRequest):
    """Rebuild the spatial index from project images."""
    try:
        index = _get_spatial_index(request.project_path)
        
        count = index.rebuild_index(request.dataset_version)
        
        return {
            "success": True,
            "indexed_count": count,
            "dataset_version": request.dataset_version
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats(project_path: str):
    """Get spatial index statistics."""
    try:
        index = _get_spatial_index(project_path)
        
        stats = index.get_stats()
        coverage = index.get_coverage()
        
        return {
            "success": True,
            "stats": stats,
            "coverage": coverage.to_dict() if coverage else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Tile Management Endpoints

@router.post("/tiles/create")
async def create_tiles(request: TileRequest):
    """Create tiles from a large image."""
    try:
        from core.tile_manager import TileManager, TileConfig
        
        config = TileConfig(
            tile_size=request.tile_size,
            overlap=request.overlap
        )
        
        manager = TileManager(config)
        
        geo_bounds = tuple(request.geo_bounds) if request.geo_bounds else None
        
        tiles = manager.create_tiles(
            image_path=request.image_path,
            output_dir=request.output_dir,
            geo_bounds=geo_bounds
        )
        
        return {
            "success": True,
            "tile_count": len(tiles),
            "tiles": [t.to_dict() for t in tiles]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tiles/needs-tiling")
async def check_needs_tiling(image_path: str, max_dimension: int = 4096):
    """Check if an image needs to be tiled."""
    try:
        from core.tile_manager import tile_manager
        
        needs = tile_manager.needs_tiling(image_path, max_dimension)
        
        return {
            "needs_tiling": needs,
            "max_dimension": max_dimension
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GeoJSON Export Endpoints

@router.get("/export/geojson")
async def export_geojson(
    project_path: str,
    item_types: Optional[str] = None,
    dataset_version: Optional[str] = None
):
    """Export spatial items as GeoJSON FeatureCollection."""
    try:
        index = _get_spatial_index(project_path)
        
        from core.spatial_index import SpatialQuery
        
        types = item_types.split(",") if item_types else None
        
        # Get all items (no bbox filter)
        query = SpatialQuery(
            item_types=types,
            dataset_version=dataset_version,
            limit=10000
        )
        
        items = index.index.query(query)
        
        # Convert to GeoJSON
        features = []
        for item in items:
            if item.bbox:
                min_lon = item.bbox.min_lon
                min_lat = item.bbox.min_lat
                max_lon = item.bbox.max_lon
                max_lat = item.bbox.max_lat
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat]
                        ]]
                    },
                    "properties": {
                        "item_id": item.item_id,
                        "item_type": item.item_type,
                        "dataset_version": item.dataset_version,
                        **item.metadata
                    }
                }
                features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Legacy endpoints for backward compatibility

@router.post("/extract-gps")
async def extract_gps(request: ExifGpsRequest):
    """Extract GPS coordinates from an image's EXIF data."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        img = Image.open(request.image_path)
        exif = img._getexif()
        
        if not exif:
            return {"success": False, "message": "No EXIF data found"}
        
        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value
        
        if not gps_info:
            return {"success": False, "message": "No GPS data in EXIF"}
        
        # Convert to decimal degrees
        def convert_to_decimal(coords, ref):
            if not coords:
                return None
            degrees = float(coords[0])
            minutes = float(coords[1])
            seconds = float(coords[2])
            decimal = degrees + (minutes / 60) + (seconds / 3600)
            if ref in ["S", "W"]:
                decimal = -decimal
            return decimal
        
        lat = convert_to_decimal(
            gps_info.get("GPSLatitude"),
            gps_info.get("GPSLatitudeRef", "N")
        )
        lon = convert_to_decimal(
            gps_info.get("GPSLongitude"),
            gps_info.get("GPSLongitudeRef", "E")
        )
        
        if lat and lon:
            return {
                "success": True,
                "coordinate": {"latitude": lat, "longitude": lon}
            }
        else:
            return {"success": False, "message": "Could not parse GPS coordinates"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
