"""
Spatial Engine for Singularity Vision
Handles geo-aware data processing, coordinate extraction, and spatial analysis.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GeoCoordinate:
    """Represents a geographic coordinate with optional altitude and timestamp."""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class GeoMetadata:
    """Geo-metadata extracted from an image or data file."""
    source_file: str
    coordinate: Optional[GeoCoordinate]
    crs: Optional[str] = None  # Coordinate Reference System (e.g., "EPSG:4326")
    bounds: Optional[Tuple[float, float, float, float]] = None  # minx, miny, maxx, maxy
    
    def to_dict(self) -> Dict:
        return {
            "source_file": self.source_file,
            "coordinate": self.coordinate.to_dict() if self.coordinate else None,
            "crs": self.crs,
            "bounds": self.bounds
        }


class SpatialEngine:
    """
    Core spatial processing engine for geo-aware computer vision.
    
    Capabilities:
    - EXIF GPS extraction from images
    - GeoTIFF reading and tiling
    - Coordinate transformations
    - Spatial indexing for datasets
    """
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.spatial_index: Dict[str, GeoMetadata] = {}
        
    def extract_exif_gps(self, image_path: str) -> Optional[GeoCoordinate]:
        """
        Extract GPS coordinates from image EXIF metadata.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            GeoCoordinate if GPS data found, None otherwise
        """
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS
            
            img = Image.open(image_path)
            exif_data = img._getexif()
            
            if not exif_data:
                return None
                
            gps_info = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    for gps_tag_id, gps_value in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_info[gps_tag] = gps_value
                        
            if not gps_info:
                return None
                
            # Extract latitude
            lat = self._convert_to_degrees(gps_info.get("GPSLatitude"))
            lat_ref = gps_info.get("GPSLatitudeRef", "N")
            if lat_ref == "S":
                lat = -lat
                
            # Extract longitude
            lon = self._convert_to_degrees(gps_info.get("GPSLongitude"))
            lon_ref = gps_info.get("GPSLongitudeRef", "E")
            if lon_ref == "W":
                lon = -lon
                
            # Extract altitude (optional)
            alt = None
            if "GPSAltitude" in gps_info:
                alt = float(gps_info["GPSAltitude"])
                if gps_info.get("GPSAltitudeRef", 0) == 1:
                    alt = -alt
                    
            # Extract timestamp (optional)
            timestamp = None
            if "GPSDateStamp" in gps_info and "GPSTimeStamp" in gps_info:
                date_str = gps_info["GPSDateStamp"]
                time_tuple = gps_info["GPSTimeStamp"]
                try:
                    hour, minute, second = [int(float(x)) for x in time_tuple]
                    timestamp = datetime.strptime(
                        f"{date_str} {hour:02d}:{minute:02d}:{second:02d}",
                        "%Y:%m:%d %H:%M:%S"
                    )
                except (ValueError, TypeError):
                    pass
                    
            return GeoCoordinate(
                latitude=lat,
                longitude=lon,
                altitude=alt,
                timestamp=timestamp
            )
            
        except Exception as e:
            print(f"Failed to extract EXIF GPS from {image_path}: {e}")
            return None
            
    def _convert_to_degrees(self, value) -> float:
        """Convert GPS coordinates from DMS to decimal degrees."""
        if value is None:
            return 0.0
        try:
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
        except (TypeError, IndexError):
            return 0.0
            
    def read_geotiff_metadata(self, tiff_path: str) -> Optional[GeoMetadata]:
        """
        Read metadata from a GeoTIFF file.
        
        Args:
            tiff_path: Path to the GeoTIFF file
            
        Returns:
            GeoMetadata with bounds and CRS information
        """
        try:
            import rasterio
            
            with rasterio.open(tiff_path) as dataset:
                bounds = dataset.bounds
                crs = str(dataset.crs) if dataset.crs else None
                
                # Get center coordinate
                center_x = (bounds.left + bounds.right) / 2
                center_y = (bounds.bottom + bounds.top) / 2
                
                coord = GeoCoordinate(
                    latitude=center_y,
                    longitude=center_x
                )
                
                return GeoMetadata(
                    source_file=tiff_path,
                    coordinate=coord,
                    crs=crs,
                    bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top)
                )
                
        except ImportError:
            print("rasterio not installed. Install with: pip install rasterio")
            return None
        except Exception as e:
            print(f"Failed to read GeoTIFF {tiff_path}: {e}")
            return None
            
    def tile_large_image(
        self,
        image_path: str,
        output_dir: str,
        tile_size: int = 512,
        overlap: int = 64
    ) -> List[Dict]:
        """
        Tile a large image (e.g., satellite/drone imagery) into smaller tiles.
        
        Args:
            image_path: Path to the source image
            output_dir: Directory to save tiles
            tile_size: Size of each tile in pixels
            overlap: Overlap between tiles in pixels
            
        Returns:
            List of tile metadata dicts with bounds and file paths
        """
        try:
            from PIL import Image
            import math
            
            os.makedirs(output_dir, exist_ok=True)
            
            img = Image.open(image_path)
            width, height = img.size
            
            tiles = []
            step = tile_size - overlap
            
            num_x = math.ceil((width - overlap) / step)
            num_y = math.ceil((height - overlap) / step)
            
            for y in range(num_y):
                for x in range(num_x):
                    x1 = x * step
                    y1 = y * step
                    x2 = min(x1 + tile_size, width)
                    y2 = min(y1 + tile_size, height)
                    
                    tile = img.crop((x1, y1, x2, y2))
                    tile_name = f"tile_{y:04d}_{x:04d}.png"
                    tile_path = os.path.join(output_dir, tile_name)
                    tile.save(tile_path)
                    
                    tiles.append({
                        "path": tile_path,
                        "x": x,
                        "y": y,
                        "bounds": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }
                    })
                    
            return tiles
            
        except Exception as e:
            print(f"Failed to tile image {image_path}: {e}")
            return []
            
    def index_dataset_spatial(self, images_dir: str) -> Dict[str, GeoMetadata]:
        """
        Build a spatial index for all geo-tagged images in a directory.
        
        Args:
            images_dir: Directory containing images
            
        Returns:
            Dict mapping image paths to their GeoMetadata
        """
        supported_extensions = {'.jpg', '.jpeg', '.tif', '.tiff', '.png'}
        
        for root, _, files in os.walk(images_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in supported_extensions:
                    continue
                    
                file_path = os.path.join(root, file)
                
                # Try EXIF first
                coord = self.extract_exif_gps(file_path)
                if coord:
                    self.spatial_index[file_path] = GeoMetadata(
                        source_file=file_path,
                        coordinate=coord
                    )
                    continue
                    
                # Try GeoTIFF for .tif files
                if ext in {'.tif', '.tiff'}:
                    geo_meta = self.read_geotiff_metadata(file_path)
                    if geo_meta:
                        self.spatial_index[file_path] = geo_meta
                        
        return self.spatial_index
        
    def get_images_in_bounds(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float
    ) -> List[str]:
        """
        Query images within geographic bounds.
        
        Args:
            min_lat, max_lat: Latitude range
            min_lon, max_lon: Longitude range
            
        Returns:
            List of image paths within the bounds
        """
        results = []
        
        for path, meta in self.spatial_index.items():
            if meta.coordinate:
                lat = meta.coordinate.latitude
                lon = meta.coordinate.longitude
                
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    results.append(path)
                    
        return results
        
    def export_to_geojson(self, output_path: str) -> str:
        """
        Export the spatial index as GeoJSON.
        
        Args:
            output_path: Path to save the GeoJSON file
            
        Returns:
            Path to the created file
        """
        features = []
        
        for path, meta in self.spatial_index.items():
            if meta.coordinate:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            meta.coordinate.longitude,
                            meta.coordinate.latitude
                        ]
                    },
                    "properties": {
                        "file_path": path,
                        "altitude": meta.coordinate.altitude,
                        "timestamp": meta.coordinate.timestamp.isoformat() if meta.coordinate.timestamp else None
                    }
                }
                features.append(feature)
                
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
            
        return output_path
