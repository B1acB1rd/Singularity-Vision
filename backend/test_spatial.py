"""
Spatial Functionality Tests for Phase A

Tests:
1. R-tree spatial index operations
2. Tile manager for large images  
3. Geo-query determinism
4. Version-safe queries
5. GeoJSON export
"""

import os
import sys
import json
import time
import tempfile
import shutil

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

results = {
    "spatial_index": {"works": [], "slow": [], "breaks": []},
    "tile_manager": {"works": [], "slow": [], "breaks": []},
    "geo_queries": {"works": [], "slow": [], "breaks": []}
}


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_spatial_index():
    """Test R-tree spatial index operations."""
    print_section("TEST 1: R-Tree Spatial Index")
    
    try:
        from core.spatial_index import (
            SpatialIndex, BoundingBox, SpatialItem, SpatialQuery, 
            RTREE_AVAILABLE
        )
        
        # Test 1a: Check R-tree availability
        print(f"   R-tree library available: {RTREE_AVAILABLE}")
        results["spatial_index"]["works"].append(
            f"rtree_available: {RTREE_AVAILABLE}"
        )
        
        # Test 1b: Create index
        index = SpatialIndex()
        results["spatial_index"]["works"].append("index_created")
        print(f"✅ SpatialIndex created")
        
        # Test 1c: Insert items
        items = [
            SpatialItem(
                item_id="img_001",
                item_type="image",
                bbox=BoundingBox(-122.5, 37.7, -122.4, 37.8),
                dataset_version="v1.0",
                metadata={"name": "San Francisco"}
            ),
            SpatialItem(
                item_id="img_002",
                item_type="image",
                bbox=BoundingBox(-118.3, 34.0, -118.2, 34.1),
                dataset_version="v1.0",
                metadata={"name": "Los Angeles"}
            ),
            SpatialItem(
                item_id="anno_001",
                item_type="annotation",
                bbox=BoundingBox(-122.45, 37.75, -122.42, 37.78),
                dataset_version="v1.0"
            )
        ]
        
        inserted = index.insert_batch(items)
        results["spatial_index"]["works"].append(f"insert_batch: {inserted} items")
        print(f"✅ Inserted {inserted} items")
        
        # Test 1d: Query by bounding box
        query_bbox = BoundingBox(-123.0, 37.0, -122.0, 38.0)
        query = SpatialQuery(bbox=query_bbox, limit=10)
        
        start = time.time()
        found = index.query(query)
        elapsed = time.time() - start
        
        results["spatial_index"]["works"].append(f"bbox_query: found {len(found)} items")
        print(f"✅ BBox query: {len(found)} items ({elapsed:.4f}s)")
        
        if elapsed > 0.1:
            results["spatial_index"]["slow"].append(f"bbox_query: {elapsed:.4f}s")
        
        # Test 1e: Query by type filter
        query_by_type = SpatialQuery(
            bbox=query_bbox,
            item_types=["annotation"],
            limit=10
        )
        anno_results = index.query(query_by_type)
        results["spatial_index"]["works"].append(
            f"type_filter: {len(anno_results)} annotations"
        )
        print(f"✅ Type filter: {len(anno_results)} annotations")
        
        # Test 1f: Version-safe query
        query_v1 = SpatialQuery(bbox=query_bbox, dataset_version="v1.0")
        query_v2 = SpatialQuery(bbox=query_bbox, dataset_version="v2.0")
        
        v1_results = index.query(query_v1)
        v2_results = index.query(query_v2)
        
        results["spatial_index"]["works"].append(
            f"version_filter: v1.0={len(v1_results)}, v2.0={len(v2_results)}"
        )
        print(f"✅ Version filter: v1.0={len(v1_results)}, v2.0={len(v2_results)}")
        
        # Test 1g: Nearest neighbor
        nearest = index.nearest(-122.45, 37.75, n=2)
        results["spatial_index"]["works"].append(f"nearest: {len(nearest)} items")
        print(f"✅ Nearest to SF: {[n.item_id for n in nearest]}")
        
        # Test 1h: Determinism - same query = same results
        results1 = index.query(query)
        results2 = index.query(query)
        
        is_deterministic = [r.item_id for r in results1] == [r.item_id for r in results2]
        
        if is_deterministic:
            results["spatial_index"]["works"].append("determinism: verified")
            print(f"✅ Determinism verified (same order)")
        else:
            results["spatial_index"]["breaks"].append("queries not deterministic")
            print(f"❌ Queries not deterministic!")
        
        # Test 1i: Get bounds
        bounds = index.get_bounds()
        results["spatial_index"]["works"].append(
            f"get_bounds: lon={bounds.min_lon:.2f}-{bounds.max_lon:.2f}"
        )
        print(f"✅ Index bounds: {bounds.to_dict()}")
        
    except Exception as e:
        results["spatial_index"]["breaks"].append(f"Exception: {str(e)}")
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()


def test_tile_manager():
    """Test tile management for large images."""
    print_section("TEST 2: Tile Manager")
    
    try:
        from core.tile_manager import (
            TileManager, TileConfig, ResultMerger, GeoOverlay,
            Detection, Tile
        )
        
        # Test 2a: Create config
        config = TileConfig(tile_size=512, overlap=64)
        results["tile_manager"]["works"].append(
            f"config: tile_size={config.tile_size}, overlap={config.overlap}"
        )
        print(f"✅ TileConfig created")
        
        # Test 2b: Initialize manager
        manager = TileManager(config)
        results["tile_manager"]["works"].append("manager_created")
        print(f"✅ TileManager created")
        
        # Test 2c: Test result merger NMS
        merger = ResultMerger(nms_threshold=0.5)
        
        # Create overlapping detections (like at tile boundary)
        detections = [
            Detection(class_id=0, class_name="car", confidence=0.9, 
                      bbox=(100, 100, 200, 200)),
            Detection(class_id=0, class_name="car", confidence=0.85,
                      bbox=(105, 105, 205, 205)),  # Overlapping, lower conf
            Detection(class_id=0, class_name="car", confidence=0.8,
                      bbox=(500, 500, 600, 600)),  # Non-overlapping
        ]
        
        merged = merger._apply_nms(detections)
        
        if len(merged) == 2:  # Should keep 2 non-overlapping
            results["tile_manager"]["works"].append(
                f"nms: 3 inputs → {len(merged)} merged"
            )
            print(f"✅ NMS: 3 detections → {len(merged)} (removed boundary duplicate)")
        else:
            results["tile_manager"]["breaks"].append(f"nms: expected 2, got {len(merged)}")
            print(f"❌ NMS: expected 2, got {len(merged)}")
        
        # Test 2d: IOU calculation
        iou = merger._iou((0, 0, 100, 100), (50, 50, 150, 150))
        expected_iou = 2500 / 17500  # intersection / union
        
        if abs(iou - expected_iou) < 0.01:
            results["tile_manager"]["works"].append(f"iou_calc: {iou:.4f}")
            print(f"✅ IOU calculation: {iou:.4f}")
        else:
            results["tile_manager"]["breaks"].append(f"iou wrong: {iou}")
        
        # Test 2e: GeoOverlay
        overlay = GeoOverlay()
        
        geo_detections = [
            Detection(
                class_id=0, class_name="building", confidence=0.95,
                bbox=(0, 0, 100, 100),
                geo_bbox=(-122.5, 37.7, -122.4, 37.8)
            )
        ]
        
        geojson = overlay.detections_to_geojson(geo_detections)
        
        if geojson["type"] == "FeatureCollection" and len(geojson["features"]) == 1:
            results["tile_manager"]["works"].append("geojson_export")
            print(f"✅ GeoJSON export: {len(geojson['features'])} features")
        else:
            results["tile_manager"]["breaks"].append("geojson failed")
        
        # Test 2f: Tile geo bounds calculation
        manager2 = TileManager()
        tile_geo = manager2._calculate_tile_geo_bounds(
            image_geo=(-122.5, 37.7, -122.3, 37.9),  # Full image
            img_width=2000, img_height=2000,
            x_offset=0, y_offset=0,
            tile_width=1000, tile_height=1000
        )
        
        # First tile should be NW quadrant
        if tile_geo[0] == -122.5 and tile_geo[3] == 37.9:
            results["tile_manager"]["works"].append("tile_geo_calc")
            print(f"✅ Tile geo calculation")
        else:
            results["tile_manager"]["breaks"].append(f"tile_geo wrong: {tile_geo}")
        
    except Exception as e:
        results["tile_manager"]["breaks"].append(f"Exception: {str(e)}")
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()


def test_geo_queries():
    """Test geo-query functionality end-to-end."""
    print_section("TEST 3: Geo Queries End-to-End")
    
    temp_dir = tempfile.mkdtemp(prefix="sv_spatial_test_")
    
    try:
        from core.spatial_index import ProjectSpatialIndex, BoundingBox
        
        # Test 3a: Create project-level index
        index = ProjectSpatialIndex(temp_dir, profile_id="general")
        results["geo_queries"]["works"].append("project_index_created")
        print(f"✅ ProjectSpatialIndex created")
        
        # Test 3b: Index sample images
        index.index_image(
            image_path="datasets/drone_001.jpg",
            bbox=BoundingBox(-122.42, 37.78, -122.40, 37.80),
            dataset_version="dataset_v1",
            metadata={"altitude": 100, "camera": "DJI"}
        )
        
        index.index_image(
            image_path="datasets/drone_002.jpg",
            bbox=BoundingBox(-122.44, 37.76, -122.42, 37.78),
            dataset_version="dataset_v1"
        )
        
        index.index_annotation(
            annotation_id="bbox_001",
            bbox=BoundingBox(-122.415, 37.785, -122.410, 37.790),
            image_path="datasets/drone_001.jpg",
            dataset_version="dataset_v1"
        )
        
        results["geo_queries"]["works"].append("items_indexed: 3")
        print(f"✅ Indexed 3 items (2 images, 1 annotation)")
        
        # Test 3c: Region query
        query_region = BoundingBox(-122.45, 37.75, -122.39, 37.81)
        region_results = index.query_region(
            bbox=query_region,
            limit=10
        )
        
        results["geo_queries"]["works"].append(f"region_query: {len(region_results)} items")
        print(f"✅ Region query: {len(region_results)} items")
        
        # Test 3d: Point query
        point_results = index.query_point(
            lon=-122.41,
            lat=37.79,
            limit=10
        )
        
        results["geo_queries"]["works"].append(f"point_query: {len(point_results)} items")
        print(f"✅ Point query: {len(point_results)} items")
        
        # Test 3e: Type filter
        image_only = index.query_region(
            bbox=query_region,
            item_types=["image"]
        )
        
        if len(image_only) == 2:
            results["geo_queries"]["works"].append("type_filter: images only")
            print(f"✅ Type filter: {len(image_only)} images")
        else:
            results["geo_queries"]["breaks"].append(f"type filter: got {len(image_only)}")
        
        # Test 3f: Version filter
        v1_items = index.query_region(
            bbox=query_region,
            dataset_version="dataset_v1"
        )
        v2_items = index.query_region(
            bbox=query_region,
            dataset_version="dataset_v2"
        )
        
        if len(v1_items) == 3 and len(v2_items) == 0:
            results["geo_queries"]["works"].append("version_filter: verified")
            print(f"✅ Version filter: v1={len(v1_items)}, v2={len(v2_items)}")
        else:
            results["geo_queries"]["breaks"].append("version filter failed")
        
        # Test 3g: Get stats
        stats = index.get_stats()
        results["geo_queries"]["works"].append(
            f"stats: {stats['total_items']} items"
        )
        print(f"✅ Stats: {stats}")
        
        # Test 3h: Coverage bounds
        coverage = index.get_coverage()
        if coverage:
            results["geo_queries"]["works"].append("coverage_bounds")
            print(f"✅ Coverage: {coverage.to_dict()}")
        
    except Exception as e:
        results["geo_queries"]["breaks"].append(f"Exception: {str(e)}")
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def print_summary():
    """Print test summary."""
    print_section("SPATIAL TEST SUMMARY")
    
    for test_name, result in results.items():
        print(f"\n{test_name.upper().replace('_', ' ')}:")
        print(f"  ✅ Works: {len(result['works'])}")
        for item in result['works']:
            print(f"     - {item}")
        
        if result['slow']:
            print(f"  ⚠️ Slow: {len(result['slow'])}")
            for item in result['slow']:
                print(f"     - {item}")
        
        if result['breaks']:
            print(f"  ❌ Breaks: {len(result['breaks'])}")
            for item in result['breaks']:
                print(f"     - {item}")
    
    total_works = sum(len(r['works']) for r in results.values())
    total_slow = sum(len(r['slow']) for r in results.values())
    total_breaks = sum(len(r['breaks']) for r in results.values())
    
    print(f"\n{'='*60}")
    print(f"  OVERALL: ✅ {total_works} | ⚠️ {total_slow} | ❌ {total_breaks}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PHASE A: SPATIAL STRENGTHENING - VALIDATION TESTS")
    print("="*60)
    
    test_spatial_index()
    test_tile_manager()
    test_geo_queries()
    
    final_results = print_summary()
    
    # Save results
    output_path = os.path.join(backend_path, "spatial_test_results.json")
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
