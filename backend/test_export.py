"""
Phase C: Export & Interoperability Tests

Tests:
1. COCO format export
2. Pascal VOC format export
3. Project bundle export
"""

import os
import sys
import json
import tempfile
import shutil

backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

results = {
    "coco_export": {"works": [], "slow": [], "breaks": []},
    "voc_export": {"works": [], "slow": [], "breaks": []},
    "bundle_export": {"works": [], "slow": [], "breaks": []}
}


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def create_test_project(base_dir: str):
    """Create a test project with annotations."""
    # Create directories
    os.makedirs(os.path.join(base_dir, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "annotations"), exist_ok=True)
    
    # Create dummy images
    for i in range(3):
        img_path = os.path.join(base_dir, "datasets", f"image_{i:03d}.jpg")
        with open(img_path, 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0' + b'\x00' * 100)
    
    # Create annotation files
    for i in range(3):
        ann_data = {
            "image_id": f"image_{i:03d}",
            "file_name": f"image_{i:03d}.jpg",
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "category": "car",
                    "bbox": [100, 100, 200, 150]
                },
                {
                    "category": "person",
                    "bbox": [300, 200, 100, 300]
                }
            ]
        }
        ann_path = os.path.join(base_dir, "annotations", f"image_{i:03d}.json")
        with open(ann_path, 'w') as f:
            json.dump(ann_data, f)
    
    return base_dir


def test_coco_export():
    """Test COCO format export."""
    print_section("TEST 1: COCO Export")
    
    temp_dir = tempfile.mkdtemp(prefix="sv_coco_test_")
    
    try:
        from core.annotation_exporter import (
            COCOExporter, ImageMetadata, AnnotationItem, CategoryInfo
        )
        
        # Test 1a: Create exporter
        exporter = COCOExporter("Test Project")
        results["coco_export"]["works"].append("exporter_created")
        print(f"✅ COCOExporter created")
        
        # Test 1b: Create sample data
        images = [
            ImageMetadata("img_001", "image_001.jpg", 640, 480),
            ImageMetadata("img_002", "image_002.jpg", 1280, 720)
        ]
        
        annotations = [
            AnnotationItem("ann_001", "img_001", "image_001.jpg", 1, "car", (100, 100, 200, 150)),
            AnnotationItem("ann_002", "img_001", "image_001.jpg", 2, "person", (300, 200, 100, 300)),
            AnnotationItem("ann_003", "img_002", "image_002.jpg", 1, "car", (50, 50, 300, 200))
        ]
        
        categories = [
            CategoryInfo(1, "car", "vehicle"),
            CategoryInfo(2, "person", "human")
        ]
        
        # Test 1c: Export
        output_path = os.path.join(temp_dir, "coco_test.json")
        result = exporter.export(images, annotations, categories, output_path)
        
        if os.path.exists(result):
            results["coco_export"]["works"].append("export_success")
            print(f"✅ Export created: {result}")
        else:
            results["coco_export"]["breaks"].append("export_failed")
        
        # Test 1d: Verify structure
        with open(result, 'r') as f:
            coco_data = json.load(f)
        
        has_all_keys = all(k in coco_data for k in ["info", "images", "annotations", "categories"])
        
        if has_all_keys:
            results["coco_export"]["works"].append("structure_valid")
            print(f"✅ COCO structure valid")
        else:
            results["coco_export"]["breaks"].append("missing_keys")
        
        # Test 1e: Verify counts
        if len(coco_data["images"]) == 2:
            results["coco_export"]["works"].append(f"images: {len(coco_data['images'])}")
            print(f"✅ Images count: {len(coco_data['images'])}")
        
        if len(coco_data["annotations"]) == 3:
            results["coco_export"]["works"].append(f"annotations: {len(coco_data['annotations'])}")
            print(f"✅ Annotations count: {len(coco_data['annotations'])}")
        
        if len(coco_data["categories"]) == 2:
            results["coco_export"]["works"].append(f"categories: {len(coco_data['categories'])}")
            print(f"✅ Categories count: {len(coco_data['categories'])}")
        
        # Test 1f: Export from project
        project_dir = os.path.join(temp_dir, "test_project")
        create_test_project(project_dir)
        
        project_output = os.path.join(temp_dir, "project_coco.json")
        exporter.export_from_project(project_dir, project_output)
        
        if os.path.exists(project_output):
            results["coco_export"]["works"].append("project_export")
            print(f"✅ Project export successful")
        else:
            results["coco_export"]["breaks"].append("project_export_failed")
        
    except Exception as e:
        results["coco_export"]["breaks"].append(f"Exception: {str(e)}")
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_voc_export():
    """Test Pascal VOC format export."""
    print_section("TEST 2: Pascal VOC Export")
    
    temp_dir = tempfile.mkdtemp(prefix="sv_voc_test_")
    
    try:
        from core.annotation_exporter import (
            PascalVOCExporter, ImageMetadata, AnnotationItem
        )
        
        # Test 2a: Create exporter
        exporter = PascalVOCExporter()
        results["voc_export"]["works"].append("exporter_created")
        print(f"✅ PascalVOCExporter created")
        
        # Test 2b: Export single image
        img = ImageMetadata("img_001", "image_001.jpg", 640, 480)
        anns = [
            AnnotationItem("ann_001", "img_001", "image_001.jpg", 1, "car", (100, 100, 200, 150)),
            AnnotationItem("ann_002", "img_001", "image_001.jpg", 2, "person", (300, 200, 100, 300))
        ]
        
        output_path = os.path.join(temp_dir, "image_001.xml")
        result = exporter.export_image(img, anns, output_path)
        
        if os.path.exists(result):
            results["voc_export"]["works"].append("single_export")
            print(f"✅ Single image export: {result}")
        else:
            results["voc_export"]["breaks"].append("single_export_failed")
        
        # Test 2c: Verify XML structure
        with open(result, 'r') as f:
            content = f.read()
        
        has_elements = all(tag in content for tag in ["<annotation>", "<filename>", "<object>", "<bndbox>"])
        
        if has_elements:
            results["voc_export"]["works"].append("xml_structure_valid")
            print(f"✅ XML structure valid")
        else:
            results["voc_export"]["breaks"].append("missing_xml_elements")
        
        # Test 2d: Verify bbox conversion (COCO to VOC)
        # COCO: x, y, width, height -> VOC: xmin, ymin, xmax, ymax
        if "<xmin>100</xmin>" in content and "<xmax>300</xmax>" in content:
            results["voc_export"]["works"].append("bbox_conversion")
            print(f"✅ BBox conversion correct (x=100, w=200 → xmax=300)")
        else:
            results["voc_export"]["breaks"].append("bbox_conversion_wrong")
        
        # Test 2e: Export dataset
        images = [
            ImageMetadata("img_001", "image_001.jpg", 640, 480),
            ImageMetadata("img_002", "image_002.jpg", 1280, 720)
        ]
        
        annotations = [
            AnnotationItem("ann_001", "img_001", "image_001.jpg", 1, "car", (100, 100, 200, 150)),
            AnnotationItem("ann_002", "img_002", "image_002.jpg", 2, "truck", (50, 50, 400, 300))
        ]
        
        output_dir = os.path.join(temp_dir, "voc_output")
        exported = exporter.export_dataset(images, annotations, output_dir)
        
        if len(exported) == 2:
            results["voc_export"]["works"].append(f"dataset_export: {len(exported)} files")
            print(f"✅ Dataset export: {len(exported)} files")
        else:
            results["voc_export"]["breaks"].append(f"wrong count: {len(exported)}")
        
    except Exception as e:
        results["voc_export"]["breaks"].append(f"Exception: {str(e)}")
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_bundle_export():
    """Test project bundle export."""
    print_section("TEST 3: Project Bundle Export")
    
    temp_dir = tempfile.mkdtemp(prefix="sv_bundle_test_")
    
    try:
        from core.annotation_exporter import ProjectBundleExporter
        
        # Setup test project
        project_dir = os.path.join(temp_dir, "test_project")
        create_test_project(project_dir)
        
        # Create exporter
        exporter = ProjectBundleExporter()
        results["bundle_export"]["works"].append("exporter_created")
        print(f"✅ ProjectBundleExporter created")
        
        # Test 3a: Export bundle
        output_path = os.path.join(temp_dir, "project_bundle.zip")
        result = exporter.export(
            project_path=project_dir,
            output_path=output_path,
            include_models=False,
            include_experiments=False,
            formats=["coco"]
        )
        
        if os.path.exists(result):
            results["bundle_export"]["works"].append("bundle_created")
            print(f"✅ Bundle created: {result}")
        else:
            results["bundle_export"]["breaks"].append("bundle_failed")
        
        # Test 3b: Verify bundle contents
        import zipfile
        
        with zipfile.ZipFile(result, 'r') as zf:
            file_list = zf.namelist()
        
        has_manifest = any("manifest.json" in f for f in file_list)
        has_datasets = any("datasets" in f for f in file_list)
        
        if has_manifest:
            results["bundle_export"]["works"].append("has_manifest")
            print(f"✅ Bundle has manifest")
        
        if has_datasets:
            results["bundle_export"]["works"].append("has_datasets")
            print(f"✅ Bundle has datasets")
        
        results["bundle_export"]["works"].append(f"total_files: {len(file_list)}")
        print(f"✅ Total files in bundle: {len(file_list)}")
        
    except Exception as e:
        results["bundle_export"]["breaks"].append(f"Exception: {str(e)}")
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def print_summary():
    """Print test summary."""
    print_section("PHASE C TEST SUMMARY")
    
    for test_name, result in results.items():
        print(f"\n{test_name.upper().replace('_', ' ')}:")
        print(f"  ✅ Works: {len(result['works'])}")
        for item in result['works']:
            print(f"     - {item}")
        
        if result['slow']:
            print(f"  ⚠️ Slow: {len(result['slow'])}")
        
        if result['breaks']:
            print(f"  ❌ Breaks: {len(result['breaks'])}")
            for item in result['breaks']:
                print(f"     - {item}")
    
    total_works = sum(len(r['works']) for r in results.values())
    total_breaks = sum(len(r['breaks']) for r in results.values())
    
    print(f"\n{'='*60}")
    print(f"  OVERALL: ✅ {total_works} | ❌ {total_breaks}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PHASE C: EXPORT & INTEROPERABILITY - VALIDATION TESTS")
    print("="*60)
    
    test_coco_export()
    test_voc_export()
    test_bundle_export()
    
    final_results = print_summary()
    
    output_path = os.path.join(backend_path, "export_test_results.json")
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
