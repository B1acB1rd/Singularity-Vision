"""
Annotation Export Module for Singularity Vision

Supports export to industry-standard formats:
- COCO (Common Objects in Context)
- Pascal VOC (Visual Object Classes)
- Full Project Bundle

PHILOSOPHY: Exported artifacts must be usable outside the platform 
without manual stitching or fixing.
"""

import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import zipfile
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnnotationItem:
    """Standard annotation representation."""
    annotation_id: str
    image_id: str
    image_path: str
    category_id: int
    category_name: str
    bbox: Tuple[float, float, float, float]  # x, y, width, height (COCO style)
    area: Optional[float] = None
    segmentation: Optional[List] = None
    iscrowd: int = 0
    score: Optional[float] = None
    
    def to_coco(self) -> Dict:
        """Convert to COCO format."""
        result = {
            "id": hash(self.annotation_id) % (10 ** 9),
            "image_id": hash(self.image_id) % (10 ** 9),
            "category_id": self.category_id,
            "bbox": list(self.bbox),
            "area": self.area or (self.bbox[2] * self.bbox[3]),
            "iscrowd": self.iscrowd
        }
        if self.segmentation:
            result["segmentation"] = self.segmentation
        if self.score is not None:
            result["score"] = self.score
        return result


@dataclass
class ImageMetadata:
    """Image metadata for export."""
    image_id: str
    file_name: str
    width: int
    height: int
    date_captured: Optional[str] = None
    
    def to_coco(self) -> Dict:
        return {
            "id": hash(self.image_id) % (10 ** 9),
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "date_captured": self.date_captured or datetime.now().isoformat()
        }


@dataclass
class CategoryInfo:
    """Category/class definition."""
    category_id: int
    name: str
    supercategory: str = ""
    
    def to_coco(self) -> Dict:
        return {
            "id": self.category_id,
            "name": self.name,
            "supercategory": self.supercategory or self.name
        }


class COCOExporter:
    """
    Export annotations in COCO format.
    
    COCO format structure:
    {
        "info": {...},
        "licenses": [...],
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }
    """
    
    def __init__(self, project_name: str = "Singularity Vision Export"):
        self.project_name = project_name
    
    def export(
        self,
        images: List[ImageMetadata],
        annotations: List[AnnotationItem],
        categories: List[CategoryInfo],
        output_path: str
    ) -> str:
        """Export to COCO JSON format."""
        coco_data = {
            "info": {
                "description": self.project_name,
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Singularity Vision",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "images": [img.to_coco() for img in images],
            "annotations": [ann.to_coco() for ann in annotations],
            "categories": [cat.to_coco() for cat in categories]
        }
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"Exported COCO: {len(images)} images, {len(annotations)} annotations")
        return output_path
    
    def export_from_project(self, project_path: str, output_path: Optional[str] = None) -> str:
        """Export entire project to COCO format."""
        if not output_path:
            output_path = os.path.join(project_path, "exports", "coco_annotations.json")
        
        images, annotations, categories = self._load_project_data(project_path)
        return self.export(images, annotations, categories, output_path)
    
    def _load_project_data(self, project_path: str) -> Tuple[List, List, List]:
        """Load annotations from project directory."""
        images = []
        annotations = []
        categories_dict: Dict[str, int] = {}
        
        # Look for annotation files
        annotations_dir = os.path.join(project_path, "annotations")
        if not os.path.exists(annotations_dir):
            annotations_dir = os.path.join(project_path, "labels")
        
        if os.path.exists(annotations_dir):
            for file_name in os.listdir(annotations_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(annotations_dir, file_name)
                    img_info, anns = self._parse_annotation_file(
                        file_path, categories_dict
                    )
                    if img_info:
                        images.append(img_info)
                        annotations.extend(anns)
        
        categories = [
            CategoryInfo(category_id=cid, name=name)
            for name, cid in categories_dict.items()
        ]
        
        return images, annotations, categories
    
    def _parse_annotation_file(
        self,
        file_path: str,
        categories_dict: Dict[str, int]
    ) -> Tuple[Optional[ImageMetadata], List[AnnotationItem]]:
        """Parse a single annotation file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            image_id = data.get("image_id", os.path.splitext(os.path.basename(file_path))[0])
            file_name = data.get("file_name", image_id + ".jpg")
            
            img_info = ImageMetadata(
                image_id=image_id,
                file_name=file_name,
                width=data.get("width", 640),
                height=data.get("height", 640)
            )
            
            annotations = []
            for i, ann_data in enumerate(data.get("annotations", [])):
                cat_name = ann_data.get("category", ann_data.get("class", "unknown"))
                
                if cat_name not in categories_dict:
                    categories_dict[cat_name] = len(categories_dict) + 1
                
                bbox = ann_data.get("bbox", [0, 0, 0, 0])
                if len(bbox) == 4:
                    ann = AnnotationItem(
                        annotation_id=f"{image_id}_{i}",
                        image_id=image_id,
                        image_path=file_name,
                        category_id=categories_dict[cat_name],
                        category_name=cat_name,
                        bbox=tuple(bbox)
                    )
                    annotations.append(ann)
            
            return img_info, annotations
            
        except Exception as e:
            logger.warning(f"Could not parse {file_path}: {e}")
            return None, []


class PascalVOCExporter:
    """Export annotations in Pascal VOC XML format."""
    
    def export_image(
        self,
        image_info: ImageMetadata,
        annotations: List[AnnotationItem],
        output_path: str
    ) -> str:
        """Export single image annotations to Pascal VOC XML."""
        annotation = ET.Element("annotation")
        
        ET.SubElement(annotation, "folder").text = "images"
        ET.SubElement(annotation, "filename").text = image_info.file_name
        
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Singularity Vision"
        
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(image_info.width)
        ET.SubElement(size, "height").text = str(image_info.height)
        ET.SubElement(size, "depth").text = "3"
        
        ET.SubElement(annotation, "segmented").text = "0"
        
        for ann in annotations:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = ann.category_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            x, y, w, h = ann.bbox
            ET.SubElement(bndbox, "xmin").text = str(int(x))
            ET.SubElement(bndbox, "ymin").text = str(int(y))
            ET.SubElement(bndbox, "xmax").text = str(int(x + w))
            ET.SubElement(bndbox, "ymax").text = str(int(y + h))
        
        xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
        xml_str = '\n'.join(xml_str.split('\n')[1:])
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(xml_str)
        
        return output_path
    
    def export_dataset(
        self,
        images: List[ImageMetadata],
        annotations: List[AnnotationItem],
        output_dir: str
    ) -> List[str]:
        """Export full dataset to Pascal VOC format."""
        os.makedirs(output_dir, exist_ok=True)
        
        anns_by_image: Dict[str, List[AnnotationItem]] = {}
        for ann in annotations:
            if ann.image_id not in anns_by_image:
                anns_by_image[ann.image_id] = []
            anns_by_image[ann.image_id].append(ann)
        
        exported = []
        for img in images:
            img_anns = anns_by_image.get(img.image_id, [])
            output_path = os.path.join(
                output_dir, 
                os.path.splitext(img.file_name)[0] + ".xml"
            )
            self.export_image(img, img_anns, output_path)
            exported.append(output_path)
        
        logger.info(f"Exported Pascal VOC: {len(exported)} files")
        return exported


class ProjectBundleExporter:
    """Export complete project as a portable bundle."""
    
    def export(
        self,
        project_path: str,
        output_path: str,
        include_models: bool = False,
        include_experiments: bool = True,
        formats: List[str] = None
    ) -> str:
        """Export project as a ZIP bundle."""
        if formats is None:
            formats = ["coco", "voc"]
        
        bundle_dir = output_path + "_temp"
        os.makedirs(bundle_dir, exist_ok=True)
        
        try:
            # Copy datasets
            datasets_src = os.path.join(project_path, "datasets")
            if os.path.exists(datasets_src):
                shutil.copytree(datasets_src, os.path.join(bundle_dir, "datasets"))
            
            # Copy annotations
            annotations_src = os.path.join(project_path, "annotations")
            if os.path.exists(annotations_src):
                shutil.copytree(annotations_src, os.path.join(bundle_dir, "annotations"))
            
            # Export in requested formats
            exports_dir = os.path.join(bundle_dir, "exports")
            os.makedirs(exports_dir, exist_ok=True)
            
            if "coco" in formats:
                coco_exporter = COCOExporter()
                coco_exporter.export_from_project(
                    project_path,
                    os.path.join(exports_dir, "coco_annotations.json")
                )
            
            if include_models:
                models_src = os.path.join(project_path, "models")
                if os.path.exists(models_src):
                    shutil.copytree(models_src, os.path.join(bundle_dir, "models"))
            
            if include_experiments:
                exp_src = os.path.join(project_path, ".experiments")
                if os.path.exists(exp_src):
                    shutil.copytree(exp_src, os.path.join(bundle_dir, "experiments"))
            
            # Create manifest
            manifest = {
                "name": os.path.basename(project_path),
                "exported_at": datetime.now().isoformat(),
                "version": "1.0",
                "formats": formats,
                "includes_models": include_models,
                "includes_experiments": include_experiments
            }
            
            with open(os.path.join(bundle_dir, "manifest.json"), 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create ZIP archive
            zip_path = output_path if output_path.endswith('.zip') else output_path + '.zip'
            shutil.make_archive(zip_path.replace('.zip', ''), 'zip', bundle_dir)
            
            logger.info(f"Exported project bundle: {zip_path}")
            return zip_path
            
        finally:
            shutil.rmtree(bundle_dir, ignore_errors=True)


# Convenience instances
coco_exporter = COCOExporter()
voc_exporter = PascalVOCExporter()
bundle_exporter = ProjectBundleExporter()
