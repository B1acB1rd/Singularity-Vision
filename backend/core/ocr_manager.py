"""
OCR Manager - Optical Character Recognition for text detection and extraction
"""
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class OCRManager:
    """
    Manages OCR operations for text detection and extraction.
    
    Features:
    - Text detection (find text regions)
    - Text extraction (read text content)
    - Multiple output formats
    - Batch processing
    """
    
    def __init__(self):
        self.reader = None
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
    
    def _ensure_reader(self, languages: List[str] = ['en']):
        """Lazy-load the OCR reader"""
        if self.reader is None:
            try:
                import easyocr
                self.reader = easyocr.Reader(languages, gpu=self._check_gpu())
            except ImportError:
                # Fallback to tesseract
                try:
                    import pytesseract
                    self.reader = "tesseract"
                except ImportError:
                    raise RuntimeError("No OCR backend available. Install easyocr or pytesseract.")
        return self.reader
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def detect_text(
        self,
        image_path: str,
        languages: List[str] = ['en'],
        conf_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect text regions in an image.
        
        Args:
            image_path: Path to image file
            languages: Languages to detect
            conf_threshold: Confidence threshold
            
        Returns:
            Dict with detected text regions
        """
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        try:
            reader = self._ensure_reader(languages)
            
            if reader == "tesseract":
                return self._detect_with_tesseract(image_path, conf_threshold)
            else:
                return self._detect_with_easyocr(image_path, reader, conf_threshold)
                
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_with_easyocr(
        self,
        image_path: str,
        reader,
        conf_threshold: float
    ) -> Dict[str, Any]:
        """Detect text using EasyOCR"""
        results = reader.readtext(image_path)
        
        detections = []
        full_text = []
        
        for bbox, text, confidence in results:
            if confidence >= conf_threshold:
                # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                
                detections.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": {
                        "x_min": min(x_coords),
                        "y_min": min(y_coords),
                        "x_max": max(x_coords),
                        "y_max": max(y_coords)
                    },
                    "polygon": [[p[0], p[1]] for p in bbox]
                })
                full_text.append(text)
        
        return {
            "status": "success",
            "image_path": image_path,
            "detection_count": len(detections),
            "detections": detections,
            "full_text": " ".join(full_text),
            "timestamp": datetime.now().isoformat()
        }
    
    def _detect_with_tesseract(
        self,
        image_path: str,
        conf_threshold: float
    ) -> Dict[str, Any]:
        """Detect text using Tesseract"""
        import pytesseract
        
        img = cv2.imread(image_path)
        
        # Get detailed data
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        detections = []
        full_text = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            conf = int(data['conf'][i])
            if conf >= conf_threshold * 100:
                text = data['text'][i].strip()
                if text:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    detections.append({
                        "text": text,
                        "confidence": conf / 100.0,
                        "bbox": {
                            "x_min": x,
                            "y_min": y,
                            "x_max": x + w,
                            "y_max": y + h
                        }
                    })
                    full_text.append(text)
        
        return {
            "status": "success",
            "image_path": image_path,
            "detection_count": len(detections),
            "detections": detections,
            "full_text": " ".join(full_text),
            "timestamp": datetime.now().isoformat()
        }
    
    def extract_text_only(
        self,
        image_path: str,
        languages: List[str] = ['en']
    ) -> str:
        """Extract just the text content from an image"""
        result = self.detect_text(image_path, languages)
        
        if "error" in result:
            return ""
        
        return result.get("full_text", "")
    
    def batch_detect(
        self,
        image_paths: List[str],
        languages: List[str] = ['en'],
        conf_threshold: float = 0.5,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run OCR on multiple images.
        
        Args:
            image_paths: List of image paths
            languages: Languages to detect
            conf_threshold: Confidence threshold
            progress_callback: Callback(current, total)
            
        Returns:
            Dict with batch results
        """
        results = []
        total = len(image_paths)
        total_text_found = 0
        
        for i, path in enumerate(image_paths):
            result = self.detect_text(path, languages, conf_threshold)
            results.append({
                "image_path": path,
                "result": result
            })
            
            if "detection_count" in result:
                total_text_found += result["detection_count"]
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return {
            "status": "success",
            "total_images": total,
            "total_text_regions": total_text_found,
            "results": results
        }
    
    def visualize_detections(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Draw text detection boxes on image.
        
        Args:
            image_path: Original image path
            detections: Detection results
            output_path: Output path (optional)
            
        Returns:
            Path to annotated image
        """
        img = cv2.imread(image_path)
        
        for det in detections:
            bbox = det.get("bbox", {})
            text = det.get("text", "")
            conf = det.get("confidence", 0)
            
            x1 = int(bbox.get("x_min", 0))
            y1 = int(bbox.get("y_min", 0))
            x2 = int(bbox.get("x_max", 0))
            y2 = int(bbox.get("y_max", 0))
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw text label
            label = f"{text[:20]}{'...' if len(text) > 20 else ''} ({conf:.2f})"
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_ocr{ext}"
        
        cv2.imwrite(output_path, img)
        return output_path
    
    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export OCR results to file.
        
        Args:
            results: OCR results dict
            output_path: Output file path
            format: 'json', 'txt', or 'csv'
        """
        import json
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        elif format == "txt":
            with open(output_path, 'w') as f:
                if "results" in results:  # Batch results
                    for r in results["results"]:
                        f.write(f"--- {r['image_path']} ---\n")
                        f.write(r["result"].get("full_text", "") + "\n\n")
                else:
                    f.write(results.get("full_text", ""))
        
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["image", "text", "confidence", "x_min", "y_min", "x_max", "y_max"])
                
                if "results" in results:
                    for r in results["results"]:
                        for det in r["result"].get("detections", []):
                            bbox = det.get("bbox", {})
                            writer.writerow([
                                r["image_path"],
                                det.get("text", ""),
                                det.get("confidence", 0),
                                bbox.get("x_min", 0),
                                bbox.get("y_min", 0),
                                bbox.get("x_max", 0),
                                bbox.get("y_max", 0)
                            ])
        
        return {"status": "success", "path": output_path, "format": format}


# Singleton instance
ocr_manager = OCRManager()
