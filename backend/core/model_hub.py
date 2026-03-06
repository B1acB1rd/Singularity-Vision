"""
Model Hub Integration for Singularity Vision

ARCHITECTURE PHILOSOPHY:
- HuggingFace is ONE MODEL SOURCE, not the backbone
- The platform MUST work without HuggingFace (offline-first)
- Local cache is always checked FIRST
- Ultralytics models are built-in and always available

Provider Priority Order:
1. LocalCacheProvider - Always first, works offline
2. UltralyticsProvider - Built-in YOLO models, works offline
3. HuggingFaceProvider - Optional online source
"""

import os
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelSource(Enum):
    """Source of a model."""
    LOCAL = "local"
    ULTRALYTICS = "ultralytics"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class ModelInfo:
    """Information about a model from any provider."""
    model_id: str
    name: str
    author: str
    task_type: str
    downloads: int = 0
    likes: int = 0
    description: str = ""
    tags: List[str] = field(default_factory=list)
    library: str = "unknown"
    pipeline_tag: Optional[str] = None
    source: ModelSource = ModelSource.LOCAL
    local_path: Optional[str] = None
    size_mb: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "author": self.author,
            "task_type": self.task_type,
            "downloads": self.downloads,
            "likes": self.likes,
            "description": self.description,
            "tags": self.tags,
            "library": self.library,
            "pipeline_tag": self.pipeline_tag,
            "source": self.source.value,
            "local_path": self.local_path,
            "size_mb": self.size_mb,
            "is_local": self.local_path is not None
        }


class ModelProvider(ABC):
    """
    Abstract base class for model providers.
    
    Each provider is a PLUGIN that can be enabled/disabled.
    The platform works without any online providers.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def source(self) -> ModelSource:
        """Source type."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (online check for remote providers)."""
        pass
    
    @abstractmethod
    def search(self, query: str, task_type: Optional[str] = None, limit: int = 20) -> List[ModelInfo]:
        """Search for models."""
        pass
    
    @abstractmethod
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get info about a specific model."""
        pass
    
    def download(self, model_id: str, destination: str) -> Optional[str]:
        """Download a model. Default: not supported."""
        return None


class LocalCacheProvider(ModelProvider):
    """
    Local model cache provider.
    
    Always available, always checked first.
    This ensures OFFLINE-FIRST operation.
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    @property
    def name(self) -> str:
        return "Local Cache"
    
    @property
    def source(self) -> ModelSource:
        return ModelSource.LOCAL
    
    def is_available(self) -> bool:
        return True  # Always available
    
    def search(self, query: str = "", task_type: Optional[str] = None, limit: int = 20) -> List[ModelInfo]:
        """Search local cached models."""
        models = []
        
        if not os.path.exists(self.cache_dir):
            return models
        
        query_lower = query.lower()
        
        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            if not os.path.isdir(item_path):
                continue
            
            # Check if matches query
            if query and query_lower not in item.lower():
                continue
            
            # Check for model files
            model_files = ["config.json", "model.safetensors", "pytorch_model.bin", "best.pt", "last.pt"]
            has_model = any(os.path.exists(os.path.join(item_path, f)) for f in model_files)
            
            if not has_model:
                continue
            
            # Load metadata if available
            metadata_path = os.path.join(item_path, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            # Calculate size
            size_mb = sum(
                os.path.getsize(os.path.join(root, file))
                for root, _, files in os.walk(item_path)
                for file in files
            ) / (1024 * 1024)
            
            model = ModelInfo(
                model_id=item.replace("_", "/"),
                name=item.split("_")[-1] if "_" in item else item,
                author=metadata.get("author", "local"),
                task_type=metadata.get("task_type", "unknown"),
                description=metadata.get("description", "Locally cached model"),
                tags=metadata.get("tags", ["local"]),
                library=metadata.get("library", "unknown"),
                source=ModelSource.LOCAL,
                local_path=item_path,
                size_mb=round(size_mb, 2)
            )
            
            if task_type and model.task_type != task_type:
                continue
            
            models.append(model)
            
            if len(models) >= limit:
                break
        
        return models
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a specific local model."""
        safe_id = model_id.replace("/", "_")
        model_path = os.path.join(self.cache_dir, safe_id)
        
        if not os.path.exists(model_path):
            return None
        
        # Search with exact match
        results = self.search(safe_id)
        return results[0] if results else None


class UltralyticsProvider(ModelProvider):
    """
    Ultralytics YOLO models provider.
    
    These are built-in and always available.
    No internet required for basic models.
    """
    
    BUILTIN_MODELS = [
        {"id": "yolov8n.pt", "name": "YOLOv8 Nano", "task": "detection", "size": "3.2", "speed": "fastest"},
        {"id": "yolov8s.pt", "name": "YOLOv8 Small", "task": "detection", "size": "11.2", "speed": "fast"},
        {"id": "yolov8m.pt", "name": "YOLOv8 Medium", "task": "detection", "size": "25.9", "speed": "balanced"},
        {"id": "yolov8l.pt", "name": "YOLOv8 Large", "task": "detection", "size": "43.7", "speed": "accurate"},
        {"id": "yolov8x.pt", "name": "YOLOv8 XLarge", "task": "detection", "size": "68.2", "speed": "most accurate"},
        {"id": "yolov8n-seg.pt", "name": "YOLOv8n Segmentation", "task": "segmentation", "size": "3.4", "speed": "fastest"},
        {"id": "yolov8s-seg.pt", "name": "YOLOv8s Segmentation", "task": "segmentation", "size": "11.8", "speed": "fast"},
        {"id": "yolov8n-cls.pt", "name": "YOLOv8n Classification", "task": "classification", "size": "2.7", "speed": "fastest"},
        {"id": "yolov8s-cls.pt", "name": "YOLOv8s Classification", "task": "classification", "size": "6.4", "speed": "fast"},
    ]
    
    @property
    def name(self) -> str:
        return "Ultralytics"
    
    @property
    def source(self) -> ModelSource:
        return ModelSource.ULTRALYTICS
    
    def is_available(self) -> bool:
        return True  # Always available (models auto-download on first use)
    
    def search(self, query: str = "", task_type: Optional[str] = None, limit: int = 20) -> List[ModelInfo]:
        """Search Ultralytics models."""
        models = []
        query_lower = query.lower()
        
        for m in self.BUILTIN_MODELS:
            # Filter by query
            if query and query_lower not in m["name"].lower() and query_lower not in m["id"].lower():
                continue
            
            # Filter by task
            if task_type:
                # Map task type names
                task_map = {
                    "detection": "detection",
                    "detect": "detection",
                    "segmentation": "segmentation",
                    "segment": "segmentation",
                    "classification": "classification",
                    "classify": "classification"
                }
                if task_map.get(task_type.lower()) != m["task"]:
                    continue
            
            models.append(ModelInfo(
                model_id=m["id"],
                name=m["name"],
                author="Ultralytics",
                task_type=m["task"],
                description=f"{m['speed'].capitalize()} model, {m['size']}MB",
                tags=["yolo", "ultralytics", m["task"], m["speed"]],
                library="ultralytics",
                source=ModelSource.ULTRALYTICS,
                size_mb=float(m["size"])
            ))
            
            if len(models) >= limit:
                break
        
        return models
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a specific Ultralytics model."""
        for m in self.BUILTIN_MODELS:
            if m["id"] == model_id:
                return ModelInfo(
                    model_id=m["id"],
                    name=m["name"],
                    author="Ultralytics",
                    task_type=m["task"],
                    description=f"{m['speed'].capitalize()} model, {m['size']}MB",
                    tags=["yolo", "ultralytics", m["task"]],
                    library="ultralytics",
                    source=ModelSource.ULTRALYTICS,
                    size_mb=float(m["size"])
                )
        return None


class HuggingFaceProvider(ModelProvider):
    """
    Hugging Face Hub provider.
    
    OPTIONAL online source - platform works without it.
    Only used when explicitly requested or when local models not found.
    """
    
    HF_API_URL = "https://huggingface.co/api"
    
    def __init__(self):
        self._available = None
    
    @property
    def name(self) -> str:
        return "Hugging Face"
    
    @property
    def source(self) -> ModelSource:
        return ModelSource.HUGGINGFACE
    
    def is_available(self) -> bool:
        """Check if HuggingFace is reachable."""
        if self._available is not None:
            return self._available
        
        try:
            response = requests.get(f"{self.HF_API_URL}/models", timeout=3, params={"limit": 1})
            self._available = response.status_code == 200
        except:
            self._available = False
        
        return self._available
    
    def search(self, query: str = "", task_type: Optional[str] = None, limit: int = 20) -> List[ModelInfo]:
        """Search HuggingFace Hub for models."""
        if not self.is_available():
            logger.warning("HuggingFace is not available (offline mode)")
            return []
        
        try:
            # Map task types to HuggingFace pipeline tags
            task_map = {
                "detection": "object-detection",
                "segmentation": "image-segmentation",
                "classification": "image-classification"
            }
            pipeline_tag = task_map.get(task_type, task_type) if task_type else "object-detection"
            
            params = {
                "search": query,
                "pipeline_tag": pipeline_tag,
                "limit": limit,
                "sort": "downloads",
                "direction": -1
            }
            
            response = requests.get(f"{self.HF_API_URL}/models", params=params, timeout=10)
            
            if response.status_code != 200:
                return []
            
            models = []
            for item in response.json():
                models.append(ModelInfo(
                    model_id=item.get("id", ""),
                    name=item.get("id", "").split("/")[-1],
                    author=item.get("author", "unknown"),
                    task_type=task_type or "detection",
                    downloads=item.get("downloads", 0),
                    likes=item.get("likes", 0),
                    description=item.get("description", ""),
                    tags=item.get("tags", []),
                    library=item.get("library_name", "unknown"),
                    pipeline_tag=item.get("pipeline_tag"),
                    source=ModelSource.HUGGINGFACE
                ))
            
            return models
            
        except Exception as e:
            logger.error(f"HuggingFace search failed: {e}")
            return []
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get detailed info about a HuggingFace model."""
        if not self.is_available():
            return None
        
        try:
            response = requests.get(f"{self.HF_API_URL}/models/{model_id}", timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            return ModelInfo(
                model_id=model_id,
                name=model_id.split("/")[-1],
                author=data.get("author", "unknown"),
                task_type=data.get("pipeline_tag", "unknown"),
                downloads=data.get("downloads", 0),
                likes=data.get("likes", 0),
                description=data.get("cardData", {}).get("description", ""),
                tags=data.get("tags", []),
                library=data.get("library_name", "unknown"),
                pipeline_tag=data.get("pipeline_tag"),
                source=ModelSource.HUGGINGFACE
            )
            
        except Exception as e:
            logger.error(f"HuggingFace get_model failed: {e}")
            return None
    
    def download(self, model_id: str, destination: str) -> Optional[str]:
        """Download a model from HuggingFace Hub."""
        if not self.is_available():
            logger.error("Cannot download: HuggingFace is not available")
            return None
        
        try:
            from huggingface_hub import snapshot_download
            
            path = snapshot_download(
                repo_id=model_id,
                local_dir=destination,
                local_dir_use_symlinks=False
            )
            return path
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return None


class ModelHub:
    """
    Unified Model Hub with Provider Architecture.
    
    DESIGN PHILOSOPHY:
    - Local cache is ALWAYS checked first
    - Ultralytics models are built-in and always available  
    - HuggingFace is an OPTIONAL online source
    - Platform works 100% offline
    
    Provider Order:
    1. LocalCacheProvider (always first, ensures offline works)
    2. UltralyticsProvider (built-in YOLO models)
    3. HuggingFaceProvider (optional, requires internet)
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), 
            ".singularity-vision", 
            "models"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize providers in priority order
        self.providers: List[ModelProvider] = [
            LocalCacheProvider(self.cache_dir),
            UltralyticsProvider(),
            HuggingFaceProvider()
        ]
        
        logger.info(f"ModelHub initialized with {len(self.providers)} providers")
    
    def get_provider_status(self) -> List[Dict]:
        """Get status of all providers."""
        return [
            {
                "name": p.name,
                "source": p.source.value,
                "available": p.is_available()
            }
            for p in self.providers
        ]
    
    def search(
        self,
        query: str = "",
        task_type: Optional[str] = None,
        limit: int = 20,
        sources: Optional[List[str]] = None
    ) -> List[ModelInfo]:
        """
        Search for models across all providers.
        
        Local cache is always searched first.
        
        Args:
            query: Search query
            task_type: Filter by task type
            limit: Maximum results per provider
            sources: Optional list of sources to search (default: all)
        
        Returns:
            Combined list of ModelInfo from all providers
        """
        all_models = []
        
        for provider in self.providers:
            # Filter by source if specified
            if sources and provider.source.value not in sources:
                continue
            
            # Skip unavailable providers
            if not provider.is_available():
                logger.debug(f"Skipping unavailable provider: {provider.name}")
                continue
            
            try:
                models = provider.search(query, task_type, limit)
                all_models.extend(models)
                logger.debug(f"{provider.name}: found {len(models)} models")
            except Exception as e:
                logger.error(f"Search failed for {provider.name}: {e}")
        
        return all_models
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get info about a specific model.
        
        Checks providers in priority order.
        """
        for provider in self.providers:
            if not provider.is_available():
                continue
            
            try:
                model = provider.get_model(model_id)
                if model:
                    return model
            except Exception as e:
                logger.error(f"Get model failed for {provider.name}: {e}")
        
        return None
    
    def download(self, model_id: str, destination: str = None) -> Optional[str]:
        """
        Download a model.
        
        First checks if already in local cache.
        """
        dest = destination or os.path.join(self.cache_dir, model_id.replace("/", "_"))
        
        # Check if already downloaded
        if os.path.exists(dest):
            logger.info(f"Model already in cache: {model_id}")
            return dest
        
        # Try to download from appropriate provider
        for provider in self.providers:
            if not provider.is_available():
                continue
            
            try:
                path = provider.download(model_id, dest)
                if path:
                    logger.info(f"Downloaded {model_id} from {provider.name}")
                    # Save metadata
                    self._save_model_metadata(dest, model_id, provider.source)
                    return path
            except Exception as e:
                logger.error(f"Download failed from {provider.name}: {e}")
        
        return None
    
    def _save_model_metadata(self, path: str, model_id: str, source: ModelSource):
        """Save metadata about a downloaded model."""
        from datetime import datetime
        metadata = {
            "model_id": model_id,
            "source": source.value,
            "downloaded_at": datetime.now().isoformat()
        }
        metadata_path = os.path.join(path, "metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except:
            pass
    
    def get_local_models(self) -> List[ModelInfo]:
        """Get all locally cached models."""
        local_provider = self.providers[0]  # LocalCacheProvider
        return local_provider.search("", None, 1000)
    
    def get_featured_models(self, task_type: str = None) -> List[ModelInfo]:
        """
        Get featured/recommended models.
        
        Returns Ultralytics models first (always available),
        then curated HuggingFace models if online.
        """
        ultralytics_provider = self.providers[1]  # UltralyticsProvider
        models = ultralytics_provider.search("", task_type, 20)
        
        # Add curated HuggingFace models if available
        hf_provider = self.providers[2]  # HuggingFaceProvider
        if hf_provider.is_available():
            curated_ids = [
                "facebook/detr-resnet-50",
                "facebook/sam-vit-base",
            ]
            for model_id in curated_ids:
                model = hf_provider.get_model(model_id)
                if model:
                    if not task_type or model.task_type == task_type:
                        models.append(model)
        
        return models
    
    # Legacy method for backwards compatibility
    def search_huggingface(
        self,
        query: str = "",
        task: str = "object-detection",
        limit: int = 20
    ) -> List[ModelInfo]:
        """Legacy method - searches HuggingFace only."""
        hf_provider = self.providers[2]  # HuggingFaceProvider
        if not hf_provider.is_available():
            logger.warning("HuggingFace not available - returning empty")
            return []
        return hf_provider.search(query, task, limit)
    
    # Legacy method for backwards compatibility
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Legacy method - alias for get_model."""
        return self.get_model(model_id)
    
    # Legacy method for backwards compatibility
    def download_model(self, model_id: str, destination: str = None) -> Optional[str]:
        """Legacy method - alias for download."""
        return self.download(model_id, destination)


# Singleton instance
model_hub = ModelHub()

