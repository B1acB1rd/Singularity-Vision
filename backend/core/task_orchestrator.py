"""
Smart Task Orchestrator for Singularity Vision

Central decision-maker for all heavy tasks.
Runs BEFORE any engine execution to ensure:
- Platform is not heavy on users' PCs
- Users trust the platform's intelligence about their hardware
- Correct execution mode (local/hybrid/remote) is chosen
"""

import os
import sys
import platform
import threading
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import psutil

# Try to import GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ExecutionMode(Enum):
    """Execution mode for heavy tasks."""
    LOCAL = "local"
    HYBRID = "hybrid"
    REMOTE = "remote"
    WARNING = "warning"  # Can run but may be slow


@dataclass
class HardwareReport:
    """Complete hardware capability report."""
    # CPU
    cpu_count: int
    cpu_count_logical: int
    cpu_freq_mhz: Optional[float]
    cpu_percent: float
    
    # Memory
    ram_total_gb: float
    ram_available_gb: float
    ram_percent_used: float
    
    # Disk
    disk_total_gb: float
    disk_free_gb: float
    disk_percent_used: float
    
    # GPU
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    gpu_memory_free_gb: Optional[float]
    cuda_version: Optional[str]
    
    # Platform
    os_name: str
    os_version: str
    python_version: str
    
    # Scores (computed)
    cpu_score: int = 0  # 0-100
    gpu_score: int = 0  # 0-100
    memory_score: int = 0  # 0-100
    overall_score: int = 0  # 0-100
    
    def to_dict(self) -> Dict:
        return {
            "cpu": {
                "cores": self.cpu_count,
                "logical_cores": self.cpu_count_logical,
                "frequency_mhz": self.cpu_freq_mhz,
                "usage_percent": self.cpu_percent,
                "score": self.cpu_score
            },
            "memory": {
                "total_gb": round(self.ram_total_gb, 2),
                "available_gb": round(self.ram_available_gb, 2),
                "used_percent": self.ram_percent_used,
                "score": self.memory_score
            },
            "disk": {
                "total_gb": round(self.disk_total_gb, 2),
                "free_gb": round(self.disk_free_gb, 2),
                "used_percent": self.disk_percent_used
            },
            "gpu": {
                "available": self.gpu_available,
                "name": self.gpu_name,
                "memory_gb": self.gpu_memory_gb,
                "memory_free_gb": self.gpu_memory_free_gb,
                "cuda_version": self.cuda_version,
                "score": self.gpu_score
            },
            "platform": {
                "os": self.os_name,
                "os_version": self.os_version,
                "python_version": self.python_version
            },
            "overall_score": self.overall_score
        }


@dataclass
class TimeEstimate:
    """Estimated execution time for a task."""
    min_seconds: int
    max_seconds: int
    confidence: float  # 0-1
    factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "min_display": self._format_time(self.min_seconds),
            "max_display": self._format_time(self.max_seconds),
            "confidence": self.confidence,
            "factors": self.factors
        }
    
    def _format_time(self, seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            return f"{hours}h {mins}m"


@dataclass
class Suggestion:
    """Optimization suggestion for user."""
    type: str  # "info", "warning", "action"
    title: str
    description: str
    action: Optional[str] = None  # Action the user can take
    impact: Optional[str] = None  # What improvement to expect


@dataclass
class ExecutionDecision:
    """Decision about how to execute a task."""
    mode: ExecutionMode
    can_proceed: bool
    warnings: List[str] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    estimated_time: Optional[TimeEstimate] = None
    reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.value,
            "can_proceed": self.can_proceed,
            "warnings": self.warnings,
            "suggestions": [
                {
                    "type": s.type,
                    "title": s.title,
                    "description": s.description,
                    "action": s.action,
                    "impact": s.impact
                } for s in self.suggestions
            ],
            "estimated_time": self.estimated_time.to_dict() if self.estimated_time else None,
            "reason": self.reason
        }


class TaskOrchestrator:
    """
    Central decision-maker for all heavy tasks.
    Runs BEFORE any engine execution.
    """
    
    # Thresholds for warnings
    MIN_RAM_GB = 4.0
    MIN_FREE_DISK_GB = 5.0
    LARGE_DATASET_THRESHOLD = 1000  # images
    VERY_LARGE_DATASET_THRESHOLD = 10000
    LARGE_IMAGE_THRESHOLD = 2048  # pixels (width or height)
    
    def __init__(self):
        self._hardware_cache: Optional[HardwareReport] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 60  # Refresh hardware info every minute
    
    def get_hardware_info(self, force_refresh: bool = False) -> HardwareReport:
        """
        Auto-detect hardware capabilities.
        Results are cached to avoid repeated expensive checks.
        """
        now = datetime.now()
        
        if (
            not force_refresh 
            and self._hardware_cache 
            and self._cache_time 
            and (now - self._cache_time).seconds < self._cache_ttl_seconds
        ):
            return self._hardware_cache
        
        # CPU Info
        cpu_count = psutil.cpu_count(logical=False) or 1
        cpu_count_logical = psutil.cpu_count(logical=True) or 1
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else None
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory Info
        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024 ** 3)
        ram_available_gb = memory.available / (1024 ** 3)
        ram_percent_used = memory.percent
        
        # Disk Info (use home directory drive)
        home = os.path.expanduser("~")
        disk = psutil.disk_usage(home)
        disk_total_gb = disk.total / (1024 ** 3)
        disk_free_gb = disk.free / (1024 ** 3)
        disk_percent_used = disk.percent
        
        # GPU Info
        gpu_available = False
        gpu_name = None
        gpu_memory_gb = None
        gpu_memory_free_gb = None
        cuda_version = None
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            try:
                gpu_memory_free_gb = (
                    torch.cuda.get_device_properties(0).total_memory - 
                    torch.cuda.memory_allocated(0)
                ) / (1024 ** 3)
            except:
                gpu_memory_free_gb = gpu_memory_gb
            cuda_version = torch.version.cuda
        
        # Platform Info
        os_name = platform.system()
        os_version = platform.release()
        python_version = sys.version.split()[0]
        
        # Calculate scores
        cpu_score = min(100, int((cpu_count * 15) + (cpu_count_logical * 5)))
        memory_score = min(100, int(ram_total_gb * 6))
        gpu_score = 100 if gpu_available else 0
        
        overall_score = int(
            (cpu_score * 0.3) + 
            (memory_score * 0.3) + 
            (gpu_score * 0.4)
        )
        
        report = HardwareReport(
            cpu_count=cpu_count,
            cpu_count_logical=cpu_count_logical,
            cpu_freq_mhz=cpu_freq_mhz,
            cpu_percent=cpu_percent,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            ram_percent_used=ram_percent_used,
            disk_total_gb=disk_total_gb,
            disk_free_gb=disk_free_gb,
            disk_percent_used=disk_percent_used,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            gpu_memory_free_gb=gpu_memory_free_gb,
            cuda_version=cuda_version,
            os_name=os_name,
            os_version=os_version,
            python_version=python_version,
            cpu_score=cpu_score,
            memory_score=memory_score,
            gpu_score=gpu_score,
            overall_score=overall_score
        )
        
        self._hardware_cache = report
        self._cache_time = now
        
        return report
    
    def assess_task(
        self,
        task_type: str,
        dataset_size: int,
        image_resolution: Optional[tuple] = None,
        project_profile: str = "general",
        config: Optional[Dict] = None
    ) -> ExecutionDecision:
        """
        Assess a task and return execution decision.
        
        Args:
            task_type: "training", "inference", "3d_reconstruction", "change_detection"
            dataset_size: Number of images
            image_resolution: (width, height) of images
            project_profile: Industry profile ID
            config: Task configuration (epochs, batch_size, etc.)
        
        Returns:
            ExecutionDecision with mode, warnings, and suggestions
        """
        hardware = self.get_hardware_info()
        warnings = []
        suggestions = []
        
        # Check industry profile constraints
        profile_constraints = self._get_profile_constraints(project_profile)
        if profile_constraints.get("offline_only"):
            # Force local execution for defense/health
            if task_type == "training" and dataset_size > self.LARGE_DATASET_THRESHOLD:
                warnings.append(
                    f"Large dataset ({dataset_size} images) with offline-only profile. "
                    "Training may be slow on local hardware."
                )
        
        # Check RAM
        if hardware.ram_available_gb < self.MIN_RAM_GB:
            warnings.append(
                f"Low available RAM ({hardware.ram_available_gb:.1f}GB). "
                "Consider closing other applications."
            )
            suggestions.append(Suggestion(
                type="warning",
                title="Low Memory",
                description=f"Only {hardware.ram_available_gb:.1f}GB RAM available",
                action="Close other applications or reduce batch size",
                impact="Faster training, fewer crashes"
            ))
        
        # Check disk space
        if hardware.disk_free_gb < self.MIN_FREE_DISK_GB:
            warnings.append(
                f"Low disk space ({hardware.disk_free_gb:.1f}GB free). "
                "May not have enough space for checkpoints."
            )
        
        # Determine execution mode
        mode = self._determine_mode(
            task_type, dataset_size, hardware, 
            profile_constraints, image_resolution
        )
        
        # Generate suggestions based on task
        suggestions.extend(
            self._generate_suggestions(task_type, dataset_size, hardware, config)
        )
        
        # Estimate execution time
        estimated_time = self.estimate_execution_time(
            task_type, dataset_size, hardware, config
        )
        
        can_proceed = True
        if mode == ExecutionMode.WARNING:
            can_proceed = True  # Can proceed but with warnings
        
        reason = self._get_mode_reason(mode, hardware, dataset_size)
        
        return ExecutionDecision(
            mode=mode,
            can_proceed=can_proceed,
            warnings=warnings,
            suggestions=suggestions,
            estimated_time=estimated_time,
            reason=reason
        )
    
    def _determine_mode(
        self,
        task_type: str,
        dataset_size: int,
        hardware: HardwareReport,
        profile_constraints: Dict,
        image_resolution: Optional[tuple]
    ) -> ExecutionMode:
        """Determine the best execution mode."""
        
        # Profile constraints take priority
        if profile_constraints.get("offline_only"):
            if dataset_size > self.VERY_LARGE_DATASET_THRESHOLD:
                return ExecutionMode.WARNING
            return ExecutionMode.LOCAL
        
        if not profile_constraints.get("cloud_allowed", True):
            if dataset_size > self.VERY_LARGE_DATASET_THRESHOLD:
                return ExecutionMode.WARNING
            return ExecutionMode.LOCAL
        
        # GPU available = local is good
        if hardware.gpu_available:
            if dataset_size <= self.VERY_LARGE_DATASET_THRESHOLD:
                return ExecutionMode.LOCAL
            else:
                return ExecutionMode.HYBRID
        
        # No GPU
        if task_type == "training":
            if dataset_size <= 500:
                return ExecutionMode.LOCAL
            elif dataset_size <= self.LARGE_DATASET_THRESHOLD:
                return ExecutionMode.WARNING
            else:
                return ExecutionMode.HYBRID
        
        if task_type == "3d_reconstruction":
            if dataset_size <= 100:
                return ExecutionMode.LOCAL
            elif dataset_size <= 500:
                return ExecutionMode.WARNING
            else:
                return ExecutionMode.HYBRID
        
        return ExecutionMode.LOCAL
    
    def _get_profile_constraints(self, profile_id: str) -> Dict:
        """Get constraints for an industry profile."""
        # These will be loaded from config files later
        # For now, hardcode common constraints
        profiles = {
            "defense": {
                "offline_only": True,
                "cloud_allowed": False,
                "encryption_required": True
            },
            "health": {
                "offline_only": True,
                "cloud_allowed": False,
                "anonymization_required": True
            },
            "mining": {
                "cloud_allowed": True,
                "offload_allowed": True
            },
            "general": {
                "cloud_allowed": True,
                "offload_allowed": True
            }
        }
        return profiles.get(profile_id, profiles["general"])
    
    def _generate_suggestions(
        self,
        task_type: str,
        dataset_size: int,
        hardware: HardwareReport,
        config: Optional[Dict]
    ) -> List[Suggestion]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # GPU suggestion
        if not hardware.gpu_available and task_type == "training":
            suggestions.append(Suggestion(
                type="info",
                title="No GPU Detected",
                description="Training will use CPU, which is slower",
                action="Install CUDA-compatible GPU for 10-50x faster training",
                impact="Major speed improvement"
            ))
        
        # Batch size suggestion
        if config and config.get("batch_size", 16) > 16 and hardware.ram_available_gb < 8:
            suggestions.append(Suggestion(
                type="action",
                title="Reduce Batch Size",
                description=f"Batch size {config.get('batch_size')} may be too high for available RAM",
                action="Reduce batch size to 8 or 4",
                impact="Prevents out-of-memory errors"
            ))
        
        # Large dataset suggestion
        if dataset_size > self.LARGE_DATASET_THRESHOLD:
            suggestions.append(Suggestion(
                type="info",
                title="Large Dataset",
                description=f"{dataset_size} images will take longer to process",
                action="Consider using a subset for initial experiments",
                impact="Faster iteration cycles"
            ))
        
        return suggestions
    
    def estimate_execution_time(
        self,
        task_type: str,
        dataset_size: int,
        hardware: HardwareReport,
        config: Optional[Dict] = None
    ) -> TimeEstimate:
        """Estimate execution time for a task."""
        factors = []
        
        # Base time per image (seconds)
        if task_type == "training":
            # Training: depends on epochs, batch size, GPU
            epochs = config.get("epochs", 50) if config else 50
            batch_size = config.get("batch_size", 16) if config else 16
            
            if hardware.gpu_available:
                time_per_epoch = dataset_size * 0.05  # ~50ms per image with GPU
                factors.append("GPU acceleration")
            else:
                time_per_epoch = dataset_size * 0.5  # ~500ms per image without GPU
                factors.append("CPU-only (slower)")
            
            base_time = time_per_epoch * epochs
            
        elif task_type == "inference":
            if hardware.gpu_available:
                time_per_image = 0.02  # 20ms per image
                factors.append("GPU acceleration")
            else:
                time_per_image = 0.2  # 200ms per image
                factors.append("CPU-only")
            
            base_time = dataset_size * time_per_image
            
        elif task_type == "3d_reconstruction":
            # SfM is very compute-intensive
            base_time = dataset_size * 5  # ~5 seconds per image
            factors.append("CPU-based SfM")
            
        else:
            base_time = dataset_size * 1  # Default 1 second per image
        
        # Add variance
        min_time = int(base_time * 0.8)
        max_time = int(base_time * 1.5)
        
        return TimeEstimate(
            min_seconds=min_time,
            max_seconds=max_time,
            confidence=0.7,
            factors=factors
        )
    
    def _get_mode_reason(
        self,
        mode: ExecutionMode,
        hardware: HardwareReport,
        dataset_size: int
    ) -> str:
        """Get human-readable reason for the execution mode."""
        if mode == ExecutionMode.LOCAL:
            if hardware.gpu_available:
                return f"GPU detected ({hardware.gpu_name}). Running locally for best performance."
            else:
                return f"Dataset size ({dataset_size}) is suitable for local CPU execution."
        
        elif mode == ExecutionMode.HYBRID:
            return (
                f"Large dataset ({dataset_size} images) may benefit from "
                "hybrid execution with cloud acceleration."
            )
        
        elif mode == ExecutionMode.REMOTE:
            return "Task requires cloud resources for efficient execution."
        
        elif mode == ExecutionMode.WARNING:
            return (
                f"Task can proceed locally but may be slow. "
                f"Dataset has {dataset_size} images without GPU acceleration."
            )
        
        return ""


# Singleton instance
task_orchestrator = TaskOrchestrator()
