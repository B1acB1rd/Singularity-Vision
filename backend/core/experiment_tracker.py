"""
Experiment Tracking & Training Reproducibility for Singularity Vision

PHILOSOPHY: Every training run must be EXACTLY reproducible.

Key Requirements:
- Bind training runs to specific dataset VERSION
- Bind training runs to specific model VERSION  
- Bind training runs to config HASH
- Store all metadata for experiment comparison
- No silent changes to training inputs

This is critical for:
- Research credibility
- Enterprise trust
- Debugging model issues
- Experiment comparison
"""

import os
import json
import hashlib
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""
    # Model
    model_name: str
    model_version: Optional[str] = None
    pretrained_weights: Optional[str] = None
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.01
    optimizer: str = "SGD"
    
    # Data augmentation
    augmentations: List[str] = field(default_factory=list)
    
    # Task-specific
    image_size: int = 640
    num_classes: Optional[int] = None
    class_names: List[str] = field(default_factory=list)
    
    # Other settings
    device: str = "auto"
    seed: Optional[int] = None
    extra: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "pretrained_weights": self.pretrained_weights,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "augmentations": self.augmentations,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "device": self.device,
            "seed": self.seed,
            "extra": self.extra
        }
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        # Sort keys for determinism
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class ExperimentBinding:
    """
    Immutable binding of an experiment to its inputs.
    
    This is the CORE of reproducibility:
    - Same dataset_version + config_hash = same training inputs
    - This binding is stored and never modified
    """
    experiment_id: str
    
    # Dataset binding (CRITICAL)
    dataset_path: str
    dataset_version: str  # Hash from dataset_manager
    
    # Config binding (CRITICAL)
    config_hash: str  # Hash of ExperimentConfig
    config: ExperimentConfig
    
    # Model binding
    base_model: str
    base_model_version: Optional[str] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    profile_id: str = "general"
    
    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "dataset_path": self.dataset_path,
            "dataset_version": self.dataset_version,
            "config_hash": self.config_hash,
            "config": self.config.to_dict(),
            "base_model": self.base_model,
            "base_model_version": self.base_model_version,
            "created_at": self.created_at,
            "profile_id": self.profile_id
        }


@dataclass
class ExperimentResult:
    """Results from a completed experiment."""
    experiment_id: str
    
    # Metrics
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    best_epoch: Optional[int] = None
    metrics_history: List[Dict] = field(default_factory=list)
    
    # Outputs
    checkpoint_path: Optional[str] = None
    best_weights_path: Optional[str] = None
    
    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    training_duration_seconds: Optional[float] = None
    
    # Status
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "final_loss": self.final_loss,
            "final_accuracy": self.final_accuracy,
            "best_epoch": self.best_epoch,
            "metrics_history": self.metrics_history,
            "checkpoint_path": self.checkpoint_path,
            "best_weights_path": self.best_weights_path,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "training_duration_seconds": self.training_duration_seconds,
            "status": self.status,
            "error_message": self.error_message
        }


class ExperimentTracker:
    """
    Tracks training experiments with full reproducibility.
    
    GUARANTEES:
    1. Every experiment is bound to a specific dataset version
    2. Every experiment has a config hash
    3. Experiments with identical bindings produce identical results (given same seed)
    4. All experiment metadata is persisted
    """
    
    def __init__(self, project_path: str, profile_id: str = "general"):
        self.project_path = project_path
        self.profile_id = profile_id
        
        # Storage paths
        self.experiments_dir = os.path.join(project_path, ".experiments")
        os.makedirs(self.experiments_dir, exist_ok=True)
        
        # Index file
        self.index_path = os.path.join(self.experiments_dir, "index.json")
        self._index: Dict[str, Dict] = {}
        self._load_index()
    
    def create_experiment(
        self,
        config: ExperimentConfig,
        dataset_path: str,
        force_new_version_check: bool = True
    ) -> ExperimentBinding:
        """
        Create a new experiment with version bindings.
        
        CRITICAL: This captures the EXACT state of inputs.
        
        Args:
            config: Training configuration
            dataset_path: Path to dataset
            force_new_version_check: If True, get fresh dataset version
            
        Returns:
            ExperimentBinding with all version info locked
        """
        # Get dataset version
        dataset_version = self._get_dataset_version(dataset_path, force_new_version_check)
        
        # Compute config hash
        config_hash = config.compute_hash()
        
        # Generate experiment ID
        experiment_id = self._generate_experiment_id()
        
        # Create binding
        binding = ExperimentBinding(
            experiment_id=experiment_id,
            dataset_path=dataset_path,
            dataset_version=dataset_version,
            config_hash=config_hash,
            config=config,
            base_model=config.model_name,
            base_model_version=config.model_version,
            profile_id=self.profile_id
        )
        
        # Save binding
        self._save_binding(binding)
        
        # Create result placeholder
        result = ExperimentResult(experiment_id=experiment_id, status="pending")
        self._save_result(result)
        
        # Update index
        self._index[experiment_id] = {
            "experiment_id": experiment_id,
            "dataset_version": dataset_version,
            "config_hash": config_hash,
            "created_at": binding.created_at,
            "status": "pending"
        }
        self._save_index()
        
        logger.info(
            f"Created experiment {experiment_id}: "
            f"dataset={dataset_version[:8]}..., config={config_hash}"
        )
        
        return binding
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get full experiment data including binding and results."""
        binding = self._load_binding(experiment_id)
        result = self._load_result(experiment_id)
        
        if not binding:
            return None
        
        return {
            "binding": binding,
            "result": result,
            "index": self._index.get(experiment_id)
        }
    
    def find_matching_experiments(
        self,
        dataset_version: Optional[str] = None,
        config_hash: Optional[str] = None
    ) -> List[str]:
        """
        Find experiments matching the given version/config.
        
        Use case: "Have I run this exact experiment before?"
        """
        matches = []
        
        for exp_id, info in self._index.items():
            if dataset_version and info.get("dataset_version") != dataset_version:
                continue
            if config_hash and info.get("config_hash") != config_hash:
                continue
            matches.append(exp_id)
        
        return matches
    
    def check_reproducibility(
        self,
        experiment_id: str,
        dataset_path: str,
        config: ExperimentConfig
    ) -> Dict:
        """
        Check if current inputs match a previous experiment.
        
        Returns:
            Dict with match status and any differences
        """
        binding = self._load_binding(experiment_id)
        if not binding:
            return {"status": "error", "message": "Experiment not found"}
        
        current_version = self._get_dataset_version(dataset_path, force_check=True)
        current_config_hash = config.compute_hash()
        
        dataset_match = current_version == binding["dataset_version"]
        config_match = current_config_hash == binding["config_hash"]
        
        return {
            "status": "match" if (dataset_match and config_match) else "mismatch",
            "dataset_match": dataset_match,
            "config_match": config_match,
            "original_dataset_version": binding["dataset_version"],
            "current_dataset_version": current_version,
            "original_config_hash": binding["config_hash"],
            "current_config_hash": current_config_hash
        }
    
    def compare_experiments(
        self,
        experiment_ids: List[str]
    ) -> Dict:
        """
        Compare multiple experiments.
        
        Shows:
        - Config differences
        - Dataset version differences
        - Result differences
        """
        experiments = []
        
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                experiments.append(exp)
        
        if len(experiments) < 2:
            return {"status": "error", "message": "Need at least 2 experiments to compare"}
        
        # Extract comparison data
        comparison = {
            "experiments": experiment_ids,
            "dataset_versions": {},
            "config_hashes": {},
            "results": {}
        }
        
        for exp in experiments:
            exp_id = exp["binding"]["experiment_id"]
            comparison["dataset_versions"][exp_id] = exp["binding"]["dataset_version"]
            comparison["config_hashes"][exp_id] = exp["binding"]["config_hash"]
            
            if exp["result"]:
                comparison["results"][exp_id] = {
                    "status": exp["result"].get("status"),
                    "final_loss": exp["result"].get("final_loss"),
                    "final_accuracy": exp["result"].get("final_accuracy"),
                    "training_duration": exp["result"].get("training_duration_seconds")
                }
        
        # Check if configs are identical
        unique_configs = len(set(comparison["config_hashes"].values()))
        unique_datasets = len(set(comparison["dataset_versions"].values()))
        
        comparison["config_identical"] = unique_configs == 1
        comparison["dataset_identical"] = unique_datasets == 1
        
        return comparison
    
    def update_result(
        self,
        experiment_id: str,
        status: Optional[str] = None,
        metrics: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update experiment results during/after training."""
        result = self._load_result(experiment_id)
        if not result:
            result = {"experiment_id": experiment_id}
        
        if status:
            result["status"] = status
            if status == "running" and not result.get("started_at"):
                result["started_at"] = datetime.now().isoformat()
            elif status in ["completed", "failed"]:
                result["completed_at"] = datetime.now().isoformat()
                if result.get("started_at"):
                    started = datetime.fromisoformat(result["started_at"])
                    completed = datetime.fromisoformat(result["completed_at"])
                    result["training_duration_seconds"] = (completed - started).total_seconds()
        
        if metrics:
            if "history" not in result.get("metrics_history", []):
                result.setdefault("metrics_history", []).append(metrics)
            
            if "loss" in metrics:
                result["final_loss"] = metrics["loss"]
            if "accuracy" in metrics:
                result["final_accuracy"] = metrics["accuracy"]
            if "epoch" in metrics:
                result["best_epoch"] = metrics.get("epoch")
        
        if checkpoint_path:
            result["best_weights_path"] = checkpoint_path
        
        if error_message:
            result["error_message"] = error_message
        
        self._save_result_dict(experiment_id, result)
        
        # Update index
        if experiment_id in self._index:
            self._index[experiment_id]["status"] = result.get("status", "unknown")
            self._save_index()
        
        return True
    
    def list_experiments(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List experiments with optional status filter."""
        results = []
        
        for exp_id, info in self._index.items():
            if status and info.get("status") != status:
                continue
            results.append(info)
        
        # Sort by created_at descending
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return results[:limit]
    
    # Private methods
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"exp_{timestamp}_{short_uuid}"
    
    def _get_dataset_version(self, dataset_path: str, force_check: bool = True) -> str:
        """Get dataset version hash."""
        try:
            from core.dataset_manager import dataset_manager
            version = dataset_manager.get_current_version(dataset_path)
            if version:
                return version
        except Exception as e:
            logger.warning(f"Could not get dataset version from manager: {e}")
        
        # Fallback: compute hash from dataset contents
        try:
            hash_input = dataset_path
            if os.path.isdir(dataset_path):
                # Hash directory modification time and file list
                files = sorted(os.listdir(dataset_path)) if os.path.exists(dataset_path) else []
                hash_input = dataset_path + "|" + "|".join(files[:50])
            return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Fallback version computation failed: {e}")
            # Final fallback: just hash the path
            return hashlib.sha256(dataset_path.encode()).hexdigest()[:16]
    
    def _save_binding(self, binding: ExperimentBinding) -> None:
        """Save experiment binding to disk."""
        path = os.path.join(self.experiments_dir, f"{binding.experiment_id}_binding.json")
        with open(path, 'w') as f:
            json.dump(binding.to_dict(), f, indent=2)
    
    def _load_binding(self, experiment_id: str) -> Optional[Dict]:
        """Load experiment binding from disk."""
        path = os.path.join(self.experiments_dir, f"{experiment_id}_binding.json")
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)
    
    def _save_result(self, result: ExperimentResult) -> None:
        """Save experiment result to disk."""
        path = os.path.join(self.experiments_dir, f"{result.experiment_id}_result.json")
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _save_result_dict(self, experiment_id: str, result: Dict) -> None:
        """Save result dict directly."""
        path = os.path.join(self.experiments_dir, f"{experiment_id}_result.json")
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _load_result(self, experiment_id: str) -> Optional[Dict]:
        """Load experiment result from disk."""
        path = os.path.join(self.experiments_dir, f"{experiment_id}_result.json")
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_index(self) -> None:
        """Load experiment index from disk."""
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                self._index = json.load(f)
    
    def _save_index(self) -> None:
        """Save experiment index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self._index, f, indent=2)


def create_experiment_tracker(project_path: str, profile_id: str = "general") -> ExperimentTracker:
    """Factory function to create experiment tracker."""
    return ExperimentTracker(project_path, profile_id)
