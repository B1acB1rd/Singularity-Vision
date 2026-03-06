import os
import yaml
import threading
import uuid
import time
from typing import Dict, Optional
from datetime import datetime
from ultralytics import YOLO


def resolve_model_path(model_name: str, project_path: str = None) -> str:
    """
    Resolve model path, checking project-local models folder first.
    
    Args:
        model_name: Model name (e.g., 'yolov8n.pt')
        project_path: Path to project directory (optional)
        
    Returns:
        Full path if found in project, otherwise returns the model_name 
        (which will trigger Ultralytics to download it)
    """
    # Check if model_name is already a full path
    if os.path.exists(model_name):
        return model_name
    
    # Check project-local models folder if project_path provided
    if project_path:
        project_models_dir = os.path.join(project_path, "models")
        os.makedirs(project_models_dir, exist_ok=True)
        
        local_path = os.path.join(project_models_dir, model_name)
        if os.path.exists(local_path):
            print(f"[TrainingManager] Using project model: {local_path}")
            return local_path
    
    # Model not found locally, Ultralytics will download to default cache
    print(f"[TrainingManager] Model not in project, will use Ultralytics default: {model_name}")
    return model_name


class TrainingManager:
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.active_threads: Dict[str, threading.Thread] = {}

    def start_training(self, project_path: str, model_name: str, config: Dict) -> str:
        """
        Start a YOLO training job in a separate thread.
        
        Args:
            project_path: Path to the project directory
            model_name: Name of the model to use (e.g., 'yolov8n.pt')
            config: Training configuration (epochs, batch_size, imgsz, etc.)
        
        Returns:
            job_id: Unique ID for the training job
        """
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        self.jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": config.get('epochs', 10),
            "logs": [],
            "metrics": {},
            "start_time": datetime.now().isoformat(),
            "error": None
        }

        # create data.yaml
        try:
            yaml_path = self._create_data_yaml(project_path)
        except Exception as e:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = f"Failed to create data.yaml: {str(e)}"
            return job_id

        # Start training thread
        thread = threading.Thread(
            target=self._run_training,
            args=(job_id, project_path, model_name, yaml_path, config)
        )
        self.active_threads[job_id] = thread
        thread.start()

        return job_id

    def stop_training(self, job_id: str):
        """Stop a training job (if possible)"""
        if job_id in self.jobs:
            # YOLO doesn't have a clean "stop" from Python API easily without callbacks
            # For now, we'll mark as stopped and hopefully we can kill the thread or logic
            # In reality, stopping a thread is hard. 
            # Ideally we'd use a custom callback in YOLO to check a flag.
            self.jobs[job_id]["status"] = "stopping"
            # We will use a flag file or simple check in callbacks if we implement them
            # For MVP, we might essentially just let it finish or kill the app :(
            # Let's mark it so the UI knows
            pass

    def get_status(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)

    def get_device_info(self) -> Dict:
        """
        Auto-detect available compute devices (CPU/GPU).
        
        Returns:
            Dict with device info and recommended device
        """
        import torch
        
        info = {
            "cpu": {
                "available": True,
                "name": "CPU",
                "recommended": False
            },
            "cuda": {
                "available": False,
                "devices": [],
                "recommended": False
            },
            "mps": {
                "available": False,
                "recommended": False
            },
            "recommended_device": "cpu"
        }
        
        # Check CUDA
        if torch.cuda.is_available():
            info["cuda"]["available"] = True
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                gpu_props = torch.cuda.get_device_properties(i)
                info["cuda"]["devices"].append({
                    "id": i,
                    "name": gpu_props.name,
                    "memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                })
            
            info["cuda"]["recommended"] = True
            info["recommended_device"] = "cuda"
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["mps"]["available"] = True
            if not info["cuda"]["available"]:
                info["mps"]["recommended"] = True
                info["recommended_device"] = "mps"
        
        return info

    def resume_training(self, project_path: str, run_name: str, config: Dict) -> str:
        """
        Resume training from the last checkpoint.
        
        Args:
            project_path: Path to the project directory
            run_name: Name of the training run to resume (e.g., 'train_abc123')
            config: Training configuration (only epochs matters - additional epochs to train)
            
        Returns:
            job_id: Unique ID for the resumed training job
        """
        job_id = str(uuid.uuid4())
        
        # Find the last checkpoint
        runs_dir = os.path.join(project_path, "runs")
        last_pt = None
        
        # Search in detect, classify, segment
        for task_dir in ["detect", "classify", "segment"]:
            weights_path = os.path.join(runs_dir, task_dir, run_name, "weights", "last.pt")
            if os.path.exists(weights_path):
                last_pt = weights_path
                break
        
        if not last_pt:
            self.jobs[job_id] = {
                "id": job_id,
                "status": "failed",
                "error": f"No checkpoint found for run: {run_name}",
                "logs": []
            }
            return job_id
        
        # Initialize job status
        self.jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": config.get('epochs', 10),
            "logs": [f"Resuming from checkpoint: {last_pt}"],
            "metrics": {},
            "start_time": datetime.now().isoformat(),
            "resumed_from": run_name,
            "error": None
        }
        
        # Start resume thread
        thread = threading.Thread(
            target=self._run_resume_training,
            args=(job_id, last_pt, config)
        )
        self.active_threads[job_id] = thread
        thread.start()
        
        return job_id

    def _run_resume_training(self, job_id: str, checkpoint_path: str, config: Dict):
        """Execute training resumption"""
        try:
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["logs"].append("Loading checkpoint...")
            
            # Load model from checkpoint
            model = YOLO(checkpoint_path)
            
            # Callback for progress
            def on_train_epoch_end(trainer):
                current = trainer.epoch + 1
                total = trainer.epochs
                self.jobs[job_id]["current_epoch"] = current
                self.jobs[job_id]["progress"] = (current / total) * 100
                
                if trainer.metrics:
                    self.jobs[job_id]["metrics"] = {
                        "map50": float(trainer.metrics.get("metrics/mAP50(B)", 0)),
                        "map5095": float(trainer.metrics.get("metrics/mAP50-95(B)", 0))
                    }
                
                self.jobs[job_id]["logs"].append(f"Epoch {current}/{total} completed.")
            
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            self.jobs[job_id]["logs"].append("Resuming training...")
            
            # Auto-select device
            device_info = self.get_device_info()
            device = config.get('device', device_info['recommended_device'])
            
            # Resume training
            results = model.train(
                resume=True,
                epochs=config.get('epochs', 10),
                device=device
            )
            
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 100
            self.jobs[job_id]["logs"].append("Training resumed and completed.")
            
        except Exception as e:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["logs"].append(f"Error: {str(e)}")
            print(f"Resume Training Error: {e}")

    def _create_data_yaml(self, project_path: str) -> str:
        """Generate data.yaml for YOLO"""
        
        # Check for autosplit files
        train_txt = os.path.join(project_path, 'autosplit_train.txt')
        val_txt = os.path.join(project_path, 'autosplit_val.txt')
        test_txt = os.path.join(project_path, 'autosplit_test.txt') # Optional

        if not os.path.exists(train_txt) or not os.path.exists(val_txt):
            raise FileNotFoundError("Dataset split files not found. Please split the dataset first.")

        # Read classes from config.json
        config_path = os.path.join(project_path, 'config.json')
        if not os.path.exists(config_path):
             raise FileNotFoundError("Project config.json not found.")
        
        import json
        with open(config_path, 'r') as f:
            proj_config = json.load(f)
            
        classes = proj_config.get('classes', [])
        names = {c['id']: c['name'] for c in classes}

        data_config = {
            'path': project_path, # Root
            'train': 'autosplit_train.txt',
            'val': 'autosplit_val.txt',
            'test': 'autosplit_test.txt' if os.path.exists(test_txt) else None,
            'names': names
        }

        yaml_path = os.path.join(project_path, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f)
            
        return yaml_path

    def start_segmentation_training(self, project_path: str, model_name: str, config: Dict) -> str:
        """
        Start an instance segmentation training job.
        
        Uses YOLO format with polygon annotations:
        - Label format: class_id x1 y1 x2 y2 ... xn yn (normalized polygon points)
        
        Args:
            project_path: Path to project directory
            model_name: Segmentation model (e.g., 'yolov8n-seg.pt')
            config: Training config (epochs, batch_size, imgsz)
            
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        self.jobs[job_id] = {
            "id": job_id,
            "task_type": "segmentation",
            "status": "pending",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": config.get('epochs', 10),
            "logs": [],
            "metrics": {},
            "start_time": datetime.now().isoformat(),
            "error": None
        }
        
        # Create data.yaml
        try:
            yaml_path = self._create_data_yaml(project_path)
        except Exception as e:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = f"Failed to create data.yaml: {str(e)}"
            return job_id
        
        # Start training thread
        thread = threading.Thread(
            target=self._run_segmentation_training,
            args=(job_id, project_path, model_name, yaml_path, config)
        )
        self.active_threads[job_id] = thread
        thread.start()
        
        return job_id

    def _run_segmentation_training(self, job_id: str, project_path: str, model_name: str, yaml_path: str, config: Dict):
        """Execute segmentation training job"""
        try:
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["logs"].append("Initializing segmentation model...")
            
            # Load segmentation model
            model = YOLO(model_name)
            
            # Auto-select device
            device_info = self.get_device_info()
            device = config.get('device', device_info['recommended_device'])
            
            # Define callback
            def on_train_epoch_end(trainer):
                current = trainer.epoch + 1
                total = trainer.epochs
                self.jobs[job_id]["current_epoch"] = current
                self.jobs[job_id]["progress"] = (current / total) * 100
                
                # Segmentation metrics
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    self.jobs[job_id]["metrics"] = {
                        "map50": float(trainer.metrics.get("metrics/mAP50(B)", 0)),
                        "map5095": float(trainer.metrics.get("metrics/mAP50-95(B)", 0)),
                        "map50_mask": float(trainer.metrics.get("metrics/mAP50(M)", 0)),
                        "map5095_mask": float(trainer.metrics.get("metrics/mAP50-95(M)", 0))
                    }
                
                self.jobs[job_id]["logs"].append(f"Epoch {current}/{total} completed.")
            
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            self.jobs[job_id]["logs"].append("Starting segmentation training...")
            
            # Train segmentation model
            results = model.train(
                data=yaml_path,
                project=os.path.join(project_path, "runs", "segment"),
                name=f"train_{job_id}",
                epochs=config.get('epochs', 10),
                batch=config.get('batch_size', 16),
                imgsz=config.get('imgsz', 640),
                device=device,
                exist_ok=True
            )
            
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 100
            self.jobs[job_id]["logs"].append("Segmentation training completed.")
            
        except Exception as e:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["logs"].append(f"Error: {str(e)}")
            print(f"Segmentation Training Error: {e}")

    def start_classification_training(self, project_path: str, model_name: str, config: Dict) -> str:
        """
        Start an image classification training job.
        
        Classification uses folder-based class structure:
        datasets/
          train/
            class1/
              img1.jpg
            class2/
              img2.jpg
          val/
            class1/
            class2/
        
        Args:
            project_path: Path to project directory
            model_name: Classification model (e.g., 'yolov8n-cls.pt')
            config: Training config (epochs, batch_size, imgsz)
            
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        self.jobs[job_id] = {
            "id": job_id,
            "task_type": "classification",
            "status": "pending",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": config.get('epochs', 10),
            "logs": [],
            "metrics": {},
            "start_time": datetime.now().isoformat(),
            "error": None
        }
        
        # Validate classification dataset structure
        train_dir = os.path.join(project_path, 'datasets', 'train')
        val_dir = os.path.join(project_path, 'datasets', 'val')
        
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = "Classification requires datasets/train and datasets/val folders with class subfolders"
            return job_id
        
        # Start training thread
        thread = threading.Thread(
            target=self._run_classification_training,
            args=(job_id, project_path, model_name, config)
        )
        self.active_threads[job_id] = thread
        thread.start()
        
        return job_id

    def _run_classification_training(self, job_id: str, project_path: str, model_name: str, config: Dict):
        """Execute classification training job"""
        try:
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["logs"].append("Initializing classification model...")
            
            # Load classification model
            model = YOLO(model_name)
            
            # Define callback
            def on_train_epoch_end(trainer):
                current = trainer.epoch + 1
                total = trainer.epochs
                self.jobs[job_id]["current_epoch"] = current
                self.jobs[job_id]["progress"] = (current / total) * 100
                
                # Classification metrics
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    self.jobs[job_id]["metrics"] = {
                        "top1_accuracy": float(trainer.metrics.get("metrics/accuracy_top1", 0)),
                        "top5_accuracy": float(trainer.metrics.get("metrics/accuracy_top5", 0)),
                        "loss": float(trainer.loss) if hasattr(trainer, 'loss') else 0
                    }
                
                self.jobs[job_id]["logs"].append(f"Epoch {current}/{total} completed.")
            
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            self.jobs[job_id]["logs"].append("Starting classification training...")
            
            # Train classification model
            results = model.train(
                data=os.path.join(project_path, 'datasets'),
                project=os.path.join(project_path, "runs", "classify"),
                name=f"train_{job_id}",
                epochs=config.get('epochs', 10),
                batch=config.get('batch_size', 32),
                imgsz=config.get('imgsz', 224),
                exist_ok=True
            )
            
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 100
            self.jobs[job_id]["logs"].append("Classification training completed.")
            
        except Exception as e:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["logs"].append(f"Error: {str(e)}")
            print(f"Classification Training Error: {e}")

    def _run_training(self, job_id: str, project_path: str, model_name: str, yaml_path: str, config: Dict):
        try:
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["logs"].append("Initializing model...")
            
            # Load model
            model = YOLO(model_name)
            
            # Define custom callback to update progress
            def on_train_epoch_end(trainer):
                current = trainer.epoch + 1
                total = trainer.epochs
                self.jobs[job_id]["current_epoch"] = current
                self.jobs[job_id]["progress"] = (current / total) * 100
                
                # Metrics
                if trainer.metrics:
                    self.jobs[job_id]["metrics"] = {
                        "map50": float(trainer.metrics.get("metrics/mAP50(B)", 0)),
                        "map5095": float(trainer.metrics.get("metrics/mAP50-95(B)", 0)),
                        "loss": float(trainer.loss_items[0]) if trainer.loss_items else 0
                    }
                
                self.jobs[job_id]["logs"].append(f"Epoch {current}/{total} completed.")

            # Attach callback
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            self.jobs[job_id]["logs"].append("Starting training...")
            
            # Start Training
            # project argument sets the save directory
            results = model.train(
                data=yaml_path,
                project=os.path.join(project_path, "runs"),
                name=f"train_{job_id}",
                epochs=config.get('epochs', 10),
                batch=config.get('batch_size', 16),
                imgsz=config.get('imgsz', 640),
                exist_ok=True
            )
            
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 100
            self.jobs[job_id]["logs"].append("Training completed successfully.")

        except Exception as e:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["logs"].append(f"Error: {str(e)}")
            print(f"Training Error: {e}")

training_manager = TrainingManager()
