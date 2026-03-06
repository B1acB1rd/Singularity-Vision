"""
Remote Executor for Singularity Vision

Manages remote job lifecycle:
- Job submission to cloud endpoints
- Progress polling
- Result download and merge
- Fallback to local on failure
"""

import os
import json
import asyncio
import logging
import hashlib
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import time

logger = logging.getLogger("singularity.remote_executor")


class RemoteJobStatus(Enum):
    """Status of a remote job."""
    PENDING = "pending"
    UPLOADING = "uploading"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    FALLBACK_LOCAL = "fallback_local"


@dataclass
class RemoteJobConfig:
    """Configuration for a remote job."""
    job_type: str  # "training", "inference", "3d_reconstruction"
    project_path: str
    dataset_pointers: list  # List of file paths or dataset IDs
    model_config: Dict[str, Any] = field(default_factory=dict)
    execution_config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 5=high
    callback_url: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "job_type": self.job_type,
            "project_path": self.project_path,
            "dataset_pointers": self.dataset_pointers,
            "model_config": self.model_config,
            "execution_config": self.execution_config,
            "priority": self.priority,
            "callback_url": self.callback_url
        }
    
    def compute_hash(self) -> str:
        """Compute unique hash for job caching."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class RemoteJob:
    """Represents a remote execution job."""
    job_id: str
    config: RemoteJobConfig
    status: RemoteJobStatus = RemoteJobStatus.PENDING
    progress: float = 0.0  # 0-100
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_path: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "job_type": self.config.job_type,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_path": self.result_path,
            "error": self.error,
            "metrics": self.metrics
        }


class RemoteExecutor:
    """
    Manages remote job lifecycle.
    
    Features:
    - Secure job submission
    - Progress polling
    - Result download
    - Fallback to local execution
    """
    
    # Default remote endpoints (configurable)
    DEFAULT_ENDPOINTS = {
        "training": "https://api.singularity.vision/v1/training",
        "inference": "https://api.singularity.vision/v1/inference",
        "3d_reconstruction": "https://api.singularity.vision/v1/reconstruction"
    }
    
    def __init__(self, endpoints: Optional[Dict[str, str]] = None):
        self.endpoints = endpoints or self.DEFAULT_ENDPOINTS
        self.jobs: Dict[str, RemoteJob] = {}
        self.polling_interval = 2  # seconds
        self._polling_threads: Dict[str, threading.Thread] = {}
        self._stop_polling = threading.Event()
        
        # Callbacks for job state changes
        self.on_progress: Optional[Callable[[str, float, str], None]] = None
        self.on_complete: Optional[Callable[[str, Dict], None]] = None
        self.on_error: Optional[Callable[[str, str], None]] = None
    
    def submit_job(
        self,
        config: RemoteJobConfig,
        fallback_handler: Optional[Callable] = None
    ) -> RemoteJob:
        """
        Submit a job for remote execution.
        
        Args:
            config: Job configuration
            fallback_handler: Function to call if remote fails
            
        Returns:
            RemoteJob instance
        """
        import uuid
        
        job_id = f"job_{config.compute_hash()}_{uuid.uuid4().hex[:8]}"
        job = RemoteJob(job_id=job_id, config=config)
        
        self.jobs[job_id] = job
        
        # Start async submission
        thread = threading.Thread(
            target=self._execute_job,
            args=(job, fallback_handler),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Submitted remote job: {job_id}")
        return job
    
    def _execute_job(self, job: RemoteJob, fallback_handler: Optional[Callable]):
        """Execute job (runs in background thread)."""
        try:
            # Phase 1: Upload data if needed
            job.status = RemoteJobStatus.UPLOADING
            job.message = "Preparing data for upload..."
            self._notify_progress(job)
            
            upload_result = self._upload_data(job)
            if not upload_result.get("success"):
                raise Exception(f"Upload failed: {upload_result.get('error')}")
            
            # Phase 2: Submit to remote
            job.status = RemoteJobStatus.QUEUED
            job.message = "Job queued on remote server"
            self._notify_progress(job)
            
            remote_job_id = self._submit_to_remote(job, upload_result)
            if not remote_job_id:
                raise Exception("Failed to submit job to remote server")
            
            # Phase 3: Poll for progress
            job.status = RemoteJobStatus.RUNNING
            job.started_at = datetime.now()
            
            self._poll_until_complete(job, remote_job_id)
            
            # Phase 4: Download results
            if job.status == RemoteJobStatus.COMPLETED:
                job.message = "Downloading results..."
                self._download_results(job, remote_job_id)
                job.completed_at = datetime.now()
                logger.info(f"Job {job.job_id} completed successfully")
                
                if self.on_complete:
                    self.on_complete(job.job_id, job.to_dict())
            
        except Exception as e:
            logger.error(f"Remote execution failed for {job.job_id}: {e}")
            job.error = str(e)
            
            if fallback_handler:
                job.status = RemoteJobStatus.FALLBACK_LOCAL
                job.message = "Falling back to local execution..."
                self._notify_progress(job)
                
                try:
                    fallback_handler(job.config)
                    job.status = RemoteJobStatus.COMPLETED
                    job.message = "Completed via local fallback"
                except Exception as fallback_error:
                    job.status = RemoteJobStatus.FAILED
                    job.error = f"Remote: {e}, Local fallback: {fallback_error}"
            else:
                job.status = RemoteJobStatus.FAILED
            
            if self.on_error:
                self.on_error(job.job_id, job.error or str(e))
    
    def _upload_data(self, job: RemoteJob) -> Dict:
        """
        Upload dataset to remote storage.
        
        For now, returns mock success. In production:
        - Compress dataset
        - Upload to S3/GCS/Azure
        - Return presigned URLs
        """
        # Mock upload - in production, implement actual upload
        time.sleep(0.5)  # Simulate upload time
        
        return {
            "success": True,
            "upload_id": f"upload_{job.job_id}",
            "dataset_urls": [f"s3://bucket/{p}" for p in job.config.dataset_pointers[:5]]
        }
    
    def _submit_to_remote(self, job: RemoteJob, upload_result: Dict) -> Optional[str]:
        """
        Submit job to remote execution endpoint.
        
        For now, returns mock job ID. In production:
        - POST to remote API
        - Include auth token
        - Return remote job ID
        """
        # Mock submission - in production, implement HTTP call
        time.sleep(0.3)
        
        return f"remote_{job.job_id}"
    
    def _poll_until_complete(self, job: RemoteJob, remote_job_id: str):
        """Poll remote server until job completes."""
        max_polls = 1000
        poll_count = 0
        
        while poll_count < max_polls and not self._stop_polling.is_set():
            poll_count += 1
            
            # Mock progress - in production, GET from remote API
            progress = min(100, poll_count * 5)
            job.progress = progress
            job.message = f"Processing... {progress:.0f}%"
            self._notify_progress(job)
            
            if progress >= 100:
                job.status = RemoteJobStatus.COMPLETED
                job.progress = 100
                job.metrics = {
                    "accuracy": 0.92,
                    "loss": 0.08,
                    "processing_time_seconds": poll_count * self.polling_interval
                }
                break
            
            time.sleep(self.polling_interval)
    
    def _download_results(self, job: RemoteJob, remote_job_id: str):
        """
        Download results from remote.
        
        In production:
        - Download model weights
        - Download metrics
        - Download logs
        """
        output_dir = os.path.join(job.config.project_path, "outputs", "remote_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Mock result file
        result_path = os.path.join(output_dir, f"{job.job_id}_result.json")
        with open(result_path, 'w') as f:
            json.dump({
                "job_id": job.job_id,
                "remote_job_id": remote_job_id,
                "metrics": job.metrics,
                "completed_at": datetime.now().isoformat()
            }, f, indent=2)
        
        job.result_path = result_path
    
    def _notify_progress(self, job: RemoteJob):
        """Notify progress callback."""
        if self.on_progress:
            self.on_progress(job.job_id, job.progress, job.message)
    
    def get_job(self, job_id: str) -> Optional[RemoteJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> list:
        """Get all jobs."""
        return [j.to_dict() for j in self.jobs.values()]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [RemoteJobStatus.PENDING, RemoteJobStatus.QUEUED, RemoteJobStatus.RUNNING]:
            job.status = RemoteJobStatus.CANCELLED
            job.message = "Job cancelled by user"
            logger.info(f"Cancelled job: {job_id}")
            return True
        
        return False
    
    def cleanup_completed(self, older_than_hours: int = 24):
        """Remove completed jobs older than specified hours."""
        cutoff = datetime.now()
        to_remove = []
        
        for job_id, job in self.jobs.items():
            if job.status in [RemoteJobStatus.COMPLETED, RemoteJobStatus.FAILED, RemoteJobStatus.CANCELLED]:
                if job.completed_at:
                    age_hours = (cutoff - job.completed_at).total_seconds() / 3600
                    if age_hours > older_than_hours:
                        to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old jobs")


class HybridExecutor:
    """
    Orchestrates hybrid execution: preprocessing local + heavy task remote.
    """
    
    def __init__(self, remote_executor: RemoteExecutor):
        self.remote = remote_executor
    
    def execute_hybrid_training(
        self,
        project_path: str,
        dataset_path: str,
        model_config: Dict,
        local_preprocessor: Optional[Callable] = None
    ) -> RemoteJob:
        """
        Execute training with local preprocessing + remote training.
        
        1. Run preprocessing locally (augmentation, validation)
        2. Upload preprocessed data
        3. Execute training remotely
        4. Download and merge results
        """
        logger.info("Starting hybrid training execution")
        
        # Step 1: Local preprocessing
        preprocessed_path = dataset_path
        if local_preprocessor:
            logger.info("Running local preprocessing...")
            preprocessed_path = local_preprocessor(dataset_path)
        
        # Step 2: Submit for remote training
        config = RemoteJobConfig(
            job_type="training",
            project_path=project_path,
            dataset_pointers=[preprocessed_path],
            model_config=model_config,
            execution_config={"mode": "hybrid", "preprocessed": True}
        )
        
        return self.remote.submit_job(config)
    
    def execute_hybrid_reconstruction(
        self,
        project_path: str,
        images_dir: str,
        local_feature_extractor: Optional[Callable] = None
    ) -> RemoteJob:
        """
        Execute 3D reconstruction with local feature extraction + remote SfM.
        
        1. Extract features locally (ORB/SIFT)
        2. Upload features
        3. Run heavy SfM remotely
        4. Download point cloud
        """
        logger.info("Starting hybrid 3D reconstruction")
        
        # Step 1: Local feature extraction
        features_path = images_dir
        if local_feature_extractor:
            logger.info("Extracting features locally...")
            features_path = local_feature_extractor(images_dir)
        
        # Step 2: Submit for remote reconstruction
        config = RemoteJobConfig(
            job_type="3d_reconstruction",
            project_path=project_path,
            dataset_pointers=[features_path],
            execution_config={
                "mode": "hybrid",
                "local_features": True,
                "output_format": "ply"
            }
        )
        
        return self.remote.submit_job(config)


# Singleton instances
remote_executor = RemoteExecutor()
hybrid_executor = HybridExecutor(remote_executor)
