"""
Background Task Manager - Queue and manage async tasks
"""
import os
import threading
import queue
import uuid
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EXPORT = "export"
    TRACKING = "tracking"
    BUNDLE = "bundle"
    OTHER = "other"


@dataclass
class BackgroundTask:
    id: str
    name: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    current_step_num: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackgroundTaskManager:
    """
    Manages background tasks with progress tracking.
    
    Features:
    - Task queue with priority
    - Progress callbacks
    - Task history
    - Cancellation support
    """
    
    def __init__(self, max_concurrent: int = 2):
        self.tasks: Dict[str, BackgroundTask] = {}
        self.task_queue: queue.Queue = queue.Queue()
        self.max_concurrent = max_concurrent
        self.running_count = 0
        self.lock = threading.Lock()
        self.history: List[str] = []  # Task IDs in order
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def create_task(
        self,
        name: str,
        task_type: TaskType,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        metadata: dict = None
    ) -> str:
        """
        Create and queue a new background task.
        
        Args:
            name: Human-readable task name
            task_type: Type of task
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            metadata: Additional metadata
            
        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())
        
        task = BackgroundTask(
            id=task_id,
            name=name,
            task_type=task_type,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        self.history.append(task_id)
        
        # Keep only last 100 in history
        if len(self.history) > 100:
            old_id = self.history.pop(0)
            if old_id in self.tasks and self.tasks[old_id].status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                del self.tasks[old_id]
        
        # Queue the task
        self.task_queue.put((task_id, func, args, kwargs or {}))
        
        return task_id
    
    def _worker(self):
        """Worker thread to process queued tasks"""
        while True:
            try:
                task_id, func, args, kwargs = self.task_queue.get()
                
                task = self.tasks.get(task_id)
                if not task:
                    continue
                
                # Wait if at max concurrent
                with self.lock:
                    while self.running_count >= self.max_concurrent:
                        pass
                    self.running_count += 1
                
                # Run task
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now().isoformat()
                
                try:
                    # Pass update callback to function
                    kwargs['progress_callback'] = lambda p, s: self.update_progress(task_id, p, s)
                    result = func(*args, **kwargs)
                    
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.progress = 100.0
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                finally:
                    task.completed_at = datetime.now().isoformat()
                    with self.lock:
                        self.running_count -= 1
                    self.task_queue.task_done()
                    
            except Exception as e:
                print(f"Worker error: {e}")
    
    def update_progress(self, task_id: str, progress: float, step: str = ""):
        """Update task progress"""
        task = self.tasks.get(task_id)
        if task:
            task.progress = progress
            task.current_step = step
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "id": task.id,
            "name": task.name,
            "type": task.task_type.value,
            "status": task.status.value,
            "progress": task.progress,
            "current_step": task.current_step,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "result": task.result,
            "error": task.error,
            "metadata": task.metadata
        }
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all running and pending tasks"""
        return [
            self.get_task(tid) for tid, task in self.tasks.items()
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        ]
    
    def get_task_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent task history"""
        recent_ids = self.history[-limit:]
        return [self.get_task(tid) for tid in reversed(recent_ids) if self.get_task(tid)]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now().isoformat()
            return True
        return False


# Singleton instance
task_manager = BackgroundTaskManager()
