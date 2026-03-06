"""
Resource Monitor - Real-time system resource usage
"""
import os
import psutil
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque


class ResourceMonitor:
    """
    Monitors system resources in real-time.
    
    Tracks:
    - CPU usage
    - Memory usage
    - GPU usage (if available)
    - Disk usage
    """
    
    def __init__(self, history_length: int = 60):
        self.history_length = history_length
        self.history: Dict[str, deque] = {
            "cpu": deque(maxlen=history_length),
            "memory": deque(maxlen=history_length),
            "gpu": deque(maxlen=history_length),
            "disk": deque(maxlen=history_length)
        }
        self.monitoring = False
        self.monitor_thread = None
        self.warning_thresholds = {
            "cpu": 90,
            "memory": 85,
            "gpu": 95,
            "disk": 90
        }
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage snapshot"""
        usage = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=0.1),
                "cores": psutil.cpu_count(),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
            },
            "disk": {
                "percent": psutil.disk_usage('/').percent,
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "used_gb": round(psutil.disk_usage('/').used / (1024**3), 2),
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
            },
            "gpu": self._get_gpu_usage()
        }
        
        # Check for warnings
        usage["warnings"] = self._check_warnings(usage)
        
        return usage
    
    def _get_gpu_usage(self) -> Dict[str, Any]:
        """Get GPU usage if available"""
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                gpus = []
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    
                    gpus.append({
                        "id": i,
                        "name": props.name,
                        "memory_total_gb": round(props.total_memory / (1024**3), 2),
                        "memory_allocated_gb": round(memory_allocated / (1024**3), 2),
                        "memory_reserved_gb": round(memory_reserved / (1024**3), 2),
                        "utilization_percent": round((memory_allocated / props.total_memory) * 100, 1) if props.total_memory > 0 else 0
                    })
                
                return {
                    "available": True,
                    "count": device_count,
                    "devices": gpus
                }
            else:
                return {"available": False, "count": 0, "devices": []}
                
        except ImportError:
            return {"available": False, "count": 0, "devices": []}
        except Exception as e:
            return {"available": False, "error": str(e), "devices": []}
    
    def _check_warnings(self, usage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for resource warnings"""
        warnings = []
        
        if usage["cpu"]["percent"] >= self.warning_thresholds["cpu"]:
            warnings.append({
                "type": "cpu",
                "level": "high",
                "message": f"CPU usage at {usage['cpu']['percent']}%"
            })
        
        if usage["memory"]["percent"] >= self.warning_thresholds["memory"]:
            warnings.append({
                "type": "memory",
                "level": "high",
                "message": f"Memory usage at {usage['memory']['percent']}%"
            })
        
        if usage["disk"]["percent"] >= self.warning_thresholds["disk"]:
            warnings.append({
                "type": "disk",
                "level": "high",
                "message": f"Disk usage at {usage['disk']['percent']}%"
            })
        
        # GPU warnings
        if usage["gpu"]["available"]:
            for gpu in usage["gpu"]["devices"]:
                if gpu.get("utilization_percent", 0) >= self.warning_thresholds["gpu"]:
                    warnings.append({
                        "type": "gpu",
                        "level": "high",
                        "message": f"GPU {gpu['id']} memory at {gpu['utilization_percent']}%"
                    })
        
        return warnings
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while self.monitoring:
            usage = self.get_current_usage()
            
            self.history["cpu"].append(usage["cpu"]["percent"])
            self.history["memory"].append(usage["memory"]["percent"])
            self.history["disk"].append(usage["disk"]["percent"])
            
            if usage["gpu"]["available"] and usage["gpu"]["devices"]:
                avg_gpu = sum(g.get("utilization_percent", 0) for g in usage["gpu"]["devices"]) / len(usage["gpu"]["devices"])
                self.history["gpu"].append(avg_gpu)
            
            time.sleep(interval)
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get usage history for graphing"""
        return {
            "cpu": list(self.history["cpu"]),
            "memory": list(self.history["memory"]),
            "gpu": list(self.history["gpu"]),
            "disk": list(self.history["disk"]),
            "length": self.history_length
        }


# Singleton instance
resource_monitor = ResourceMonitor()
resource_monitor.start_monitoring()
