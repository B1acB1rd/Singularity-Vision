"""
Resources API - System resource monitoring endpoints
"""
from fastapi import APIRouter
from core.resource_monitor import resource_monitor

router = APIRouter()


@router.get("/current")
async def get_current_usage():
    """Get current system resource usage"""
    return resource_monitor.get_current_usage()


@router.get("/history")
async def get_usage_history():
    """Get resource usage history for graphing"""
    return resource_monitor.get_history()


@router.get("/warnings")
async def get_resource_warnings():
    """Get any active resource warnings"""
    usage = resource_monitor.get_current_usage()
    return {"warnings": usage.get("warnings", [])}


@router.get("/summary")
async def get_resource_summary():
    """Get a quick summary of system resources"""
    usage = resource_monitor.get_current_usage()
    
    return {
        "cpu_percent": usage["cpu"]["percent"],
        "memory_percent": usage["memory"]["percent"],
        "disk_percent": usage["disk"]["percent"],
        "gpu_available": usage["gpu"]["available"],
        "gpu_count": usage["gpu"]["count"],
        "has_warnings": len(usage.get("warnings", [])) > 0
    }
