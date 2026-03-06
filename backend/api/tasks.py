"""
Tasks API - Background task management endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from core.task_manager import task_manager

router = APIRouter()


@router.get("/active")
async def get_active_tasks():
    """Get all running and pending tasks"""
    return {"tasks": task_manager.get_active_tasks()}


@router.get("/history")
async def get_task_history(limit: int = 20):
    """Get recent task history"""
    return {"tasks": task_manager.get_task_history(limit)}


@router.get("/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a pending task"""
    cancelled = task_manager.cancel_task(task_id)
    
    if not cancelled:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be cancelled (not pending or not found)"
        )
    
    return {"status": "cancelled", "task_id": task_id}


@router.get("/stats")
async def get_task_stats():
    """Get task statistics"""
    all_tasks = task_manager.get_task_history(100)
    
    stats = {
        "total": len(all_tasks),
        "completed": sum(1 for t in all_tasks if t and t.get("status") == "completed"),
        "failed": sum(1 for t in all_tasks if t and t.get("status") == "failed"),
        "pending": sum(1 for t in all_tasks if t and t.get("status") == "pending"),
        "running": sum(1 for t in all_tasks if t and t.get("status") == "running")
    }
    
    return stats
