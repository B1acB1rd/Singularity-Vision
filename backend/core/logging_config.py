"""
Logging Configuration - Centralized logging setup for Singularity Vision
"""
import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: str = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    Set up centralized logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        console: Enable console output
        file: Enable file output
        
    Returns:
        Root logger
    """
    if log_dir is None:
        log_dir = os.path.join(
            os.path.expanduser("~"),
            ".singularity-vision",
            "logs"
        )
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create root logger for singularity
    root_logger = logging.getLogger("singularity")
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if file:
        log_file = os.path.join(log_dir, "singularity.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Security log (separate file for security events)
    security_logger = logging.getLogger("singularity.security")
    security_log_file = os.path.join(log_dir, "security.log")
    security_handler = logging.handlers.RotatingFileHandler(
        security_log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=10,
        encoding='utf-8'
    )
    security_handler.setFormatter(formatter)
    security_logger.addHandler(security_handler)
    
    root_logger.info(f"Logging initialized. Log directory: {log_dir}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the singularity prefix"""
    return logging.getLogger(f"singularity.{name}")


# Initialize logging on import
setup_logging()
