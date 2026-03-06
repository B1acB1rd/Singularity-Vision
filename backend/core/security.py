"""
Security Utilities - Path validation, encryption, input sanitization
"""
import os
import re
import hashlib
import logging
from typing import Optional, List, Tuple
from pathlib import Path
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("singularity.security")


# Allowed base directories for file operations
ALLOWED_BASE_DIRS: List[str] = []


def set_allowed_dirs(dirs: List[str]):
    """Set the allowed base directories for file operations"""
    global ALLOWED_BASE_DIRS
    ALLOWED_BASE_DIRS = [os.path.abspath(d) for d in dirs]


def add_allowed_dir(directory: str):
    """Add a directory to the allowed list"""
    ALLOWED_BASE_DIRS.append(os.path.abspath(directory))


class PathValidationError(Exception):
    """Raised when path validation fails"""
    pass


class SecurityError(Exception):
    """General security error"""
    pass


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and special characters.
    
    Args:
        filename: Raw filename from user input
        
    Returns:
        Sanitized filename safe for filesystem
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    # Remove path separators
    filename = os.path.basename(filename)
    
    # Remove null bytes
    filename = filename.replace('\0', '')
    
    # Remove path traversal attempts
    filename = filename.replace('..', '')
    
    # Only allow alphanumeric, underscore, hyphen, dot, space
    filename = re.sub(r'[^\w\-. ]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure not empty after sanitization
    if not filename:
        filename = "unnamed"
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext
    
    return filename


def validate_path(
    path: str,
    must_exist: bool = False,
    allow_creation: bool = True,
    allowed_extensions: Optional[List[str]] = None
) -> str:
    """
    Validate a file path for security.
    
    Args:
        path: Path to validate
        must_exist: If True, path must already exist
        allow_creation: If True, allow paths that don't exist yet
        allowed_extensions: List of allowed file extensions (e.g., ['.jpg', '.png'])
        
    Returns:
        Validated absolute path
        
    Raises:
        PathValidationError: If validation fails
    """
    if not path:
        raise PathValidationError("Path cannot be empty")
    
    # Convert to absolute path
    abs_path = os.path.abspath(path)
    
    # Check for null bytes
    if '\0' in abs_path:
        logger.warning(f"Null byte in path attempt: {repr(path)}")
        raise PathValidationError("Invalid path: contains null bytes")
    
    # Normalize the path
    abs_path = os.path.normpath(abs_path)
    
    # Check if path is within allowed directories
    if ALLOWED_BASE_DIRS:
        is_allowed = False
        for allowed_dir in ALLOWED_BASE_DIRS:
            try:
                # Use commonpath to check if path is under allowed dir
                common = os.path.commonpath([abs_path, allowed_dir])
                if common == allowed_dir:
                    is_allowed = True
                    break
            except ValueError:
                # Different drives on Windows
                continue
        
        if not is_allowed:
            logger.warning(f"Path outside allowed directories: {abs_path}")
            raise PathValidationError(
                f"Access denied: path is outside allowed directories"
            )
    
    # Check existence
    if must_exist and not os.path.exists(abs_path):
        raise PathValidationError(f"Path does not exist: {abs_path}")
    
    if not allow_creation and not os.path.exists(abs_path):
        raise PathValidationError(f"Path does not exist and creation not allowed")
    
    # Check extension
    if allowed_extensions:
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in [e.lower() for e in allowed_extensions]:
            raise PathValidationError(
                f"Invalid file extension: {ext}. Allowed: {allowed_extensions}"
            )
    
    return abs_path


def validate_path_within_project(path: str, project_path: str) -> str:
    """
    Validate that a path is within a project directory.
    
    Args:
        path: Path to validate
        project_path: Project root directory
        
    Returns:
        Validated absolute path
    """
    abs_path = os.path.abspath(path)
    abs_project = os.path.abspath(project_path)
    
    try:
        common = os.path.commonpath([abs_path, abs_project])
        if common != abs_project:
            raise PathValidationError(
                f"Path must be within project directory"
            )
    except ValueError:
        raise PathValidationError("Path must be on same drive as project")
    
    return abs_path


def validate_zip_member(member_name: str, target_dir: str) -> Tuple[bool, str]:
    """
    Validate a ZIP archive member name to prevent Zip Slip attacks.
    
    Args:
        member_name: Name of file within ZIP
        target_dir: Target extraction directory
        
    Returns:
        Tuple of (is_safe, absolute_target_path)
    """
    # Normalize the member name
    member_name = member_name.replace('\\', '/')
    
    # Check for absolute paths
    if os.path.isabs(member_name):
        logger.warning(f"Absolute path in ZIP: {member_name}")
        return False, ""
    
    # Check for path traversal
    if '..' in member_name:
        logger.warning(f"Path traversal in ZIP: {member_name}")
        return False, ""
    
    # Compute target path
    target_path = os.path.abspath(os.path.join(target_dir, member_name))
    target_dir_abs = os.path.abspath(target_dir)
    
    # Ensure target is within extraction directory
    try:
        common = os.path.commonpath([target_path, target_dir_abs])
        if common != target_dir_abs:
            logger.warning(f"ZIP member escapes target: {member_name}")
            return False, ""
    except ValueError:
        return False, ""
    
    return True, target_path


# Resource limits
MAX_FILE_SIZE_MB = 500  # Maximum file size for uploads
MAX_ZIP_SIZE_MB = 2000  # Maximum ZIP file size
MAX_ZIP_RATIO = 10  # Maximum compression ratio (prevents zip bombs)
MAX_BATCH_SIZE = 1000  # Maximum batch processing size
MAX_IMAGE_DIMENSION = 10000  # Maximum image dimension


def check_file_size(file_path: str, max_mb: float = MAX_FILE_SIZE_MB) -> bool:
    """Check if file is within size limits"""
    if not os.path.exists(file_path):
        return True  # Allow non-existent paths (for creation)
    
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > max_mb:
        raise SecurityError(f"File too large: {size_mb:.1f}MB (max: {max_mb}MB)")
    return True


def validate_zip_safe(zip_path: str) -> bool:
    """
    Validate a ZIP file for safety.
    
    Checks for:
    - Zip bombs (excessive compression ratio)
    - Zip slip attacks
    - Reasonable file count
    """
    import zipfile
    
    if not os.path.exists(zip_path):
        raise SecurityError("ZIP file not found")
    
    compressed_size = os.path.getsize(zip_path)
    if compressed_size > MAX_ZIP_SIZE_MB * 1024 * 1024:
        raise SecurityError(f"ZIP file too large (max: {MAX_ZIP_SIZE_MB}MB)")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Calculate total uncompressed size
        total_uncompressed = sum(info.file_size for info in zf.infolist())
        
        # Check compression ratio (zip bomb detection)
        if compressed_size > 0:
            ratio = total_uncompressed / compressed_size
            if ratio > MAX_ZIP_RATIO:
                logger.warning(f"Suspicious ZIP ratio: {ratio}")
                raise SecurityError(
                    f"Suspicious compression ratio: {ratio:.1f}x (max: {MAX_ZIP_RATIO}x)"
                )
        
        # Check for dangerous paths
        for info in zf.infolist():
            is_safe, _ = validate_zip_member(info.filename, "/tmp")
            if not is_safe:
                raise SecurityError(f"Dangerous path in ZIP: {info.filename}")
        
        # Check file count
        if len(zf.infolist()) > 10000:
            raise SecurityError("ZIP contains too many files (max: 10000)")
    
    return True


# Input validation
def validate_positive_int(value: int, name: str, max_val: int = None) -> int:
    """Validate a positive integer parameter"""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    if max_val and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}")
    return value


def validate_float_range(
    value: float,
    name: str,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> float:
    """Validate a float within range"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}")
    return float(value)
