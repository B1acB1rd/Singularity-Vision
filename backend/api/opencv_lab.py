"""
OpenCV Lab API - Visual OpenCV operations playground
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image
import logging

router = APIRouter()
logger = logging.getLogger("singularity.api.opencv_lab")


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to numpy array"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def encode_image(img: np.ndarray, format: str = "png") -> bytes:
    """Encode numpy array to image bytes"""
    if len(img.shape) == 2:
        # Grayscale
        pass
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format=format.upper())
    return buffer.getvalue()


# ============================================================
# FILTERING OPERATIONS
# ============================================================

@router.post("/filter/gaussian-blur")
async def gaussian_blur(
    image: UploadFile = File(...),
    kernel_size: int = Form(5),
    sigma_x: float = Form(0)
):
    """Apply Gaussian blur"""
    try:
        img = decode_image(await image.read())
        ksize = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        result = cv2.GaussianBlur(img, (ksize, ksize), sigma_x)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/filter/median-blur")
async def median_blur(
    image: UploadFile = File(...),
    kernel_size: int = Form(5)
):
    """Apply median blur - good for salt & pepper noise"""
    try:
        img = decode_image(await image.read())
        ksize = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        result = cv2.medianBlur(img, ksize)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/filter/bilateral")
async def bilateral_filter(
    image: UploadFile = File(...),
    d: int = Form(9),
    sigma_color: float = Form(75),
    sigma_space: float = Form(75)
):
    """Bilateral filter - edge-preserving smoothing"""
    try:
        img = decode_image(await image.read())
        result = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/filter/box-blur")
async def box_blur(
    image: UploadFile = File(...),
    kernel_size: int = Form(5)
):
    """Simple box blur (averaging)"""
    try:
        img = decode_image(await image.read())
        result = cv2.blur(img, (kernel_size, kernel_size))
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# EDGE DETECTION
# ============================================================

@router.post("/edge/canny")
async def canny_edge(
    image: UploadFile = File(...),
    threshold1: float = Form(100),
    threshold2: float = Form(200),
    aperture_size: int = Form(3)
):
    """Canny edge detection"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aperture = max(3, min(7, aperture_size if aperture_size % 2 == 1 else aperture_size + 1))
        result = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edge/sobel")
async def sobel_edge(
    image: UploadFile = File(...),
    dx: int = Form(1),
    dy: int = Form(0),
    ksize: int = Form(3)
):
    """Sobel edge detection"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ksize = max(1, min(31, ksize if ksize % 2 == 1 else ksize + 1))
        result = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        result = cv2.convertScaleAbs(result)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edge/laplacian")
async def laplacian_edge(
    image: UploadFile = File(...),
    ksize: int = Form(3)
):
    """Laplacian edge detection"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ksize = max(1, min(31, ksize if ksize % 2 == 1 else ksize + 1))
        result = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        result = cv2.convertScaleAbs(result)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MORPHOLOGICAL OPERATIONS
# ============================================================

def get_kernel(size: int, shape: str = "rect") -> np.ndarray:
    """Get morphological kernel"""
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }
    return cv2.getStructuringElement(shapes.get(shape, cv2.MORPH_RECT), (size, size))


@router.post("/morph/erode")
async def erode(
    image: UploadFile = File(...),
    kernel_size: int = Form(5),
    iterations: int = Form(1),
    kernel_shape: str = Form("rect")
):
    """Erosion - shrinks bright regions"""
    try:
        img = decode_image(await image.read())
        kernel = get_kernel(kernel_size, kernel_shape)
        result = cv2.erode(img, kernel, iterations=iterations)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/morph/dilate")
async def dilate(
    image: UploadFile = File(...),
    kernel_size: int = Form(5),
    iterations: int = Form(1),
    kernel_shape: str = Form("rect")
):
    """Dilation - expands bright regions"""
    try:
        img = decode_image(await image.read())
        kernel = get_kernel(kernel_size, kernel_shape)
        result = cv2.dilate(img, kernel, iterations=iterations)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/morph/open")
async def morphology_open(
    image: UploadFile = File(...),
    kernel_size: int = Form(5),
    kernel_shape: str = Form("rect")
):
    """Opening - erosion then dilation (removes small bright spots)"""
    try:
        img = decode_image(await image.read())
        kernel = get_kernel(kernel_size, kernel_shape)
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/morph/close")
async def morphology_close(
    image: UploadFile = File(...),
    kernel_size: int = Form(5),
    kernel_shape: str = Form("rect")
):
    """Closing - dilation then erosion (fills small dark holes)"""
    try:
        img = decode_image(await image.read())
        kernel = get_kernel(kernel_size, kernel_shape)
        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/morph/gradient")
async def morphology_gradient(
    image: UploadFile = File(...),
    kernel_size: int = Form(5),
    kernel_shape: str = Form("rect")
):
    """Morphological gradient - difference between dilation and erosion (outlines)"""
    try:
        img = decode_image(await image.read())
        kernel = get_kernel(kernel_size, kernel_shape)
        result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/morph/tophat")
async def morphology_tophat(
    image: UploadFile = File(...),
    kernel_size: int = Form(9),
    kernel_shape: str = Form("rect")
):
    """Top hat - difference between input and opening (bright spots)"""
    try:
        img = decode_image(await image.read())
        kernel = get_kernel(kernel_size, kernel_shape)
        result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/morph/blackhat")
async def morphology_blackhat(
    image: UploadFile = File(...),
    kernel_size: int = Form(9),
    kernel_shape: str = Form("rect")
):
    """Black hat - difference between closing and input (dark spots)"""
    try:
        img = decode_image(await image.read())
        kernel = get_kernel(kernel_size, kernel_shape)
        result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# THRESHOLDING
# ============================================================

@router.post("/threshold/simple")
async def simple_threshold(
    image: UploadFile = File(...),
    threshold: int = Form(127),
    max_value: int = Form(255),
    threshold_type: str = Form("binary")  # binary, binary_inv, trunc, tozero, tozero_inv
):
    """Simple thresholding"""
    types = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV
    }
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(gray, threshold, max_value, types.get(threshold_type, cv2.THRESH_BINARY))
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threshold/adaptive")
async def adaptive_threshold(
    image: UploadFile = File(...),
    max_value: int = Form(255),
    method: str = Form("gaussian"),  # mean, gaussian
    threshold_type: str = Form("binary"),
    block_size: int = Form(11),
    c: float = Form(2)
):
    """Adaptive thresholding"""
    methods = {
        "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
        "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    }
    types = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV
    }
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block = max(3, block_size if block_size % 2 == 1 else block_size + 1)
        result = cv2.adaptiveThreshold(
            gray, max_value,
            methods.get(method, cv2.ADAPTIVE_THRESH_GAUSSIAN_C),
            types.get(threshold_type, cv2.THRESH_BINARY),
            block, c
        )
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threshold/otsu")
async def otsu_threshold(
    image: UploadFile = File(...),
    max_value: int = Form(255)
):
    """Otsu's automatic thresholding"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threshold/inrange")
async def threshold_inrange(
    image: UploadFile = File(...),
    h_min: int = Form(0),
    s_min: int = Form(0),
    v_min: int = Form(0),
    h_max: int = Form(180),
    s_max: int = Form(255),
    v_max: int = Form(255)
):
    """Color range thresholding in HSV space"""
    try:
        img = decode_image(await image.read())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        result = cv2.inRange(hsv, lower, upper)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# COLOR OPERATIONS
# ============================================================

@router.post("/color/grayscale")
async def to_grayscale(image: UploadFile = File(...)):
    """Convert to grayscale"""
    try:
        img = decode_image(await image.read())
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/color/hsv")
async def to_hsv(image: UploadFile = File(...)):
    """Convert to HSV color space"""
    try:
        img = decode_image(await image.read())
        result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/color/equalize")
async def histogram_equalize(image: UploadFile = File(...)):
    """Histogram equalization"""
    try:
        img = decode_image(await image.read())
        if len(img.shape) == 3:
            # Convert to YUV, equalize Y channel, convert back
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            result = cv2.equalizeHist(img)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/color/clahe")
async def clahe(
    image: UploadFile = File(...),
    clip_limit: float = Form(2.0),
    tile_grid_size: int = Form(8)
):
    """CLAHE - Contrast Limited Adaptive Histogram Equalization"""
    try:
        img = decode_image(await image.read())
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        l = clahe_obj.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/color/invert")
async def invert_colors(image: UploadFile = File(...)):
    """Invert colors (negative)"""
    try:
        img = decode_image(await image.read())
        result = cv2.bitwise_not(img)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/color/channels")
async def split_channels(
    image: UploadFile = File(...),
    channel: str = Form("b")  # b, g, r
):
    """Extract single color channel"""
    try:
        img = decode_image(await image.read())
        channels = {"b": 0, "g": 1, "r": 2}
        idx = channels.get(channel.lower(), 0)
        result = img[:, :, idx]
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# GEOMETRIC TRANSFORMS
# ============================================================

@router.post("/transform/resize")
async def resize_image(
    image: UploadFile = File(...),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    scale: Optional[float] = Form(None),
    interpolation: str = Form("linear")  # nearest, linear, cubic, lanczos
):
    """Resize image"""
    interp = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    try:
        img = decode_image(await image.read())
        h, w = img.shape[:2]
        
        if scale:
            new_w, new_h = int(w * scale), int(h * scale)
        elif width and height:
            new_w, new_h = width, height
        elif width:
            new_w = width
            new_h = int(h * width / w)
        elif height:
            new_h = height
            new_w = int(w * height / h)
        else:
            new_w, new_h = w, h
        
        result = cv2.resize(img, (new_w, new_h), interpolation=interp.get(interpolation, cv2.INTER_LINEAR))
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transform/rotate")
async def rotate_image(
    image: UploadFile = File(...),
    angle: float = Form(90),
    scale: float = Form(1.0)
):
    """Rotate image around center"""
    try:
        img = decode_image(await image.read())
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        result = cv2.warpAffine(img, M, (w, h))
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transform/flip")
async def flip_image(
    image: UploadFile = File(...),
    mode: str = Form("horizontal")  # horizontal, vertical, both
):
    """Flip image"""
    flip_codes = {"horizontal": 1, "vertical": 0, "both": -1}
    try:
        img = decode_image(await image.read())
        result = cv2.flip(img, flip_codes.get(mode, 1))
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# FEATURE DETECTION
# ============================================================

@router.post("/features/contours")
async def find_contours(
    image: UploadFile = File(...),
    mode: str = Form("tree"),  # external, list, ccomp, tree
    method: str = Form("simple"),  # none, simple, tc89_l1, tc89_kcos
    draw_thickness: int = Form(2),
    threshold: int = Form(127)
):
    """Find and draw contours"""
    modes = {
        "external": cv2.RETR_EXTERNAL,
        "list": cv2.RETR_LIST,
        "ccomp": cv2.RETR_CCOMP,
        "tree": cv2.RETR_TREE
    }
    methods = {
        "none": cv2.CHAIN_APPROX_NONE,
        "simple": cv2.CHAIN_APPROX_SIMPLE,
        "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
        "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS
    }
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary,
            modes.get(mode, cv2.RETR_TREE),
            methods.get(method, cv2.CHAIN_APPROX_SIMPLE)
        )
        result = img.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), draw_thickness)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/harris")
async def harris_corners(
    image: UploadFile = File(...),
    block_size: int = Form(2),
    ksize: int = Form(3),
    k: float = Form(0.04),
    threshold: float = Form(0.01)
):
    """Harris corner detection"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        dst = cv2.dilate(dst, None)
        result = img.copy()
        result[dst > threshold * dst.max()] = [0, 0, 255]
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/hough-lines")
async def hough_lines(
    image: UploadFile = File(...),
    rho: float = Form(1),
    theta_degrees: float = Form(1),
    threshold: int = Form(100),
    min_line_length: int = Form(50),
    max_line_gap: int = Form(10)
):
    """Probabilistic Hough Line Transform"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        theta = np.pi * theta_degrees / 180
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        result = img.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/hough-circles")
async def hough_circles(
    image: UploadFile = File(...),
    dp: float = Form(1),
    min_dist: int = Form(50),
    param1: float = Form(100),
    param2: float = Form(30),
    min_radius: int = Form(0),
    max_radius: int = Form(0)
):
    """Hough Circle Transform"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp, min_dist,
            param1=param1, param2=param2,
            minRadius=min_radius, maxRadius=max_radius
        )
        result = img.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ADVANCED OPERATIONS
# ============================================================

@router.post("/advanced/sharpen")
async def sharpen_image(
    image: UploadFile = File(...),
    strength: float = Form(1.0)
):
    """Unsharp masking - sharpen image"""
    try:
        img = decode_image(await image.read())
        gaussian = cv2.GaussianBlur(img, (0, 0), 3)
        result = cv2.addWeighted(img, 1 + strength, gaussian, -strength, 0)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/denoise")
async def denoise_image(
    image: UploadFile = File(...),
    h: float = Form(10),
    template_window_size: int = Form(7),
    search_window_size: int = Form(21)
):
    """Non-local means denoising"""
    try:
        img = decode_image(await image.read())
        result = cv2.fastNlMeansDenoisingColored(
            img, None, h, h, template_window_size, search_window_size
        )
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/dft")
async def discrete_fourier_transform(image: UploadFile = File(...)):
    """Compute and visualize DFT magnitude spectrum"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute DFT
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Magnitude spectrum
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude = np.log(magnitude + 1)
        
        # Normalize to 0-255
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        result = np.uint8(magnitude)
        
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/histogram")
async def draw_histogram(
    image: UploadFile = File(...),
    channel: str = Form("all")  # all, blue, green, red, gray
):
    """Draw histogram visualization"""
    try:
        img = decode_image(await image.read())
        
        # Create histogram image
        hist_h = 256
        hist_w = 512
        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        
        if channel == "gray":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
            for i in range(1, 256):
                cv2.line(hist_img,
                        (i * 2, hist_h - int(hist[i - 1])),
                        (i * 2, hist_h - int(hist[i])),
                        (200, 200, 200), 2)
        else:
            colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255)}
            for idx, (name, color) in enumerate(colors.items()):
                if channel != "all" and channel != name:
                    continue
                hist = cv2.calcHist([img], [idx], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
                for i in range(1, 256):
                    cv2.line(hist_img,
                            (i * 2, hist_h - int(hist[i - 1])),
                            (i * 2, hist_h - int(hist[i])),
                            color, 1)
        
        return Response(content=encode_image(hist_img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/watershed")
async def watershed_segmentation(
    image: UploadFile = File(...),
    marker_threshold: int = Form(50)
):
    """Watershed segmentation"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Distance transform for foreground
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, marker_threshold / 100 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(img, markers)
        result = img.copy()
        result[markers == -1] = [0, 0, 255]  # Red boundaries
        
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/distance-transform")
async def distance_transform(
    image: UploadFile = File(...),
    distance_type: str = Form("l2"),  # l1, l2, c
    mask_size: int = Form(5)
):
    """Distance transform"""
    dist_types = {
        "l1": cv2.DIST_L1,
        "l2": cv2.DIST_L2,
        "c": cv2.DIST_C
    }
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        dist = cv2.distanceTransform(
            binary,
            dist_types.get(distance_type, cv2.DIST_L2),
            mask_size
        )
        dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
        result = np.uint8(dist)
        
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/skeleton")
async def skeletonize(image: UploadFile = File(...)):
    """Morphological skeletonization"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        skeleton = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()
            
            if cv2.countNonZero(binary) == 0:
                break
        
        return Response(content=encode_image(skeleton), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/inpaint")
async def inpaint_image(
    image: UploadFile = File(...),
    radius: int = Form(3),
    method: str = Form("telea")  # telea, ns
):
    """Inpainting - remove white regions"""
    methods = {
        "telea": cv2.INPAINT_TELEA,
        "ns": cv2.INPAINT_NS
    }
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use white pixels as mask
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        
        result = cv2.inpaint(
            img, mask, radius,
            methods.get(method, cv2.INPAINT_TELEA)
        )
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/convex-hull")
async def convex_hull(
    image: UploadFile = File(...),
    threshold: int = Form(127)
):
    """Find and draw convex hulls of contours"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = img.copy()
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(result, [hull], 0, (0, 255, 0), 2)
            cv2.drawContours(result, [cnt], 0, (255, 0, 0), 1)
        
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/bounding-rects")
async def bounding_rectangles(
    image: UploadFile = File(...),
    threshold: int = Form(127),
    rotated: bool = Form(False)
):
    """Draw bounding rectangles around contours"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = img.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
                
            if rotated:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced/blend")
async def blend_images(
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
    alpha: float = Form(0.5)
):
    """Blend two images (or add white overlay if no second image)"""
    try:
        img1 = decode_image(await image1.read())
        
        if image2:
            img2_bytes = await image2.read()
            img2 = decode_image(img2_bytes)
            # Resize img2 to match img1
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        else:
            # Create white overlay
            img2 = np.ones_like(img1) * 255
        
        result = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/color/brightness-contrast")
async def adjust_brightness_contrast(
    image: UploadFile = File(...),
    brightness: int = Form(0),  # -100 to 100
    contrast: float = Form(1.0)  # 0.5 to 3.0
):
    """Adjust brightness and contrast"""
    try:
        img = decode_image(await image.read())
        result = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/color/gamma")
async def gamma_correction(
    image: UploadFile = File(...),
    gamma: float = Form(1.0)
):
    """Gamma correction"""
    try:
        img = decode_image(await image.read())
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(img, table)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transform/crop")
async def crop_image(
    image: UploadFile = File(...),
    x: int = Form(0),
    y: int = Form(0),
    width: int = Form(100),
    height: int = Form(100)
):
    """Crop image region"""
    try:
        img = decode_image(await image.read())
        h, w = img.shape[:2]
        
        # Clamp values
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        x2 = max(x + 1, min(x + width, w))
        y2 = max(y + 1, min(y + height, h))
        
        result = img[y:y2, x:x2]
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/filter/sharpen-kernel")
async def sharpen_kernel(
    image: UploadFile = File(...),
    intensity: float = Form(1.0)
):
    """Apply sharpening kernel"""
    try:
        img = decode_image(await image.read())
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        kernel = kernel * intensity
        kernel[1, 1] = 5 * intensity
        
        result = cv2.filter2D(img, -1, kernel)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/filter/emboss")
async def emboss_image(image: UploadFile = File(...)):
    """Apply emboss effect"""
    try:
        img = decode_image(await image.read())
        kernel = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
        result = cv2.filter2D(img, -1, kernel) + 128
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edge/scharr")
async def scharr_edge(
    image: UploadFile = File(...),
    dx: int = Form(1),
    dy: int = Form(0)
):
    """Scharr edge detection (more accurate than Sobel)"""
    try:
        img = decode_image(await image.read())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.Scharr(gray, cv2.CV_64F, dx, dy)
        result = cv2.convertScaleAbs(result)
        return Response(content=encode_image(result), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# GET AVAILABLE OPERATIONS
# ============================================================

@router.get("/operations")
async def list_operations():
    """List all available OpenCV operations"""
    return {
        "categories": [
            {
                "id": "filter",
                "name": "Filtering",
                "icon": "blur",
                "operations": [
                    {"id": "gaussian-blur", "name": "Gaussian Blur", "params": ["kernel_size", "sigma_x"]},
                    {"id": "median-blur", "name": "Median Blur", "params": ["kernel_size"]},
                    {"id": "bilateral", "name": "Bilateral Filter", "params": ["d", "sigma_color", "sigma_space"]},
                    {"id": "box-blur", "name": "Box Blur", "params": ["kernel_size"]},
                    {"id": "sharpen-kernel", "name": "Sharpen", "params": ["intensity"]},
                    {"id": "emboss", "name": "Emboss", "params": []},
                ]
            },
            {
                "id": "edge",
                "name": "Edge Detection",
                "icon": "scan-line",
                "operations": [
                    {"id": "canny", "name": "Canny Edge", "params": ["threshold1", "threshold2", "aperture_size"]},
                    {"id": "sobel", "name": "Sobel", "params": ["dx", "dy", "ksize"]},
                    {"id": "laplacian", "name": "Laplacian", "params": ["ksize"]},
                    {"id": "scharr", "name": "Scharr", "params": ["dx", "dy"]},
                ]
            },
            {
                "id": "morph",
                "name": "Morphology",
                "icon": "shapes",
                "operations": [
                    {"id": "erode", "name": "Erosion", "params": ["kernel_size", "iterations", "kernel_shape"]},
                    {"id": "dilate", "name": "Dilation", "params": ["kernel_size", "iterations", "kernel_shape"]},
                    {"id": "open", "name": "Opening", "params": ["kernel_size", "kernel_shape"]},
                    {"id": "close", "name": "Closing", "params": ["kernel_size", "kernel_shape"]},
                    {"id": "gradient", "name": "Gradient", "params": ["kernel_size", "kernel_shape"]},
                    {"id": "tophat", "name": "Top Hat", "params": ["kernel_size", "kernel_shape"]},
                    {"id": "blackhat", "name": "Black Hat", "params": ["kernel_size", "kernel_shape"]},
                ]
            },
            {
                "id": "threshold",
                "name": "Thresholding",
                "icon": "contrast",
                "operations": [
                    {"id": "simple", "name": "Simple Threshold", "params": ["threshold", "max_value", "threshold_type"]},
                    {"id": "adaptive", "name": "Adaptive Threshold", "params": ["max_value", "method", "block_size", "c"]},
                    {"id": "otsu", "name": "Otsu's Threshold", "params": ["max_value"]},
                    {"id": "inrange", "name": "Color Range (HSV)", "params": ["h_min", "s_min", "v_min", "h_max", "s_max", "v_max"]},
                ]
            },
            {
                "id": "color",
                "name": "Color Operations",
                "icon": "palette",
                "operations": [
                    {"id": "grayscale", "name": "Grayscale", "params": []},
                    {"id": "hsv", "name": "Convert to HSV", "params": []},
                    {"id": "equalize", "name": "Histogram Equalization", "params": []},
                    {"id": "clahe", "name": "CLAHE", "params": ["clip_limit", "tile_grid_size"]},
                    {"id": "invert", "name": "Invert Colors", "params": []},
                    {"id": "channels", "name": "Extract Channel", "params": ["channel"]},
                    {"id": "brightness-contrast", "name": "Brightness/Contrast", "params": ["brightness", "contrast"]},
                    {"id": "gamma", "name": "Gamma Correction", "params": ["gamma"]},
                ]
            },
            {
                "id": "transform",
                "name": "Transforms",
                "icon": "move",
                "operations": [
                    {"id": "resize", "name": "Resize", "params": ["width", "height", "scale", "interpolation"]},
                    {"id": "rotate", "name": "Rotate", "params": ["angle", "scale"]},
                    {"id": "flip", "name": "Flip", "params": ["mode"]},
                    {"id": "crop", "name": "Crop", "params": ["x", "y", "width", "height"]},
                ]
            },
            {
                "id": "features",
                "name": "Feature Detection",
                "icon": "scan",
                "operations": [
                    {"id": "contours", "name": "Find Contours", "params": ["mode", "method", "threshold"]},
                    {"id": "harris", "name": "Harris Corners", "params": ["block_size", "ksize", "k", "threshold"]},
                    {"id": "hough-lines", "name": "Hough Lines", "params": ["rho", "theta_degrees", "threshold", "min_line_length", "max_line_gap"]},
                    {"id": "hough-circles", "name": "Hough Circles", "params": ["dp", "min_dist", "param1", "param2", "min_radius", "max_radius"]},
                ]
            },
            {
                "id": "advanced",
                "name": "Advanced",
                "icon": "sparkles",
                "operations": [
                    {"id": "sharpen", "name": "Unsharp Mask", "params": ["strength"]},
                    {"id": "denoise", "name": "Denoise (NLM)", "params": ["h", "template_window_size", "search_window_size"]},
                    {"id": "dft", "name": "DFT Spectrum", "params": []},
                    {"id": "histogram", "name": "Histogram", "params": ["channel"]},
                    {"id": "watershed", "name": "Watershed", "params": ["marker_threshold"]},
                    {"id": "distance-transform", "name": "Distance Transform", "params": ["distance_type", "mask_size"]},
                    {"id": "skeleton", "name": "Skeletonize", "params": []},
                    {"id": "inpaint", "name": "Inpainting", "params": ["radius", "method"]},
                    {"id": "convex-hull", "name": "Convex Hull", "params": ["threshold"]},
                    {"id": "bounding-rects", "name": "Bounding Rectangles", "params": ["threshold", "rotated"]},
                ]
            }
        ]
    }

