"""
preprocessing.py — Ben Graham Preprocessing Pipeline
=====================================================
Implements the full fundus image preprocessing chain:
  1. Auto-crop uninformative black borders
  2. Resize to target resolution
  3. Gaussian blur subtraction (local lighting normalization)
  4. CLAHE on the green channel
  5. Automated quality filtering for EyePACS
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def auto_crop_fundus(
    image: np.ndarray,
    threshold: int = 10,
    margin_pct: float = 0.02,
) -> np.ndarray:
    """
    Detect the circular fundus region and crop away black borders.

    Strategy: Convert to grayscale → threshold → find largest contour
    (the fundus) → crop to its bounding box with a small margin.

    Args:
        image: BGR image (H, W, 3).
        threshold: Grayscale intensity below which pixels are "black border."
        margin_pct: Fractional margin to add around the detected fundus.

    Returns:
        Cropped BGR image containing primarily the fundus.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Morphological close to fill small gaps inside the fundus
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image  # Fallback: return uncropped if no contour found

    # Pick the largest contour (should be the fundus circle)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add margin
    margin_x = int(w * margin_pct)
    margin_y = int(h * margin_pct)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image.shape[1], x + w + margin_x)
    y2 = min(image.shape[0], y + h + margin_y)

    return image[y1:y2, x1:x2]


def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize to (target_size, target_size) using area interpolation for
    downsampling (anti-aliased) and cubic for upsampling.

    Args:
        image: BGR image (H, W, 3).
        target_size: Target side length in pixels.

    Returns:
        Resized BGR image (target_size, target_size, 3).
    """
    h, w = image.shape[:2]
    if h > target_size or w > target_size:
        interp = cv2.INTER_AREA  # Best for downsampling
    else:
        interp = cv2.INTER_CUBIC  # Best for upsampling
    return cv2.resize(image, (target_size, target_size), interpolation=interp)


def gaussian_blur_subtraction(
    image: np.ndarray,
    sigma_factor: float = 1 / 30,
) -> np.ndarray:
    """
    Ben Graham's local lighting normalization.

    Subtracts a Gaussian-blurred version of the image from itself,
    then re-centers pixel values. This removes low-frequency illumination
    variations while preserving high-frequency pathological features
    (microaneurysms, hemorrhages).

    Formula: output = image - GaussianBlur(image, σ) + 128

    Args:
        image: BGR image (H, W, 3), uint8.
        sigma_factor: σ = image_width × sigma_factor.

    Returns:
        Lighting-normalized BGR image, uint8.
    """
    h, w = image.shape[:2]
    sigma = int(w * sigma_factor)
    # Ensure kernel size is odd
    ksize = sigma * 2 + 1 if sigma > 0 else 3

    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # Subtract with re-centering at 128 to avoid negative values
    # Cast to int16 to prevent overflow, then clip
    result = image.astype(np.int16) - blurred.astype(np.int16) + 128
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe_green_channel(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    specifically to the green channel.

    The green channel in fundus images carries the most discriminative
    information for DR lesion detection (microaneurysms, hemorrhages
    have highest contrast in green).

    Args:
        image: BGR image (H, W, 3), uint8.
        clip_limit: CLAHE clip limit (higher = more contrast).
        tile_grid_size: Grid size for adaptive equalization.

    Returns:
        BGR image with CLAHE applied to green channel, uint8.
    """
    b, g, r = cv2.split(image)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    g_enhanced = clahe.apply(g)

    return cv2.merge([b, g_enhanced, r])


def assess_image_quality(
    image: np.ndarray,
    min_brightness: float = 15.0,
    max_brightness: float = 245.0,
    min_contrast: float = 20.0,
    min_fundus_ratio: float = 0.15,
) -> Tuple[bool, str]:
    """
    Automated quality filter for EyePACS images.

    Rejects images that are:
    - Too dark (underexposed / nearly black)
    - Too bright (overexposed / washed out)
    - Too low contrast (out of focus / heavily blurred)
    - Too small fundus area (mostly black border, possibly mislabeled)

    Args:
        image: BGR image (H, W, 3), uint8.
        min_brightness: Minimum mean intensity threshold.
        max_brightness: Maximum mean intensity threshold.
        min_contrast: Minimum standard deviation threshold.
        min_fundus_ratio: Minimum ratio of non-black pixels to total pixels.

    Returns:
        (passes_quality, reason): Boolean and human-readable reason if rejected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness check
    mean_intensity = np.mean(gray)
    if mean_intensity < min_brightness:
        return False, f"Too dark (mean={mean_intensity:.1f} < {min_brightness})"
    if mean_intensity > max_brightness:
        return False, f"Too bright (mean={mean_intensity:.1f} > {max_brightness})"

    # Contrast check (standard deviation of pixel intensities)
    std_intensity = np.std(gray)
    if std_intensity < min_contrast:
        return False, f"Low contrast (std={std_intensity:.1f} < {min_contrast})"

    # Fundus area ratio check
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    fundus_ratio = np.sum(binary > 0) / binary.size
    if fundus_ratio < min_fundus_ratio:
        return False, f"Small fundus (ratio={fundus_ratio:.3f} < {min_fundus_ratio})"

    return True, "OK"


def preprocess_fundus(
    image: np.ndarray,
    target_size: int = 512,
    sigma_factor: float = 1 / 30,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Full Ben Graham preprocessing pipeline.

    Pipeline:
        1. Auto-crop black borders
        2. Resize to target_size × target_size
        3. Gaussian blur subtraction (lighting normalization)
        4. CLAHE on green channel (contrast enhancement)

    Args:
        image: Raw BGR fundus image, uint8.
        target_size: Output resolution (e.g., 512 for CNN/Swin, 224 for RETFound).
        sigma_factor: Gaussian blur σ factor.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid.

    Returns:
        Preprocessed BGR image (target_size, target_size, 3), uint8.
    """
    # Step 1: Crop
    image = auto_crop_fundus(image)

    # Step 2: Resize
    image = resize_image(image, target_size)

    # Step 3: Gaussian blur subtraction
    image = gaussian_blur_subtraction(image, sigma_factor)

    # Step 4: CLAHE on green channel
    image = apply_clahe_green_channel(image, clip_limit, tile_grid_size)

    return image


def preprocess_and_filter(
    image_path: str,
    target_size: int = 512,
    quality_check: bool = True,
    sigma_factor: float = 1 / 30,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    min_brightness: float = 15.0,
    max_brightness: float = 245.0,
    min_contrast: float = 20.0,
    min_fundus_ratio: float = 0.15,
) -> Tuple[Optional[np.ndarray], bool, str]:
    """
    Load an image from disk, optionally quality-filter, and preprocess.

    Args:
        image_path: Absolute path to the fundus image.
        target_size: Output resolution.
        quality_check: Whether to apply the quality filter.
        (remaining args): See individual function docstrings.

    Returns:
        (preprocessed_image_or_None, passed_quality, reason_string)
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, False, f"Failed to load: {image_path}"

    # Quality filter (before expensive preprocessing)
    if quality_check:
        passed, reason = assess_image_quality(
            image, min_brightness, max_brightness, min_contrast, min_fundus_ratio
        )
        if not passed:
            return None, False, reason

    # Full preprocessing
    processed = preprocess_fundus(
        image, target_size, sigma_factor, clip_limit, tile_grid_size
    )

    return processed, True, "OK"
