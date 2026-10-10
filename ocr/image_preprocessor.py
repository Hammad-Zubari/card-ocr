"""Light-weight image preprocessing.

PP-OCR models are trained on natural colour images, so aggressive
binarisation hurts accuracy. We only: deskew small angles, normalise the
resolution, and boost contrast when a scan is washed out.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import settings


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (255 = ink) of dark content on a lighter background."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # If Otsu labelled most of the image as "ink" the page is dark; nothing useful.
    if mask.mean() > 255 * 0.6:
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def text_mask(gray: np.ndarray) -> np.ndarray:
    """Mask of thin strokes (text) only: local thresholding ignores the inside of
    large uniform areas such as coloured logo blocks or header bands."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 18)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def is_blank(img: np.ndarray) -> bool:
    gray = to_gray(img)
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) if max(gray.shape) > 1200 else gray
    if float(small.std()) < 4.0:
        return True
    # Count only clearly dark pixels, so scanner noise doesn't count as content.
    dark = (small < min(170, int(np.median(small)) - 60)).mean()
    return dark < 0.0008


def estimate_skew(img: np.ndarray) -> float:
    """Estimate small text skew in degrees (positive = counter-clockwise)."""
    gray = to_gray(img)
    scale = 1000 / max(gray.shape) if max(gray.shape) > 1000 else 1.0
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = ink_mask(gray)
    k = max(9, gray.shape[1] // 40)
    lines = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles, weights = [], []
    for c in contours:
        (_, _), (w, h), a = cv2.minAreaRect(c)
        if w < h:
            w, h = h, w
            a += 90
        if h < 4 or w < 4 * h or w < gray.shape[1] * 0.08:
            continue
        a = ((a + 45) % 90) - 45  # normalise to [-45, 45)
        angles.append(a)
        weights.append(w)
    if len(angles) < 2:
        return 0.0
    order = np.argsort(angles)
    a_sorted, w_sorted = np.array(angles)[order], np.array(weights)[order]
    cum = np.cumsum(w_sorted)
    median = float(a_sorted[np.searchsorted(cum, cum[-1] / 2)])
    # cv2 angle convention -> image rotation convention
    return -median


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def deskew(img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    angle = estimate_skew(img)
    if 0.4 <= abs(angle) <= max_angle:
        return rotate(img, -angle)
    return img


def prepare_for_ocr(img: np.ndarray) -> np.ndarray:
    """Normalise resolution and contrast of a card crop before OCR."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side < settings.min_card_long_side:
        f = settings.min_card_long_side / long_side
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    elif long_side > settings.max_card_long_side:
        f = settings.max_card_long_side / long_side
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    gray = to_gray(img)
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 - p2 < 110:  # washed-out / low-contrast scan
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.ascontiguousarray(img)
