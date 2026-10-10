"""Detect and crop individual business cards on a rendered page.

Strategy, from most to least reliable:
  1. "contour": find card-shaped quadrilaterals (card edges / shadows / a card
     on a contrasting background) and warp each one upright.
  2. "content": when the page is a true-scale scan large enough to hold several
     cards but no edges are visible (white cards on a white scanner bed), group
     ink blobs into clusters that each fit inside a physical card.
  3. "page": treat the whole page as one card (one-card-per-page PDFs, photos,
     anything ambiguous). This is the safe fallback and never fails.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from ocr.image_preprocessor import deskew, ink_mask, text_mask, to_gray
from ocr.types import CardCrop, PageImage
from utils.logging_setup import ctx

log = logging.getLogger(__name__)

# ISO/US business card is ~3.5 x 2 in (252 x 144 pt); EU 85 x 55 mm is similar.
CARD_LONG_PT, CARD_SHORT_PT = 252.0, 144.0
WORK_SIDE = 1400  # detection runs on a downscaled copy for speed


def detect_cards(page: PageImage) -> list[CardCrop]:
    img = page.image
    h, w = img.shape[:2]
    scale = min(1.0, WORK_SIDE / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    gray = to_gray(small)
    px_per_pt = page.dpi / 72.0 * scale

    # A page that is physically about one card in size *is* the card.
    page_can_hold_several = page.width_pt * page.height_pt >= 2.2 * CARD_LONG_PT * CARD_SHORT_PT
    if not page_can_hold_several:
        return _finalise(page, [deskew(img)], "page", [(0, 0, w, h)])

    try:
        rects = _contour_cards(gray)
        if rects and _covers_ink(gray, rects):
            # Contact sheets have a regular layout, but strong signatures or
            # logos can make a contour stop inside the last card. Prefer the
            # complete grid when it accounts for at least as many cards.
            grid = _regular_grid_boxes(gray)
            if len(grid) >= 4 and len(grid) >= len(rects):
                full = [_pad_box((int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)),
                                 w, h)
                        for x, y, bw, bh in grid]
                crops = [deskew(img[y:y + bh, x:x + bw]) for x, y, bw, bh in full]
                return _finalise(page, crops, "grid", full)
            crops = [_warp(img, r, scale) for r in rects]
            return _finalise(page, crops, "contour", [_rect_bbox(r, scale) for r in rects])
    except Exception:
        log.exception("%s contour detection failed, trying fallbacks", ctx(page.pdf_name, page.page_number))

    try:
        if _is_true_scale(gray, px_per_pt):
            boxes = _content_clusters(gray, px_per_pt)
            if len(boxes) >= 2:
                full = [(int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)) for x, y, bw, bh in boxes]
                crops = [deskew(img[y:y + bh, x:x + bw]) for x, y, bw, bh in full]
                return _finalise(page, crops, "content", full)
    except Exception:
        log.exception("%s content clustering failed, trying grid fallback", ctx(page.pdf_name, page.page_number))

    grid = _regular_grid_boxes(gray)
    if len(grid) >= 2:
        full = [_pad_box((int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)),
                         w, h) for x, y, bw, bh in grid]
        crops = [deskew(img[y:y + bh, x:x + bw]) for x, y, bw, bh in full]
        return _finalise(page, crops, "grid", full)

    # Rescaled screenshots and portrait scans can hide physical card size, so
    # the true-scale guard above cannot help. If clearly separated horizontal
    # ink bands fill the page, treat each band as one vertically stacked card.
    bands = _horizontal_card_bands(gray)
    if len(bands) >= 2:
        full = [(int(x / scale), int(y / scale), int(bw / scale), int(bh / scale))
                for x, y, bw, bh in bands]
        crops = [deskew(img[y:y + bh, x:x + bw]) for x, y, bw, bh in full]
        return _finalise(page, crops, "content", full)

    return _finalise(page, [deskew(img)], "page", [(0, 0, w, h)])


# --------------------------------------------------------------------------- contour

def _rect_iou(r1: tuple, r2: tuple) -> float:
    p1 = np.intp(cv2.boxPoints(r1))
    p2 = np.intp(cv2.boxPoints(r2))
    x0 = min(p1[:, 0].min(), p2[:, 0].min())
    x1 = max(p1[:, 0].max(), p2[:, 0].max())
    y0 = min(p1[:, 1].min(), p2[:, 1].min())
    y1 = max(p1[:, 1].max(), p2[:, 1].max())
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask1 = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    mask2 = np.zeros_like(mask1)
    cv2.fillPoly(mask1, [p1 - [x0, y0]], 1)
    cv2.fillPoly(mask2, [p2 - [x0, y0]], 1)
    inter = int(np.count_nonzero(mask1 & mask2))
    union = int(np.count_nonzero(mask1 | mask2))
    return inter / union if union > 0 else 0.0


def _contour_cards(gray: np.ndarray) -> list[tuple]:
    h, w = gray.shape
    page_area = float(h * w)
    candidates: list[tuple] = []

    # 1. Bilateral filter preserves faint card edges against scanner bed
    blurred = cv2.bilateralFilter(gray, 7, 50, 50)
    for c_low, c_high in [(15, 50), (25, 80)]:
        edges = cv2.Canny(blurred, c_low, c_high)
        for k_sz in [3, 5]:
            dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (k_sz, k_sz)), iterations=2)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if rw < 10 or rh < 10:
                    continue
                area = rw * rh
                frac = area / page_area
                if not (0.02 <= frac <= 0.88):
                    continue
                aspect = max(rw, rh) / min(rw, rh)
                if not (1.15 <= aspect <= 2.6):
                    continue
                if min(rw, rh) < int(h * 0.08) or max(rw, rh) < int(w * 0.22):
                    continue
                hull = cv2.convexHull(c)
                if cv2.contourArea(hull) / area < 0.45:
                    continue
                candidates.append(rect)

    # 2. Otsu threshold masks for dark cards or cards on darker backgrounds
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for o_mask in [otsu, 255 - otsu]:
        contours, _ = cv2.findContours(o_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if rw < 10 or rh < 10:
                continue
            area = rw * rh
            frac = area / page_area
            if not (0.02 <= frac <= 0.88):
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if not (1.15 <= aspect <= 2.6):
                continue
            if min(rw, rh) < int(h * 0.08) or max(rw, rh) < int(w * 0.22):
                continue
            hull = cv2.convexHull(c)
            if cv2.contourArea(hull) / area < 0.50:
                continue
            candidates.append(rect)

    kept = _suppress_nested(candidates)
    return _reading_order(kept, key=lambda r: (r[0][0], r[0][1]))


def _suppress_nested(rects: list[tuple]) -> list[tuple]:
    rects = sorted(rects, key=lambda r: r[1][0] * r[1][1], reverse=True)
    kept: list[tuple] = []
    for r in rects:
        (cx, cy), (rw, rh), _ = r
        drop = False
        for k in kept:
            (kcx, kcy), (krw, krh), _ = k
            iou = _rect_iou(r, k)
            dist = np.hypot(cx - kcx, cy - kcy)
            min_dim = min(rw, rh, krw, krh)
            if iou > 0.35 or dist < min_dim * 0.55:
                drop = True
                break
        if not drop:
            kept.append(r)
    return kept


def _covers_ink(gray: np.ndarray, rects: list[tuple]) -> bool:
    """Reject detections that leave a lot of text outside every card.
    Uses a stroke mask so solid logo blocks don't count as "text"."""
    mask = text_mask(gray)
    total = int(np.count_nonzero(mask))
    if total == 0:
        return False
    region = np.zeros_like(mask)
    for r in rects:
        (cx, cy), (rw, rh), a = r
        grown = ((cx, cy), (rw * 1.04 + 6, rh * 1.04 + 6), a)
        cv2.fillPoly(region, [np.intp(cv2.boxPoints(grown))], 255)
    inside = int(np.count_nonzero(cv2.bitwise_and(mask, region)))
    return inside / total >= 0.80


def _order_points(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _warp(img: np.ndarray, rect: tuple, scale: float) -> np.ndarray:
    (cx, cy), (rw, rh), a = rect
    # Contours can lock onto an inner border or a strong printed line (especially
    # on cards at the bottom of a contact sheet). Keep a generous safety margin
    # so logos, QR codes, and contact lines at the card edges are not clipped.
    rw = min(img.shape[1] * scale, rw * 1.12 + 12)
    rh = min(img.shape[0] * scale, rh * 1.12 + 12)
    rect = ((cx, cy), (rw, rh), a)
    pts = _order_points(cv2.boxPoints(rect) / scale)
    pts[:, 0] = np.clip(pts[:, 0], 0, img.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, img.shape[0] - 1)
    tl, tr, br, bl = pts
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width <= 0 or height <= 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, m, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _rect_bbox(rect: tuple, scale: float) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(np.intp(cv2.boxPoints(rect) / scale))
    return int(x), int(y), int(w), int(h)


# --------------------------------------------------------------------------- content

def _is_true_scale(gray: np.ndarray, px_per_pt: float) -> bool:
    """True when glyph sizes look like a real-size card scan (not a zoomed photo)."""
    mask = ink_mask(gray)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    heights = heights[(heights >= 3) & (stats[1:, cv2.CC_STAT_AREA] >= 6)]
    if len(heights) < 20:
        return False
    return float(np.median(heights)) / px_per_pt <= 12.0


def _horizontal_card_bands(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find strongly separated stacked card regions on portrait/rescaled pages."""
    h, w = gray.shape
    mask = text_mask(gray)
    row_counts = np.count_nonzero(mask, axis=1)
    active = row_counts >= max(2, int(w * 0.004))
    # Bridge small line gaps inside a card, but preserve a real card-to-card gap.
    active = cv2.morphologyEx(active.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones(11, np.uint8)).astype(bool)
    spans: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= max(20, int(h * 0.12)):
                spans.append((start, i))
            start = None
    if start is not None and h - start >= max(20, int(h * 0.12)):
        spans.append((start, h))
    if len(spans) < 2:
        return []

    # Merge only near-adjacent spans; a genuine card gap is at least 4% of page.
    merged: list[list[int]] = []
    for y0, y1 in spans:
        if merged and y0 - merged[-1][1] < h * 0.04:
            merged[-1][1] = y1
        else:
            merged.append([y0, y1])
    if len(merged) < 2 or any((y1 - y0) < h * 0.16 for y0, y1 in merged):
        return []
    boxes = []
    for y0, y1 in merged:
        ys, xs = np.where(mask[y0:y1] > 0)
        if len(xs) == 0:
            return []
        x0 = max(0, int(xs.min()) - int(w * 0.04))
        x1 = min(w, int(xs.max()) + int(w * 0.04))
        if x1 - x0 < w * 0.45:
            return []
        boxes.append((x0, max(0, y0 - int(h * 0.02)), x1 - x0,
                      min(h, y1 + int(h * 0.02)) - max(0, y0 - int(h * 0.02))))
    return boxes


def _regular_grid_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find occupied repeated card cells on contact-sheet style PDF pages."""
    h, w = gray.shape
    mask = text_mask(gray)
    best: tuple[float, list[tuple[int, int, int, int]]] = (0.0, [])
    for cols in range(1, 3):
        for rows in range(2, 6):
            gap_x, gap_y = max(8, int(w * 0.035)), max(8, int(h * 0.025))
            margin_x, margin_y = max(4, int(w * 0.02)), max(4, int(h * 0.015))
            cell_w = (w - 2 * margin_x - (cols - 1) * gap_x) / cols
            cell_h = (h - 2 * margin_y - (rows - 1) * gap_y) / rows
            aspect = cell_w / cell_h
            if not 1.25 <= aspect <= 2.35:
                continue
            if cell_w < w * 0.3 or cell_h < h * 0.12:
                continue
            boxes: list[tuple[int, int, int, int]] = []
            occupied = 0
            ink_total = 0
            for row in range(rows):
                for col in range(cols):
                    x = int(margin_x + col * (cell_w + gap_x))
                    y = int(margin_y + row * (cell_h + gap_y))
                    bw, bh = int(cell_w), int(cell_h)
                    count = int(np.count_nonzero(mask[y:y + bh, x:x + bw]))
                    if count >= max(300, int(cell_w * cell_h * 0.008)):
                        occupied += 1
                        ink_total += count
                        boxes.append((x, y, bw, bh))
            expected = rows * cols
            if occupied != expected:
                continue
            score = ink_total / expected / max(1, cell_w * cell_h)
            score += 0.0001 / expected
            if score > best[0]:
                best = (score, boxes)
    return best[1]


def _content_clusters(gray: np.ndarray, px_per_pt: float) -> list[tuple[int, int, int, int]]:
    h, w = gray.shape
    # Local thresholding keeps thin / grey body text that a global threshold
    # loses at detection resolution (losing it would crop text off the card).
    mask = text_mask(gray)
    k = max(3, int(0.12 * 72 * px_per_pt))  # ~0.12 inch
    blobs = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    # Judge noise by real ink pixels, not blob area: a short fragment such as
    # ".com" is a small blob but must still be kept and attached to its card.
    ink_per_blob = np.bincount(labels[mask > 0], minlength=n)
    boxes = [list(stats[i, :4]) for i in range(1, n) if ink_per_blob[i] >= 8]
    if len(boxes) < 2 or len(boxes) > 400:
        return []

    long_px, short_px = CARD_LONG_PT * px_per_pt * 1.15, CARD_SHORT_PT * px_per_pt * 1.15

    def fits(b):
        a, c = max(b[2], b[3]), min(b[2], b[3])
        return a <= long_px and c <= short_px

    def union(a, b):
        x0, y0 = min(a[0], b[0]), min(a[1], b[1])
        x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
        return [x0, y0, x1 - x0, y1 - y0]

    def gap(a, b):
        dx = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
        dy = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
        return max(dx, dy)

    groups = [b for b in boxes]
    while True:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                u = union(groups[i], groups[j])
                if not fits(u):
                    continue
                g = gap(groups[i], groups[j])
                if best is None or g < best[0]:
                    best = (g, i, j, u)
        if best is None:
            break
        _, i, j, u = best
        groups[i] = u
        del groups[j]

    card_area = long_px * short_px / 1.15 ** 2
    big = [g for g in groups if g[2] * g[3] >= 0.15 * card_area]
    small = [g for g in groups if g[2] * g[3] < 0.15 * card_area]
    if len(big) < 2:
        return []
    for s in small:  # attach stray fragments to the nearest card instead of losing text
        idx = min(range(len(big)), key=lambda i: gap(big[i], s))
        big[idx] = union(big[idx], s)

    pad = int(0.1 * 72 * px_per_pt)
    out = []
    for x, y, bw, bh in big:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        out.append((x0, y0, x1 - x0, y1 - y0))
    return _reading_order(out, key=lambda b: (b[0] + b[2] / 2, b[1] + b[3] / 2))


# --------------------------------------------------------------------------- helpers

def _pad_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Add a small safety margin around grid cells so border logos/text survive."""
    x, y, bw, bh = box
    px, py = max(8, int(bw * 0.04)), max(8, int(bh * 0.06))
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(width, x + bw + px), min(height, y + bh + py)
    return x0, y0, x1 - x0, y1 - y0

def _reading_order(items: list, key) -> list:
    """Sort top-to-bottom in rows, then left-to-right."""
    if len(items) < 2:
        return list(items)
    pts = [key(it) for it in items]
    order = sorted(range(len(items)), key=lambda i: pts[i][1])
    ys = [pts[i][1] for i in order]
    spread = (max(ys) - min(ys)) or 1
    rows: list[list[int]] = [[order[0]]]
    for prev, cur in zip(order, order[1:]):
        if pts[cur][1] - pts[prev][1] > spread * 0.12:
            rows.append([])
        rows[-1].append(cur)
    return [items[i] for row in rows for i in sorted(row, key=lambda i: pts[i][0])]


def _finalise(page: PageImage, crops: list[np.ndarray], how: str, boxes: list) -> list[CardCrop]:
    out = []
    for i, (crop, box) in enumerate(zip(crops, boxes), start=1):
        if crop.size == 0 or min(crop.shape[:2]) < 20:
            continue
        out.append(CardCrop(pdf_name=page.pdf_name, page_number=page.page_number, index_on_page=i,
                            image=crop, detection=how, bbox=tuple(int(v) for v in box)))
    log.debug("%s %d card(s) via %s", ctx(page.pdf_name, page.page_number), len(out), how)
    return out
