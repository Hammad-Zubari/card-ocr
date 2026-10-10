"""Plain data objects passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PageImage:
    pdf_name: str
    page_number: int          # 1-based
    image: np.ndarray         # BGR uint8
    width_pt: float           # physical page size in PDF points (1/72 inch)
    height_pt: float
    dpi: int


@dataclass
class CardCrop:
    pdf_name: str
    page_number: int
    index_on_page: int        # 1-based, reading order
    image: np.ndarray         # BGR uint8, deskewed
    detection: str            # "contour", "content", "page"
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h on the page image


@dataclass
class OCRLine:
    text: str
    score: float
    box: tuple[float, float, float, float]   # x0, y0, x1, y1

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2


@dataclass
class OCRResult:
    lines: list[OCRLine]
    image_size: tuple[int, int]               # width, height
    error: str | None = None

    @property
    def raw_text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def mean_score(self) -> float:
        return float(np.mean([l.score for l in self.lines])) if self.lines else 0.0


@dataclass
class ExtractedFields:
    name: str = ""
    company_name: str = ""
    address: str = ""
    telephone: str = ""
    mobile_phone: str = ""
    email: str = ""
    # Secondary info kept for review/debugging, not part of the required output.
    job_title: str = ""
    company: str = ""
    website: str = ""
    fax: str = ""
    notes: list[str] = field(default_factory=list)
