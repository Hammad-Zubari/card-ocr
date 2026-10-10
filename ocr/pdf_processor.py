"""PDF validation and page rendering (PyMuPDF).

PyMuPDF is not thread-safe, so pages are rendered sequentially by one producer
thread; the heavier per-page image work happens in a worker pool afterwards.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pymupdf

from config import settings
from ocr.types import PageImage
from utils.logging_setup import ctx

log = logging.getLogger(__name__)


class PDFError(Exception):
    """The PDF cannot be processed (corrupted, encrypted, not a PDF, ...)."""


@dataclass
class PDFInfo:
    path: Path
    name: str
    page_count: int


def inspect_pdf(path: Path, display_name: str | None = None) -> PDFInfo:
    """Validate a PDF and return its page count. Raises PDFError."""
    name = display_name or path.name
    with open(path, "rb") as fh:
        if not fh.read(1024).lstrip().startswith(b"%PDF"):
            raise PDFError("not a PDF file")
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        log.warning("%s cannot be opened: %s", ctx(name), exc)  # details (with server path) go to the log only
        raise PDFError("corrupted or unreadable PDF") from exc
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise PDFError("PDF is password-protected")
        if doc.page_count == 0:
            raise PDFError("PDF has no pages")
        if doc.page_count > settings.max_pages_per_pdf:
            raise PDFError(
                f"PDF has {doc.page_count} pages (limit MAX_PAGES_PER_PDF={settings.max_pages_per_pdf})"
            )
        return PDFInfo(path=path, name=name, page_count=doc.page_count)
    finally:
        doc.close()


def render_pages(info: PDFInfo, dpi: int | None = None) -> Iterator[PageImage | tuple[int, Exception]]:
    """Yield each page as a BGR image. A page that fails yields (page_number, error)."""
    dpi = dpi or settings.render_dpi
    doc = pymupdf.open(info.path)
    if doc.needs_pass:
        doc.authenticate("")
    try:
        for i in range(doc.page_count):
            page_no = i + 1
            try:
                page = doc[i]
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                bgr = rgb[:, :, ::-1].copy()  # own the memory; pixmap buffer is freed below
                del pix
                yield PageImage(
                    pdf_name=info.name,
                    page_number=page_no,
                    image=bgr,
                    width_pt=page.rect.width,
                    height_pt=page.rect.height,
                    dpi=dpi,
                )
            except Exception as exc:  # one bad page must not stop the PDF
                log.exception("%s page render failed", ctx(info.name, page_no))
                yield page_no, exc
    finally:
        doc.close()
