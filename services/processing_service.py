"""Batch processing orchestration.

Concurrency model (chosen to be fast without exhausting RAM/VRAM):

    producer thread ── renders pages sequentially (PyMuPDF is not thread-safe)
         │             and hands each page to a small thread pool that does
         │             blank-check + card detection + preprocessing (OpenCV
         │             releases the GIL, so this runs truly in parallel)
         ▼
    bounded queue  ── back-pressure: at most a few pages of images in memory
         ▼
    consumer (job thread) ── sends each isolated crop to Qwen Vision.

Rendering/detection of upcoming pages overlaps with OCR of earlier pages.
"""
from __future__ import annotations

import logging
import queue
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

from config import settings
from database import db
from ocr.card_detector import detect_cards
from ocr.image_preprocessor import is_blank, prepare_for_ocr
from ocr.pdf_processor import PDFError, PDFInfo, inspect_pdf, render_pages
from ocr.types import CardCrop, PageImage
from services import duplicate_service, validation_service
from services import qwen_extractor
from services import nvidia_extractor
from services import gemini_verifier
from services import business_classifier
from utils.logging_setup import ctx

log = logging.getLogger(__name__)

STATUS_OK, STATUS_REVIEW, STATUS_FAILED, STATUS_SCANNING = "OK", "Needs Review", "Failed", "Scanning"
_SENTINEL = object()


@dataclass
class CardRecord:
    id: int
    pdf_name: str
    page_number: int
    card_number: int
    detection: str = ""
    name: str = ""
    company_name: str = ""
    address: str = ""
    telephone: str = ""
    mobile_phone: str = ""
    email: str = ""
    job_title: str = ""
    business_type: str = ""
    business_confidence: int = 0
    raw_ocr_text: str = ""
    ocr_status: str = STATUS_OK
    notes: list[str] = field(default_factory=list)
    duplicate_status: str = ""
    duplicate_reason: str = ""
    has_image: bool = False
    db_id: int | None = None
    extraction_ms: int = 0


@dataclass
class FileEntry:
    name: str
    path: str
    pages: int = 0
    error: str | None = None
    cards: int = 0


class Job:
    def __init__(self, job_id: str, files: list[FileEntry], workdir: Path, extraction_provider: str):
        self.id = job_id
        self.files = files
        self.workdir = workdir
        self.extraction_provider = extraction_provider
        self.status = "pending"           # pending | running | done | failed
        self.message = "Ready to process"
        self.created = time.time()
        self.cards: list[CardRecord] = []
        self.errors: list[str] = []
        self.timings: dict[str, float] = {}
        self.pages_total = sum(f.pages for f in files if not f.error)
        self.pages_prepared = 0
        self.empty_pages = 0
        self.cards_detected = 0
        self.lock = threading.RLock()

    # -------------------------------------------------------------- views
    def stats(self) -> dict:
        with self.lock:
            ok = sum(1 for c in self.cards if c.ocr_status == STATUS_OK)
            review = sum(1 for c in self.cards if c.ocr_status == STATUS_REVIEW)
            failed = sum(1 for c in self.cards if c.ocr_status == STATUS_FAILED)
            return {
                "pdfs": len(self.files),
                "pdfs_failed": sum(1 for f in self.files if f.error),
                "pages": self.pages_total,
                "empty_pages": self.empty_pages,
                "cards_detected": self.cards_detected,
                "cards_processed": len(self.cards),
                "successful": ok,
                "needs_review": review,
                "failed": failed,
                "duplicates": sum(1 for c in self.cards if c.duplicate_status == duplicate_service.DUPLICATE),
                "possible_duplicates": sum(1 for c in self.cards if c.duplicate_status == duplicate_service.POSSIBLE),
                "saved": sum(1 for c in self.cards if c.db_id),
            }

    def progress(self) -> float:
        with self.lock:
            if self.status == "done":
                return 1.0
            if not self.pages_total:
                return 0.0
            prepared = self.pages_prepared / self.pages_total
            done = len(self.cards) / self.cards_detected if self.cards_detected else 0.0
            return round(min(0.99, 0.2 * prepared + 0.8 * prepared * done), 4)

    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "message": self.message,
                "extraction_provider": self.extraction_provider,
                "progress": self.progress(),
                "files": [asdict(f) | {"path": None} for f in self.files],
                "stats": self.stats(),
                "timings": {k: round(v, 2) for k, v in self.timings.items()},
                "errors": list(self.errors[-50:]),
                "total_cards": len(self.cards),
                "cards": [asdict(c) for c in self.cards[since:]],
                "since": since,
            }


class JobStore:
    """In-memory registry of recent jobs (oldest finished ones are pruned)."""

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > settings.max_jobs_in_memory:
                old_id, old = next(iter(self._jobs.items()))
                if old.status == "running":
                    break
                self._jobs.pop(old_id)
                shutil.rmtree(old.workdir / "crops", ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


jobs = JobStore()


# ------------------------------------------------------------------ job creation

def create_job(uploads: list[tuple[str, Path]], extraction_provider: str = "qwen") -> Job:
    """uploads: (original file name, saved path). Each PDF is validated up-front."""
    job_id = uuid.uuid4().hex[:12]
    workdir = settings.upload_dir / job_id
    (workdir / "crops").mkdir(parents=True, exist_ok=True)
    files: list[FileEntry] = []
    for original, path in uploads:
        try:
            info = inspect_pdf(path, original)
            files.append(FileEntry(name=original, path=str(path), pages=info.page_count))
        except PDFError as exc:
            log.warning("%s rejected: %s", ctx(original), exc)
            files.append(FileEntry(name=original, path=str(path), error=str(exc)))
    provider = extraction_provider.strip().lower()
    if provider not in {"qwen", "nvidia"}:
        provider = "qwen"
    job = Job(job_id, files, workdir, provider)
    if all(f.error for f in files):
        job.status = "failed"
        job.message = "No valid PDF files"
    jobs.add(job)
    return job


def start_job(job: Job) -> bool:
    with job.lock:
        if job.status != "pending":
            return False
        job.status = "running"
        job.message = "Starting…"
    threading.Thread(target=_run_job, args=(job,), name=f"job-{job.id}", daemon=True).start()
    return True


# ------------------------------------------------------------------ pipeline

def _prepare_page(page: PageImage) -> tuple[PageImage, list[CardCrop] | None, float]:
    """Worker-pool task: blank check, card detection, OCR preprocessing."""
    t0 = time.perf_counter()
    if is_blank(page.image):
        return page, None, time.perf_counter() - t0
    crops = detect_cards(page)
    for c in crops:
        c.image = prepare_for_ocr(c.image)
    page.image = None  # free the full page as early as possible
    return page, crops, time.perf_counter() - t0


def _produce(job: Job, out: "queue.Queue", stage: dict) -> None:
    workers = max(1, settings.preprocess_workers)
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prep") as pool:
            for f in job.files:
                if f.error:
                    continue
                info = PDFInfo(path=Path(f.path), name=f.name, page_count=f.pages)
                pending: deque = deque()
                t_render = time.perf_counter()
                for item in render_pages(info):
                    stage["render"] += time.perf_counter() - t_render
                    if isinstance(item, tuple):  # (page_no, exception)
                        out.put(("page_error", f.name, item[0], item[1]))
                    else:
                        pending.append(pool.submit(_prepare_page, item))
                        while len(pending) > workers * 2:
                            out.put(("page", *pending.popleft().result()))
                    t_render = time.perf_counter()
                while pending:
                    out.put(("page", *pending.popleft().result()))
    except Exception as exc:
        log.exception("Producer failed")
        out.put(("fatal", str(exc)))
    finally:
        out.put(_SENTINEL)


def _run_job(job: Job) -> None:
    t_start = time.perf_counter()
    stage = {"render": 0.0, "detect_preprocess": 0.0, "extraction": 0.0, "duplicates": 0.0}
    job.message = f"Processing cards with {job.extraction_provider.upper()} OCR + Gemini verification…"
    q: queue.Queue = queue.Queue(maxsize=max(4, settings.preprocess_workers * 3))
    producer = threading.Thread(target=_produce, args=(job, q, stage), name=f"render-{job.id}", daemon=True)
    producer.start()

    per_pdf_counter: dict[str, int] = {}
    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        kind = item[0]
        if kind == "fatal":
            job.errors.append(f"Processing error: {item[1]}")
            continue
        if kind == "page_error":
            _, pdf_name, page_no, exc = item
            _add_failed(job, pdf_name, page_no, per_pdf_counter, f"page could not be rendered: {exc}")
            with job.lock:
                job.pages_prepared += 1
            continue
        _, page, crops, dt = item
        stage["detect_preprocess"] += dt
        with job.lock:
            job.pages_prepared += 1
            if crops is None:
                job.empty_pages += 1
                log.info("%s empty page skipped", ctx(page.pdf_name, page.page_number))
                continue
            job.cards_detected += len(crops)
        log.info("%s %d card(s) detected (%s)", ctx(page.pdf_name, page.page_number), len(crops),
                 crops[0].detection if crops else "-")
        for crop in crops:
            _finish_card(job, crop, per_pdf_counter, stage,
                         0)
    producer.join(timeout=5)

    # Duplicate labelling inside the batch and against the database.
    t = time.perf_counter()
    _label_duplicates(job)
    stage["duplicates"] = time.perf_counter() - t

    for f in job.files:
        f.cards = sum(1 for c in job.cards if c.pdf_name == f.name)
    with job.lock:
        job.timings = stage | {"total": time.perf_counter() - t_start}
        job.status = "done"
        s = job.stats()
        job.message = (f"Done: {s['cards_processed']} card(s) from {s['pdfs'] - s['pdfs_failed']} PDF(s) "
                       f"in {job.timings['total']:.1f}s")
    log.info("Job %s finished: %s | timings %s", job.id, job.stats(),
             {k: round(v, 2) for k, v in job.timings.items()})


def _next_card_number(counter: dict[str, int], pdf_name: str) -> int:
    counter[pdf_name] = counter.get(pdf_name, 0) + 1
    return counter[pdf_name]


def _add_failed(job: Job, pdf_name: str, page_no: int, counter: dict, error: str) -> None:
    with job.lock:
        rec = CardRecord(id=len(job.cards) + 1, pdf_name=pdf_name, page_number=page_no,
                         card_number=_next_card_number(counter, pdf_name), ocr_status=STATUS_FAILED, notes=[error])
        job.cards.append(rec)
        job.errors.append(f"{pdf_name} page {page_no}: {error}")


def _finish_card(job: Job, crop: CardCrop, counter: dict, stage: dict, extraction_ms: int) -> None:
    card_no = _next_card_number(counter, crop.pdf_name)
    where = ctx(crop.pdf_name, crop.page_number, card_no)
    with job.lock:  # only the consumer thread appends, so this id stays unique
        card_id = len(job.cards) + 1
    rec = CardRecord(id=card_id, pdf_name=crop.pdf_name, page_number=crop.page_number, card_number=card_no,
                     detection=crop.detection, ocr_status=STATUS_SCANNING)
    try:
        img = crop.image
        f = 900 / max(img.shape[:2])
        if f < 1:
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            (job.workdir / "crops" / f"{rec.id}.jpg").write_bytes(buf.tobytes())
            rec.has_image = True
    except Exception:
        log.warning("%s could not store card preview", where, exc_info=True)
    with job.lock:
        job.cards.append(rec)
    try:
        started = time.perf_counter()
        prov = job.extraction_provider
        result = None
        extract_err = None

        try:
            result = (
                nvidia_extractor.extract_image(crop.image)
                if prov == "nvidia"
                else qwen_extractor.extract_image(crop.image)
            )
        except Exception as primary_error:
            extract_err = str(primary_error)
            fallback = qwen_extractor.extract_image if prov == "nvidia" else nvidia_extractor.extract_image
            log.warning("%s %s OCR failed (%s), trying %s fallback...", where, prov.upper(), primary_error,
                        "Qwen" if prov == "nvidia" else "NVIDIA")
            result = fallback(crop.image)

        elapsed = time.perf_counter() - started
        stage["extraction"] += elapsed
        rec.extraction_ms = int(elapsed * 1000)
        for key in ("name", "job_title", "address", "telephone", "mobile_phone", "email"):
            setattr(rec, key, getattr(result.fields, key) or "")
        rec.company_name = getattr(result.fields, "company_name", "") or getattr(result, "company", "") or ""
        rec.raw_ocr_text = result.raw_text

        # Secondary Gemini verification: strictly check card isolation, email, and phones
        verification_issues = []
        if prov != "gemini" and settings.gemini_api_key:
            try:
                verified, verification_issues = gemini_verifier.verify(crop.image, {
                    key: getattr(rec, key) for key in
                    ("name", "job_title", "address", "telephone", "mobile_phone", "email")
                })
                for key in ("name", "job_title", "address", "telephone", "mobile_phone", "email"):
                    if key in verified:
                        setattr(rec, key, verified[key] or "")
            except Exception as exc:
                log.warning("%s Gemini verification failed; retaining primary result: %s", where, exc)

        rec.notes = [f"missing {label}" for label, value in (
            ("name", rec.name), ("designation", rec.job_title),
            ("telephone", rec.telephone), ("mobile phone", rec.mobile_phone),
            ("email", rec.email)) if not value]
        rec.notes.extend(verification_issues)
        if extract_err:
            rec.notes.append(f"Fallback used ({extract_err})")
        rec.ocr_status = STATUS_REVIEW if rec.notes else STATUS_OK

        # Business classification via web search (best-effort, never blocks)
        try:
            # Only use an explicitly extracted printed website. Never treat an
            # email domain or arbitrary OCR text as the card's website.
            website_hint = (
                getattr(result, "website", "")
                or getattr(result.fields, "website", "")
                or ""
            ).strip()
            classification = business_classifier.classify_with_confidence(
                company_name=rec.company_name or rec.name,
                website=website_hint,
                address=rec.address,
                business_hint=rec.job_title,
                raw_text=rec.raw_ocr_text,
                person_name=rec.name,
            )
            rec.business_type = classification.business_type
            rec.business_confidence = classification.confidence
            if rec.business_type:
                log.info("%s classified as: %s (%d%%)", where, rec.business_type[:60], rec.business_confidence)
        except Exception as cls_exc:
            log.debug("%s business_classifier failed: %s", where, cls_exc)

        log.info("%s extracted (%s) name=%r email=%r tel=%r", where, rec.ocr_status,
                 rec.name, rec.email, rec.telephone)
    except Exception as exc:  # extraction bug on one card must not stop the job
        log.exception("%s field extraction failed", where)
        rec.ocr_status, rec.notes = STATUS_REVIEW, [
            f"{job.extraction_provider.upper()} extraction failed: {exc}"
        ]
        with job.lock:
            job.errors.append(
                f"{where}: {job.extraction_provider.upper()} extraction failed; card sent to Human Review"
            )

    crop.image = None


def _label_duplicates(job: Job) -> None:
    records = [asdict(c) for c in job.cards if c.ocr_status != STATUS_FAILED]
    if not records:
        return
    results, db_error = duplicate_service.find_duplicates(records)
    by_id = {c.id: c for c in job.cards}
    for rec, res in zip(records, results):
        card = by_id[rec["id"]]
        card.duplicate_status, card.duplicate_reason = res.status, res.reason
    if db_error:
        job.errors.append(f"Duplicate check against database skipped: {db_error}")


def save_rows(job: Job, rows: list[dict]) -> dict:
    """Validate the user-reviewed rows and write them to the database.
    Allows saving records even if some fields are null/empty.
    """
    by_id = {c.id: c for c in job.cards}
    errors: list[dict] = []
    skipped: list[dict] = []
    candidates: list[tuple[CardRecord, dict, dict]] = []

    for row in rows:
        card = by_id.get(int(row.get("id", 0)))
        if card is None or not row.get("include"):
            continue
        clean, issues = validation_service.validate_record(
            row.get("name", ""), row.get("address", ""), row.get("telephone", ""),
            row.get("email", ""), row.get("mobile_phone", ""))
        hard = [i for i in issues if i.startswith("invalid email")]
        if hard:
            errors.append({"id": card.id, "error": "; ".join(hard)})
            continue
        # Also clean job_title and business_type
        clean["company_name"] = (row.get("company_name") or card.company_name or "").strip()
        clean["job_title"] = (row.get("job_title") or "").strip()
        clean["business_type"] = (row.get("business_type") or card.business_type or "").strip()
        clean["business_confidence"] = int(row.get("business_confidence") or card.business_confidence or 0)

        # Check if literally every single field is blank
        if not any(clean.values()) and not card.raw_ocr_text:
            errors.append({"id": card.id, "error": "row is empty"})
            continue
        for k, v in clean.items():
            setattr(card, k, v)
        candidates.append((card, clean, row))

    t = time.perf_counter()
    dup_input = [asdict(c) for c, _, _ in candidates]
    dups, db_error = duplicate_service.find_duplicates(dup_input)
    to_save: list[tuple[CardRecord, dict]] = []
    for (card, clean, row), d in zip(candidates, dups):
        card.duplicate_status, card.duplicate_reason = d.status, d.reason
        if d.status == duplicate_service.DUPLICATE and not card.db_id and row.get("dup_ack") != d.status:
            skipped.append({"id": card.id, "reason": f"newly detected duplicate ({d.reason}); review and save again"})
            continue
        to_save.append((card, {
            **clean, "db_id": card.db_id,
            "duplicate_status": card.duplicate_status or None,
        }))

    saved_ids: list[int] = []
    if to_save:
        ids = db.save_cards([payload for _, payload in to_save])
        for (card, _), db_id in zip(to_save, ids):
            card.db_id = db_id
        saved_ids = ids
    dt = time.perf_counter() - t
    with job.lock:
        job.timings["database_save"] = job.timings.get("database_save", 0.0) + dt
    log.info("Job %s: saved %d row(s) in %.2fs (%d skipped, %d invalid)", job.id, len(saved_ids), dt,
             len(skipped), len(errors))
    return {
        "saved": len(saved_ids), "ids": saved_ids, "skipped": skipped, "errors": errors,
        "db_seconds": round(dt, 3), "duplicate_check_error": db_error,
        "cards": [asdict(c) for c in job.cards],
    }


def recheck_duplicates(job: Job, rows: list[dict]) -> list[dict]:
    """Re-run duplicate detection on edited values without saving."""
    by_id = {c.id: c for c in job.cards}
    recs = []
    for row in rows:
        card = by_id.get(int(row.get("id", 0)))
        if card is None:
            continue
        recs.append({**asdict(card), **{k: row.get(k, "") for k in ("name", "address", "telephone", "email")}})
    results, _ = duplicate_service.find_duplicates(recs)
    return [{"id": r["id"], "duplicate_status": d.status, "duplicate_reason": d.reason} for r, d in zip(recs, results)]


def image_path(job: Job, card_id: int) -> Path | None:
    p = job.workdir / "crops" / f"{card_id}.jpg"
    return p if p.exists() else None
