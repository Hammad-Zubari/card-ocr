"""Business Card PDF OCR — web application entry point.

    python app.py            # then open http://127.0.0.1:5000
"""
from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import settings
from utils.logging_setup import setup_logging

setup_logging()

from database import db  # noqa: E402
from services import processing_service as ps  # noqa: E402
from services import qwen_extractor  # noqa: E402
from services import nvidia_extractor  # noqa: E402
from services import gemini_verifier  # noqa: E402

log = logging.getLogger("app")

app = Flask(__name__, template_folder="gui/templates", static_folder="gui/static")
app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
app.config["SECRET_KEY"] = settings.secret_key

ADMIN_USER = "admin"
ADMIN_PASS = "admin123"

_db_state: dict = {"ok": False, "error": "not initialised"}


def _init_database() -> None:
    try:
        db.init_db()
        _db_state.update(ok=True, error=None)
    except db.DatabaseError as exc:
        _db_state.update(ok=False, error=str(exc))


# ---------------------------------------------------------------- auth & access

@app.before_request
def require_login():
    path = request.path
    if path.startswith("/static/") or path in ("/login", "/favicon.ico"):
        return None
    if not session.get("logged_in"):
        if path.startswith("/api/"):
            return jsonify({"error": "Unauthorized. Please log in first."}), 401
        return redirect(url_for("login_page"))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        if session.get("logged_in"):
            return redirect(url_for("index"))
        return render_template("login.html")

    data = request.get_json(silent=True) or request.form
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))

    if username == ADMIN_USER and password == ADMIN_PASS:
        session["logged_in"] = True
        session["username"] = ADMIN_USER
        log.info("User %s logged in successfully", username)
        return jsonify({"ok": True, "redirect": "/"})

    log.warning("Invalid login attempt for username: %r", username)
    return jsonify({"ok": False, "error": "Invalid username or password"}), 401


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.post("/api/auth/lock")
def api_auth_lock():
    """Immediately invalidate session on back navigation or screen leave."""
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/auth/check")
def api_auth_check():
    """Verify session validity from front-end polling/visibility changes."""
    if not session.get("logged_in"):
        return jsonify({"authenticated": False}), 401
    return jsonify({"authenticated": True, "user": session.get("username")})


@app.after_request
def add_no_cache_headers(response):
    """Ensure pages are never cached in browser history (prevents back-button bypass)."""
    if not request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------- pages

@app.get("/")
def index():
    return render_template(
        "index.html",
        max_upload_mb=settings.max_upload_mb,
        current_user=session.get("username", "admin"),
    )



# ---------------------------------------------------------------- API

@app.get("/api/status")
def status():
    health = db.health()
    if health["ok"] and not _db_state["ok"]:
        _init_database()  # database came back online: create the table now
    return jsonify({"qwen": qwen_extractor.info(), "nvidia": nvidia_extractor.info(),
                    "gemini": gemini_verifier.info(),
                    "database": health | {"schema_ready": _db_state["ok"]}})


@app.get("/api/qwen/check")
def qwen_check():
    return jsonify(qwen_extractor.check_connectivity())


@app.get("/api/nvidia/check")
def nvidia_check():
    return jsonify(nvidia_extractor.check_connectivity())


@app.get("/api/gemini/check")
def gemini_check():
    return jsonify(gemini_verifier.check_connectivity())


@app.post("/api/upload")
def upload():
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return jsonify({"error": "No files received"}), 400
    staging = settings.upload_dir / "_incoming" / uuid.uuid4().hex[:12]
    staging.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[str, Path]] = []
    for i, f in enumerate(files, start=1):
        original = Path(f.filename).name  # strip any client-side path
        safe = secure_filename(original) or f"upload_{i}.pdf"
        dest = staging / f"{i:03d}_{safe}"
        f.save(dest)
        saved.append((original, dest))
    provider = request.form.get("provider", "qwen")
    job = ps.create_job(saved, provider)
    # Move the PDFs next to the job's crops so everything for a batch lives together.
    for entry in job.files:
        src = Path(entry.path)
        dst = job.workdir / src.name
        src.replace(dst)
        entry.path = str(dst)
    staging.rmdir()
    log.info("Job %s created with %d file(s)", job.id, len(job.files))
    return jsonify(job.snapshot())


@app.post("/api/jobs/<job_id>/start")
def start(job_id: str):
    job = _job_or_404(job_id)
    if not ps.start_job(job):
        return jsonify({"error": f"Job is {job.status}"}), 409
    return jsonify(job.snapshot())


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = _job_or_404(job_id)
    since = max(0, request.args.get("since", default=0, type=int))
    return jsonify(job.snapshot(since))


@app.get("/api/jobs/<job_id>/cards/<int:card_id>/image")
def card_image(job_id: str, card_id: int):
    path = ps.image_path(_job_or_404(job_id), card_id)
    if path is None:
        abort(404)
    return send_file(path, mimetype="image/jpeg", max_age=3600)


@app.post("/api/jobs/<job_id>/duplicates")
def recheck(job_id: str):
    job = _job_or_404(job_id)
    rows = (request.get_json(silent=True) or {}).get("rows", [])
    return jsonify({"results": ps.recheck_duplicates(job, rows)})


@app.post("/api/jobs/<job_id>/save")
def save_job(job_id: str):
    job = _job_or_404(job_id)
    if job.status != "done":
        return jsonify({"error": "Processing has not finished yet"}), 409
    rows = (request.get_json(silent=True) or {}).get("rows", [])
    if not rows:
        return jsonify({"error": "Nothing selected"}), 400
    if not _db_state["ok"]:
        _init_database()
        if not _db_state["ok"]:
            return jsonify({"error": _db_state["error"]}), 503
    try:
        return jsonify(ps.save_rows(job, rows))
    except db.DatabaseError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/save")
def api_save():
    if not _db_state["ok"]:
        _init_database()
        if not _db_state["ok"]:
            return jsonify({"error": _db_state["error"]}), 503

    data = request.get_json(silent=True) or {}
    cards = data.get("cards", [])
    if not cards:
        return jsonify({"error": "No cards provided to save"}), 400

    to_save = []
    card_index_map = []
    for c in cards:
        c_id = c.get("id")
        row_data = {
            "name": (c.get("name") or "").strip() or None,
            "company_name": (c.get("company_name") or "").strip() or None,
            "job_title": (c.get("job_title") or "").strip() or None,
            "address": (c.get("address") or "").strip() or None,
            "telephone": (c.get("telephone") or "").strip() or None,
            "mobile_phone": (c.get("mobile_phone") or "").strip() or None,
            "email": (c.get("email") or "").strip() or None,
            "business_type": (c.get("business_type") or "").strip() or None,
            "business_confidence": c.get("business_confidence") or 0,
            "db_id": c.get("db_id"),
        }
        to_save.append(row_data)
        card_index_map.append(c_id)

    try:
        db_ids = db.save_cards(to_save)
        results = [{"id": cid, "db_id": dbid, "error": None} for cid, dbid in zip(card_index_map, db_ids)]
        return jsonify({
            "ok": True,
            "saved": len(db_ids),
            "failed": 0,
            "results": results
        })
    except db.DatabaseError as exc:
        log.exception("Save failed")
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        log.exception("Save failed unexpectedly")
        return jsonify({"error": str(exc)}), 500


@app.post("/api/check-duplicates")
def api_check_duplicates():
    data = request.get_json(silent=True) or {}
    cards = data.get("cards", [])
    if not cards:
        return jsonify({"results": [], "duplicates_found": 0})

    from services import duplicate_service
    results, db_error = duplicate_service.find_duplicates(cards)
    out = []
    dup_count = 0
    for c, res in zip(cards, results):
        if res.status:
            dup_count += 1
        out.append({
            "id": c.get("id"),
        })
    return jsonify({"results": out, "duplicates_found": dup_count, "error": db_error})


@app.get("/api/records")
def records():
    limit = min(1000, max(1, request.args.get("limit", default=200, type=int)))
    offset = max(0, request.args.get("offset", default=0, type=int))
    search = request.args.get("q", "").strip()
    status_filter = request.args.get("status", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    sort_by = request.args.get("sort", "id_desc").strip()
    try:
        rows, total = db.list_records(
            limit=limit,
            offset=offset,
            search=search,
            status_filter=status_filter,
            date_from=date_from,
            date_to=date_to,
            sort_by=sort_by,
        )
    except db.DatabaseError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"rows": rows, "total": total})


@app.get("/api/records/<int:card_id>")
def get_record(card_id: int):
    try:
        record = db.get_record(card_id)
        if not record:
            return jsonify({"error": "Record not found"}), 404
        return jsonify(record)
    except db.DatabaseError as exc:
        return jsonify({"error": str(exc)}), 503


@app.put("/api/records/<int:card_id>")
def update_record(card_id: int):
    data = request.get_json(silent=True) or {}
    try:
        updated = db.update_record(card_id, data)
        return jsonify({"ok": True, "record": updated})
    except db.DatabaseError as exc:
        return jsonify({"error": str(exc)}), 400


@app.delete("/api/records/<int:card_id>")
def delete_record(card_id: int):
    try:
        deleted = db.delete_record(card_id)
        if not deleted:
            return jsonify({"error": "Record not found"}), 404
        return jsonify({"ok": True, "id": card_id})
    except db.DatabaseError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/records/bulk-delete")
def bulk_delete_records():
    data = request.get_json(silent=True) or {}
    ids = [int(x) for x in data.get("ids", []) if str(x).isdigit()]
    if not ids:
        return jsonify({"error": "No valid IDs provided"}), 400
    try:
        count = db.bulk_delete_records(ids)
        return jsonify({"ok": True, "count": count})
    except db.DatabaseError as exc:
        return jsonify({"error": str(exc)}), 503


def _job_or_404(job_id: str) -> ps.Job:
    job = ps.jobs.get(job_id)
    if job is None:
        abort(404, description="Job not found (the server may have restarted)")
    return job


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return jsonify({"error": f"Upload exceeds MAX_UPLOAD_MB={settings.max_upload_mb}"}), 413


@app.errorhandler(Exception)
def on_error(exc):
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.description}), exc.code
    log.exception("Unhandled error on %s", request.path)
    return jsonify({"error": "Internal server error (see logs/app.log)"}), 500


def main() -> None:
    _init_database()
    url = f"http://{settings.host}:{settings.port}"
    log.info("Business Card Qwen Vision running at %s (database: %s)", url, db.safe_url())
    try:
        from waitress import serve
        serve(app, host=settings.host, port=settings.port, threads=8, max_request_body_size=app.config["MAX_CONTENT_LENGTH"])
    except ImportError:
        app.run(host=settings.host, port=settings.port, threaded=True)


if __name__ == "__main__":
    main()
