"""End-to-end test against the running web app (python app.py must be running).

    python scripts/smoke_test_http.py [--base http://127.0.0.1:5000]

Uploads the sample PDFs plus broken inputs (corrupted PDF, non-PDF, blank page),
processes them, edits one row, saves to the database, and verifies that the
stored records match what the GUI table holds.
"""
from __future__ import annotations

import argparse
import io
import sys
import tempfile
import time
from pathlib import Path

import pymupdf
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def make_bad_inputs(tmp: Path) -> list[Path]:
    corrupted = tmp / "corrupted.pdf"
    corrupted.write_bytes(b"%PDF-1.4\n%garbage that is not a real pdf structure\n" * 3)
    not_pdf = tmp / "notes_not_a_pdf.pdf"
    not_pdf.write_text("hello, this is a text file", encoding="utf-8")
    with_blank = tmp / "with_blank_page.pdf"
    src = pymupdf.open(ROOT / "samples" / "cards_one_per_page.pdf")
    doc = pymupdf.open()
    doc.new_page(width=252, height=144)               # empty page
    doc.insert_pdf(src, from_page=0, to_page=0)      # one real card
    doc.save(with_blank)
    return [corrupted, not_pdf, with_blank]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    base = ap.parse_args().base
    s = requests.Session()
    s.trust_env = False  # don't route localhost calls through a system HTTP proxy

    print("status:", s.get(f"{base}/api/status").json())
    tmp = Path(tempfile.mkdtemp())
    pdfs = [ROOT / "samples" / n for n in ("cards_one_per_page.pdf", "cards_multi_per_page.pdf", "cards_low_quality.pdf")]
    pdfs += make_bad_inputs(tmp)
    files = [("files", (p.name, p.read_bytes(), "application/pdf")) for p in pdfs]
    job = s.post(f"{base}/api/upload", files=files).json()
    for f in job["files"]:
        print(f"  uploaded {f['name']:<28} pages={f['pages']:<3} error={f['error']}")

    t0 = time.perf_counter()
    s.post(f"{base}/api/jobs/{job['id']}/start").raise_for_status()
    seen = 0
    while True:
        snap = s.get(f"{base}/api/jobs/{job['id']}", params={"since": seen}).json()
        seen += len(snap["cards"])
        print(f"  progress {snap['progress'] * 100:5.1f}%  cards={seen:<3} {snap['message']}")
        if snap["status"] != "running":
            break
        time.sleep(1.0)
    wall = time.perf_counter() - t0
    snap = s.get(f"{base}/api/jobs/{job['id']}").json()
    cards = snap["cards"]
    print("stats:", snap["stats"])
    print("timings:", snap["timings"], f"| wall clock incl. polling: {wall:.1f}s")
    print("errors:", snap["errors"])
    for c in cards:
        if c["ocr_status"] != "OK" or c["pdf_name"] == "with_blank_page.pdf":
            print(f"  [{c['pdf_name']} #{c['card_number']}] {c['ocr_status']} dup={c['duplicate_status']!r} "
                  f"notes={c['notes']} name={c['name']!r}")

    # image endpoint
    img = s.get(f"{base}/api/jobs/{job['id']}/cards/{cards[0]['id']}/image")
    print("card image:", img.status_code, img.headers.get("content-type"), len(img.content), "bytes")

    # Simulate the reviewer: tick everything that isn't Failed / Duplicate, fix one value, enter one bad email.
    rows = []
    for c in cards:
        include = c["ocr_status"] != "Failed" and c["duplicate_status"] != "Duplicate"
        row = {k: c[k] for k in ("id", "name", "address", "telephone", "email")} | {
            "include": include, "dup_ack": c["duplicate_status"]}
        rows.append(row)
    edited = next(r for r in rows if r["include"])
    edited["name"] = edited["name"] + " (verified)"
    bad = next(r for r in rows if r["include"] and r is not edited)
    bad["email"] = "broken@@example"
    res = s.post(f"{base}/api/jobs/{job['id']}/save", json={"rows": rows}).json()
    print(f"save: saved={res['saved']} errors={res['errors']} skipped={len(res['skipped'])} db_seconds={res['db_seconds']}")

    # Fix the bad email and save again (the already-saved rows are updated, not duplicated).
    bad["email"] = "fixed@example.com"
    res2 = s.post(f"{base}/api/jobs/{job['id']}/save", json={"rows": rows}).json()
    print(f"re-save: saved={res2['saved']} errors={res2['errors']} skipped={len(res2['skipped'])}")

    # Verify DB == GUI
    gui = {c["db_id"]: c for c in res2["cards"] if c["db_id"]}
    db_rows = s.get(f"{base}/api/records", params={"limit": 1000}).json()["rows"]
    db_by_id = {r["id"]: r for r in db_rows}
    mismatches = 0
    for db_id, c in gui.items():
        r = db_by_id.get(db_id)
        for k in ("pdf_name", "card_number", "name", "address", "telephone", "email"):
            if r is None or r[k] != c[k]:
                mismatches += 1
                print(f"  MISMATCH #{db_id} {k}: gui={c[k]!r} db={(r or {}).get(k)!r}")
    print(f"DB check: {len(gui)} saved cards in GUI, {len(db_rows)} rows in DB, {mismatches} mismatches")
    edited_card = next(c for c in res2["cards"] if c["id"] == edited["id"])
    print("edited row stored as:", db_by_id[edited_card["db_id"]]["name"])
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
