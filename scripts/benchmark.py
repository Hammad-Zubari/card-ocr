"""Run the real pipeline headless, report timings, and score against ground truth.

    python scripts/benchmark.py samples/cards_one_per_page.pdf [more.pdf ...] [--save] [--verbose]

--save  also writes every non-failed card to the database and verifies that the
        stored rows match the extracted values (end-to-end DB check).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.logging_setup import setup_logging  # noqa: E402

setup_logging()

from database import db  # noqa: E402
from ocr.paddle_ocr_engine import OCREngine  # noqa: E402
from services import processing_service as ps  # noqa: E402


def digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", (s or "").lower()).strip()


def score_card(card: dict, truth: dict) -> dict:
    # Bilingual cards may legitimately yield "王伟", "Wang Wei" or "王伟 Wang Wei".
    alt = truth.get("alt_name", "")
    names = {norm(truth["name"]), norm(alt), norm(f"{truth['name']} {alt}")} - {""}
    compact = lambda s: s.replace(" ", "")  # noqa: E731
    got_name = norm(card["name"])
    name_ok = compact(got_name) in {compact(n) for n in names}
    email_ok = (card["email"].lower() == truth["email"].lower())
    want_phones = {digits(p)[-9:] for p in truth["telephone"]}
    got_phones = {digits(p)[-9:] for p in card["telephone"].split(";") if digits(p)}
    phone_ok = want_phones == got_phones
    if truth["address"]:
        addr_ratio = SequenceMatcher(None, norm(card["address"]), norm(truth["address"])).ratio()
        addr_ok = addr_ratio >= 0.85
    else:
        addr_ratio, addr_ok = (1.0 if not card["address"] else 0.0), not card["address"]
    return {"name": name_ok, "email": email_ok, "telephone": phone_ok, "address": addr_ok,
            "address_ratio": round(addr_ratio, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gt_path = ROOT / "samples" / "ground_truth.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}

    t = time.perf_counter()
    OCREngine.get().ensure_loaded()
    load = time.perf_counter() - t
    print(f"OCR engine: {OCREngine.get().info()}  (load {load:.1f}s, excluded from job time)")

    job = ps.create_job([(Path(p).name, Path(p)) for p in args.pdfs])
    ps.start_job(job)
    while job.status == "running":
        time.sleep(0.2)
    snap = job.snapshot()
    print(f"\nStatus: {snap['status']} - {snap['message']}")
    print("Stats:", json.dumps(snap["stats"]))
    print("Timings (s):", json.dumps(snap["timings"]))
    n = max(1, snap["stats"]["cards_processed"])
    print(f"Per card: {snap['timings'].get('total', 0) / n:.3f}s  |  OCR per card: {snap['timings'].get('ocr', 0) / n:.3f}s")
    for e in snap["errors"]:
        print("ERROR:", e)

    totals = {"name": 0, "email": 0, "telephone": 0, "address": 0}
    scored = 0
    for card in snap["cards"]:
        truth_list = gt.get(card["pdf_name"]) or []
        truth = next((t for t in truth_list if t["card"] == card["card_number"]), None)
        res = score_card(card, truth) if truth else None
        if res:
            scored += 1
            for k in totals:
                totals[k] += int(res[k])
        bad = res and not all(res[k] for k in totals)
        if args.verbose or bad or card["ocr_status"] == "Failed":
            print(f"\n[{card['pdf_name']} #{card['card_number']} p{card['page_number']} {card['detection']}] "
                  f"{card['ocr_status']} conf={card['ocr_confidence']} dup={card['duplicate_status']!r} notes={card['notes']}")
            print(f"   name={card['name']!r}  title={card['job_title']!r}  company={card['company']!r}")
            print(f"   tel={card['telephone']!r}  email={card['email']!r}")
            print(f"   addr={card['address']!r}")
            if res:
                print(f"   score={res}")
                if bad:
                    print(f"   want name={truth['name']!r} tel={truth['telephone']} email={truth['email']!r}")
                    print(f"        addr={truth['address']!r}")
                    print("   RAW:", card["raw_ocr_text"].replace("\n", " | "))
    if scored:
        print(f"\nField accuracy over {scored} ground-truth cards:")
        for k, v in totals.items():
            print(f"   {k:<10} {v}/{scored}  ({100 * v / scored:.0f}%)")

    if args.save:
        db.init_db()
        rows = [{**c, "include": True, "dup_ack": c["duplicate_status"]} for c in snap["cards"]
                if c["ocr_status"] != "Failed"]
        result = ps.save_rows(job, rows)
        print(f"\nDB save: saved={result['saved']} skipped={len(result['skipped'])} errors={result['errors']} "
              f"in {result['db_seconds']}s  (backend={db.backend_name()})")
        for s in result["skipped"]:
            print("   skipped:", s)
        mismatches = 0
        stored, _ = db.list_records(limit=10000)
        by_id = {r["id"]: r for r in stored}
        for c in result["cards"]:
            if not c["db_id"]:
                continue
            r = by_id.get(c["db_id"])
            for k in ("name", "address", "telephone", "email", "pdf_name", "card_number"):
                if r is None or r[k] != c[k]:
                    mismatches += 1
                    print(f"   MISMATCH id={c['db_id']} {k}: gui={c[k]!r} db={(r or {}).get(k)!r}")
        print(f"DB verification: {result['saved']} rows checked, {mismatches} field mismatches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
