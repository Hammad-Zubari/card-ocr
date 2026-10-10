"""Duplicate detection. Never deletes anything; it only labels records.

  Duplicate           same email, or same name + same phone, or same name + same address
  Possible Duplicate  same phone only (offices share switchboard numbers), or
                      very similar name + similar address
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from database import db
from services.validation_service import phone_digits

log = logging.getLogger(__name__)

DUPLICATE = "Duplicate"
POSSIBLE = "Possible Duplicate"


@dataclass
class DupResult:
    status: str = ""
    reason: str = ""


def _norm(text: str | None) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = re.sub(r"\b(mr|mrs|ms|miss|dr|prof|eng|engr)\.?\s+", "", t)
    return re.sub(r"[^\w]+", " ", t).strip()


def _emails(rec: dict) -> set[str]:
    return {e.strip().lower() for e in re.split(r"[;,\s]+", rec.get("email") or "") if "@" in e}


def _phones(rec: dict) -> set[str]:
    out = set()
    for p in re.split(r"[;,/]", ";".join(
        filter(None, (rec.get("telephone") or "", rec.get("mobile_phone") or "")))):
        d = phone_digits(p)
        if len(d) >= 7:
            out.add(d[-9:])  # compare without country/trunk prefix
    return out


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _compare(a: dict, b: dict) -> DupResult:
    na, nb = _norm(a.get("name")), _norm(b.get("name"))
    same_name = bool(na) and na == nb
    if _emails(a) & _emails(b):
        return DupResult(DUPLICATE, "same email")
    phone_match = bool(_phones(a) & _phones(b))
    if phone_match and (same_name or _similar(na, nb) >= 0.85):
        return DupResult(DUPLICATE, "same name and phone")
    aa, ab = _norm(a.get("address")), _norm(b.get("address"))
    if same_name and aa and aa == ab:
        return DupResult(DUPLICATE, "same name and address")
    if phone_match:
        return DupResult(POSSIBLE, "same phone")
    if na and nb and _similar(na, nb) >= 0.88 and aa and ab and _similar(aa, ab) >= 0.85:
        return DupResult(POSSIBLE, "similar name and address")
    return DupResult()


def _better(cur: DupResult, new: DupResult) -> bool:
    rank = {"": 0, POSSIBLE: 1, DUPLICATE: 2}
    return rank[new.status] > rank[cur.status]


def find_duplicates(records: list[dict], check_db: bool = True) -> tuple[list[DupResult], str | None]:
    """Label each record against earlier records in the list and against the DB.

    Records may carry `db_id` (already saved) so they don't match themselves.
    Returns (results, db_error) — a DB failure degrades to batch-only checking.
    """
    results = [DupResult() for _ in records]

    for i, rec in enumerate(records):
        for j in range(i):
            r = _compare(rec, records[j])
            if r.status and _better(results[i], r):
                label = f"card {records[j].get('card_number')} of {records[j].get('pdf_name')}"
                results[i] = DupResult(r.status, f"{r.reason} as {label}")
                if r.status == DUPLICATE:
                    break

    db_error = None
    if check_db:
        emails = set().union(*(_emails(r) for r in records)) if records else set()
        phones = set().union(*(_phones(r) for r in records)) if records else set()
        names = {(r.get("name") or "").strip().lower() for r in records if (r.get("name") or "").strip()}
        try:
            existing = db.dedupe_candidates(emails, phones, names)
        except db.DatabaseError as exc:
            log.warning("Duplicate check against database skipped: %s", exc)
            existing, db_error = [], str(exc)
        for i, rec in enumerate(records):
            for row in existing:
                if rec.get("db_id") and rec["db_id"] == row["id"]:
                    continue
                r = _compare(rec, row)
                if r.status and _better(results[i], r):
                    results[i] = DupResult(r.status, f"{r.reason} as saved record #{row['id']}")
                    if r.status == DUPLICATE:
                        break
    return results, db_error
