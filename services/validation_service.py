"""Cleaning / normalisation / validation of extracted and user-edited fields.

The rule here is "fix OCR noise, never rewrite real data": we normalise
whitespace, Unicode width, labels and obvious OCR artefacts, but keep the
user's spelling and capitalisation.
"""
from __future__ import annotations

import re
import unicodedata

import phonenumbers

from config import settings

EMAIL_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,24}"
)
_EMAIL_FULL = re.compile(rf"^{EMAIL_RE.pattern}$")

_CONTROL = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f​-‏﻿]")
_EDGE_JUNK = re.compile(r"^[\s|•·*~_=,;:\-–—>»<«]+|[\s|•·*~_=,;:\-–—<«>»]+$")
_REPEAT_PUNCT = re.compile(r"([,;:.\-])(\s*\1)+")
_SPACES = re.compile(r"[ \t 　]+")

_ADDRESS_LABEL = re.compile(r"^(?:address|addr|add|adresse|location|地址)\s*[:.：\-]?\s*", re.I)
_NAME_LABEL = re.compile(r"^(?:name|姓名)\s*[:.：\-]\s*", re.I)


def clean_text(text: str | None) -> str:
    """Generic OCR-noise cleanup for a single field value."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = _CONTROL.sub("", t)
    t = t.replace("\r", "\n")
    t = " ".join(part.strip() for part in t.split("\n") if part.strip())
    t = _SPACES.sub(" ", t)
    t = _REPEAT_PUNCT.sub(r"\1", t)
    t = re.sub(r"\s+([,;.:])", r"\1", t)
    t = _EDGE_JUNK.sub("", t)
    return t.strip()


def clean_name(text: str | None) -> str:
    t = _NAME_LABEL.sub("", clean_text(text))
    t = re.sub(r"[^\w\s.'\-,()]", "", t)   # drop stray symbols (©, |, etc.) but keep Unicode letters
    t = re.sub(r"(?<=[一-鿿])(?=[A-Za-z])|(?<=[A-Za-z])(?=[一-鿿])", " ", t)  # "李娜Li Na"
    return _SPACES.sub(" ", t).strip(" ,-_")


def clean_address(text: str | None) -> str:
    t = clean_text(text)
    t = _ADDRESS_LABEL.sub("", t)
    t = re.sub(r"(,\s*){2,}", ", ", t)
    # OCR sometimes drops spaces: "Zerktouni,Casablanca" / "18Boulevard". Only touch
    # unambiguous spots ("3rd", "15F", "45-B" stay as they are).
    t = re.sub(r",(?=[^\s\d])", ", ", t)
    t = re.sub(r"(?<=\d)(?=[A-Z][a-z]{2,})", " ", t)
    # German "ß" is commonly misread as "B"/"lB" in "...straße".
    t = re.sub(r"(?<=[a-z])stra[lI1]?B(?=e\b)", "straß", t)
    return t.strip(" ,;")


def normalize_email(text: str | None) -> str:
    """Return a normalised email or '' if the value isn't a plausible email."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).strip()
    t = re.sub(r"\s*[\(\[\{]\s*at\s*[\)\]\}]\s*", "@", t, flags=re.I)
    t = re.sub(r"\s*[\(\[\{]\s*dot\s*[\)\]\}]\s*", ".", t, flags=re.I)
    t = re.sub(r"^(?:e-?mail|mail|email id|e)\s*[:.：\-]\s*", "", t, flags=re.I)
    t = t.replace(" ", "").strip(".,;:")
    if "@" in t:
        local, _, domain = t.rpartition("@")
        domain = domain.replace(",", ".").lower()
        domain = re.sub(r"\.{2,}", ".", domain)
        # OCR reads lowercase "l" as "I"; in an otherwise lowercase address the
        # capital I is almost certainly an "l" (emails are case-insensitive anyway).
        if "I" in local and not re.search(r"[A-HJ-Z]", local):
            local = local.replace("I", "l")
        t = f"{local.lower()}@{domain}"
    m = EMAIL_RE.search(t)
    return m.group(0) if m else ""


def is_valid_email(text: str) -> bool:
    return bool(_EMAIL_FULL.match(text or ""))


def phone_digits(text: str | None) -> str:
    return re.sub(r"\D", "", text or "")


def format_phone(raw: str, region: str | None = None) -> str:
    """Format a phone number in international format when it can be parsed;
    otherwise return the cleaned original (never invent digits)."""
    raw = clean_text(raw)
    candidate = re.sub(r"^00", "+", raw.strip())
    region = region if region is not None else (settings.default_phone_region or None)
    try:
        if candidate.startswith("+") or region:
            num = phonenumbers.parse(candidate, None if candidate.startswith("+") else region)
            if phonenumbers.is_possible_number(num):
                return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
    except phonenumbers.NumberParseException:
        pass
    t = re.sub(r"\s*([\-/.])\s*", r"\1", raw)
    t = re.sub(r"[^\d+()\-./ ]", "", t)
    t = re.sub(r"\)(?=\d)", ") ", t)
    return _SPACES.sub(" ", t).strip(" -./")


def clean_telephone(text: str | None) -> str:
    """Clean a (possibly multi-number, '; '-separated) telephone field."""
    if not text:
        return ""
    parts = [p for p in re.split(r"\s*[;,]\s*|\s+/\s+", clean_text(text)) if p]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        d = phone_digits(p)
        if len(d) < 5:
            continue
        f = format_phone(p)
        key = phone_digits(f)[-9:]
        if key not in seen:
            seen.add(key)
            out.append(f)
    return "; ".join(out)


def clean_email_field(text: str | None) -> tuple[str, list[str]]:
    """Clean a (possibly multi-email) field. Returns (value, invalid parts)."""
    if not text:
        return "", []
    good, bad = [], []
    for part in re.split(r"[;,\s]+", text.strip()):
        if not part:
            continue
        e = normalize_email(part)
        (good if e and is_valid_email(e) else bad).append(e or part)
    return "; ".join(dict.fromkeys(good)), bad


def validate_record(name: str, address: str, telephone: str, email: str,
                    mobile_phone: str = "") -> tuple[dict, list[str]]:
    """Clean user-reviewed values before saving. Returns (clean dict, issues)."""
    issues: list[str] = []
    email_clean, bad = clean_email_field(email)
    if bad:
        issues.append(f"invalid email: {', '.join(bad)}")
    rec = {
        "name": clean_name(name) or clean_text(name),
        "address": clean_address(address),
        "telephone": clean_telephone(telephone) or clean_text(telephone),
        "mobile_phone": clean_telephone(mobile_phone) or clean_text(mobile_phone),
        "email": email_clean,
    }
    return rec, issues
