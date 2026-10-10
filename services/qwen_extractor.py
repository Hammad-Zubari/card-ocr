"""Qwen Vision extraction for one isolated business-card image at a time."""
from __future__ import annotations

import json
import logging
import re
import base64
import os
import time
from dataclasses import dataclass
from urllib import error, request

from config import settings
from ocr.types import ExtractedFields

log = logging.getLogger(__name__)

_FIELDS = ("name", "company_name", "job_title", "address", "telephone", "mobile_phone", "email")
_AUX_FIELDS = ("company", "website")


def _numbers(value: str) -> str:
    parts = [part.strip() for part in re.split(r"[;\n|]+", value) if part.strip()]
    return ", ".join(dict.fromkeys(parts))


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return ""


def _parse_object(content) -> dict:
    """Parse provider JSON, including fenced/embedded responses from VLMs."""
    text = _content_text(content).strip()
    if not text:
        raise ValueError("empty model response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        value, _ = decoder.raw_decode(text[start:])
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, dict)), None)
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    return value


@dataclass
class VisionResult:
    fields: ExtractedFields
    raw_text: str = ""
    error: str | None = None
    company: str = ""
    website: str = ""


def info() -> dict:
    return {
        "configured": bool(settings.qwen_api_key),
        "model": settings.qwen_model,
        "provider": "openrouter",
        "base_url": settings.qwen_base_url,
    }


def extract(raw_text: str) -> ExtractedFields:
    if not settings.qwen_api_key:
        raise RuntimeError("Qwen is not configured: set GROQ_API_KEY")
    if not raw_text.strip():
        raise ValueError("Qwen extraction requires OCR text")

    prompt = (
        "Extract exactly one business card from the OCR text below. Return only a JSON object "
        "with these keys: name, company_name, job_title, address, telephone, mobile_phone, email. "
        "Use empty strings for missing values. Translate Urdu, Arabic, Sindhi, or other non-English "
        "person, company, designation, and address text into clear English. Keep names close "
        "to their pronunciation; never translate phone numbers or email addresses. Do not invent "
        "or combine values. Preserve the "
        "printed order and spelling as much as possible. name is the person's name and "
        "job_title is the designation. telephone is office/landline and mobile_phone is mobile/cell. "
        "Keep them separate. Extract the printed company into company_name. OCR text is from one card only.\n\nOCR TEXT:\n" + raw_text
    )
    payload = {
        "model": settings.qwen_model,
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You extract contact fields accurately from business cards."},
            {"role": "user", "content": prompt},
        ],
    }
    req = request.Request(
        settings.qwen_base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + settings.qwen_api_key,
            "Content-Type": "application/json",
            "User-Agent": "Visitors-OCR/1.0",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://127.0.0.1:5000"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Visitors-OCR"),
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=settings.qwen_timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Qwen API returned HTTP {exc.code}: {detail}") from exc
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Qwen API connection failed: {exc}") from exc

    try:
        content = body["choices"][0]["message"]["content"]
        data = _parse_object(content)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Qwen returned an invalid structured response") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Qwen returned a non-object response")
    values = {key: str(data.get(key) or "").strip() for key in _FIELDS}
    values["telephone"] = _numbers(values["telephone"])
    values["mobile_phone"] = _numbers(values["mobile_phone"])
    # Qwen must not invent contact data or move a neighbouring value into a
    # field. Paddle's OCR/regex result remains authoritative for contacts.
    compact_text = re.sub(r"\s+", "", raw_text).lower()
    for key in ("email", "telephone", "mobile_phone"):
        if values[key] and re.sub(r"\s+", "", values[key]).lower() not in compact_text:
            values[key] = ""
    return ExtractedFields(**values)


FALLBACK_MODELS = (
    "qwen/qwen-2.5-vl-72b-instruct",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen-2.5-72b-instruct",
)


def extract_image(image) -> VisionResult:
    """Extract one already-isolated card image with Qwen vision."""
    if not settings.qwen_api_key:
        raise RuntimeError("Qwen is not configured: set OPENROUTER_API_KEY in .env")
    import cv2
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > 1100:
        scale = 1100 / longest
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise RuntimeError("Could not encode card image")
    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
    prompt = (
        "This image contains exactly ONE business card. Extract only visible text in this card. "
        "Return a JSON object with keys: name, company_name, job_title, address, telephone, mobile_phone, email, website. "
        "Use empty strings for missing fields. Translate Urdu, Arabic, Sindhi, or other non-English "
        "person, company, designation, and address text into clear English. Keep names close "
        "to their pronunciation; never translate phone numbers or email addresses. "
        "name is the person's full name, job_title is designation/role, "
        "telephone is office/landline, mobile_phone is mobile/cell. "
        "Keep them separate. Multiple numbers must be comma-separated. "
        "Copy the printed website/domain exactly; do not infer one from an email."
    )

    models_to_try = [settings.qwen_model]
    for fm in FALLBACK_MODELS:
        if fm not in models_to_try:
            models_to_try.append(fm)

    last_error = None
    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "temperature": 0.1,
            "max_tokens": 700,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a precise business-card vision extractor."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}", "detail": "high"
                    }},
                ]},
            ],
        }
        try:
            body = _post(payload)
            content = body["choices"][0]["message"]["content"]
            data = _parse_object(content)
            values = {key: str(data.get(key) or "").strip() for key in _FIELDS}
            values["telephone"] = _numbers(values["telephone"])
            values["mobile_phone"] = _numbers(values["mobile_phone"])
            raw_text = str(data.get("raw_text") or "").strip()
            return VisionResult(
                ExtractedFields(**values), raw_text,
                company=str(data.get("company_name") or data.get("company") or "").strip(),
                website=str(data.get("website") or "").strip(),
            )
        except Exception as exc:
            last_error = f"{model_name}: {exc}"
            log.warning("Qwen vision model %s failed: %s, trying fallback...", model_name, exc)
            continue

    raise RuntimeError(f"Qwen extraction failed: {last_error}")


def _post(payload: dict) -> dict:
    req = request.Request(
        settings.qwen_base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + settings.qwen_api_key,
            "Content-Type": "application/json",
            "User-Agent": "Visitors-OCR/1.0",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://127.0.0.1:5000"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Visitors-OCR"),
        },
        method="POST",
    )
    attempts = max(0, settings.qwen_rate_limit_retries)
    for attempt in range(attempts + 1):
        try:
            with request.urlopen(req, timeout=settings.qwen_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code != 429 or attempt >= attempts:
                raise RuntimeError(f"Qwen API returned HTTP {exc.code}: {detail}") from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = min(20.0, max(1.0, float(retry_after))) if retry_after else 5.0
            except ValueError:
                delay = 5.0
            log.warning("Qwen rate limit reached; retrying in %.0fs (%d/%d)",
                        delay, attempt + 1, attempts)
            time.sleep(delay)
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Qwen API connection failed: {exc}") from exc


def check_connectivity() -> dict:
    """Perform a fast authenticated model request for the GUI health check."""
    if not settings.qwen_api_key:
        return info() | {"ok": False, "error": "OPENROUTER_API_KEY is not configured"}
    payload = {
        "model": settings.qwen_model,
        "temperature": 0,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Reply with: OK"}],
    }
    try:
        body = _post(payload)
        content = body["choices"][0]["message"]["content"]
        if not str(content).strip():
            raise RuntimeError("Qwen returned an empty response")
        return info() | {"ok": True, "error": None}
    except Exception as exc:
        log.warning("Qwen connectivity check failed: %s", exc)
        return info() | {"ok": False, "error": str(exc)}
