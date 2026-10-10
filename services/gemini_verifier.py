"""Gemini vision extraction and secondary verification."""
from __future__ import annotations

import base64
import json
import logging
from urllib import error, parse, request

from config import settings
from ocr.types import ExtractedFields
from services.qwen_extractor import _FIELDS, _numbers, _parse_object, VisionResult

log = logging.getLogger(__name__)
FIELDS = ("name", "job_title", "address", "telephone", "mobile_phone", "email")

FALLBACK_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)


def info() -> dict:
    return {
        "configured": bool(settings.gemini_api_key),
        "model": settings.gemini_model,
        "provider": "gemini",
    }


def _call_gemini(prompt: str, image_b64: str | None = None, model_name: str | None = None) -> dict:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    models_to_try = [model_name or settings.gemini_model]
    for fm in FALLBACK_MODELS:
        if fm not in models_to_try:
            models_to_try.append(fm)

    last_error = None
    meaningful_error = None
    for m in models_to_try:
        parts: list[dict] = [{"text": prompt}]
        if image_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{parse.quote(m, safe='')}:generateContent?"
            f"{parse.urlencode({'key': settings.gemini_api_key})}"
        )
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Visitors-OCR/1.0"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=settings.gemini_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["candidates"][0]["content"]["parts"][0]["text"]
            return _parse_object(content)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            current_error = f"Gemini ({m}) HTTP {exc.code}: {detail}"
            last_error = current_error
            if exc.code != 404:
                meaningful_error = current_error
            if exc.code == 429:
                log.warning("Gemini model %s is quota-limited: %s", m, current_error)
                raise RuntimeError(
                    "Gemini API quota/rate limit exhausted. Check Google AI Studio billing "
                    "or wait for the free-tier limit to reset."
                ) from exc
            log.warning("Gemini model %s failed with %s, trying fallback...", m, current_error)
            continue
        except Exception as exc:
            last_error = str(exc)
            log.warning("Gemini model %s error: %s, trying fallback...", m, exc)
            continue

    raise RuntimeError(
        f"All Gemini models failed. Last error: {meaningful_error or last_error}"
    )


def extract_image(image) -> VisionResult:
    """Extract fields from a business card image using Gemini Vision."""
    import cv2
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > 1200:
        scale = 1200 / longest
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("Could not encode card image for Gemini")
    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")

    prompt = (
        "This image contains exactly ONE business card. Extract only visible text from this card. "
        "Return a JSON object with keys: name, company_name, job_title, address, telephone, mobile_phone, email, website. "
        "Use empty strings (not null) for missing fields. Translate Urdu, Arabic, or other "
        "non-English text (including Urdu, Arabic, or Sindhi) into clear English for person, company, designation, and address fields. "
        "Keep names close to their pronunciation and never translate phone numbers or emails. "
        "name: person's full name. "
        "job_title: designation or role. "
        "address: office or physical address. "
        "telephone: landline / office number(s). "
        "mobile_phone: cell / mobile number(s). "
        "email: email address(es). website: printed website/domain, if visible. "
        "If multiple numbers exist in a field, join them with comma. Do not invent details."
    )
    data = _call_gemini(prompt, image_data)
    values = {key: str(data.get(key) or "").strip() for key in _FIELDS}
    values["telephone"] = _numbers(values["telephone"])
    values["mobile_phone"] = _numbers(values["mobile_phone"])
    return VisionResult(
        ExtractedFields(**values),
        company=str(data.get("company_name") or data.get("company") or "").strip(),
        website=str(data.get("website") or "").strip(),
    )


def verify(image, extracted: dict) -> tuple[dict, list[str]]:
    """Strictly verify extracted fields against the isolated card image to prevent cross-card overlap and fix errors."""
    if not settings.gemini_api_key:
        return extracted, []
    import cv2
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > 1200:
        scale = 1200 / longest
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("Could not encode card for Gemini verification")
    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")

    prompt = (
        "You are an expert AI verification system for business cards at Ecommerce Gateway.\n"
        "Inspect this ONE business card crop image and strictly verify the extracted contact data.\n\n"
        "STRICT VERIFICATION REQUIREMENTS:\n"
        "1. STRICT ISOLATION: Ensure EVERY extracted field belongs SOLELY to this specific card. Never include data, names, or contacts from neighbouring cards or external context.\n"
        "2. EMAIL VERIFICATION: Carefully check the email address character-by-character against the card. Fix common OCR mistakes (e.g. 'rn' vs 'm', 'I' vs 'l', '.' vs ',').\n"
        "3. PHONE ACCURACY: Distinguish landline (telephone) and cell/WhatsApp (mobile_phone). Clean up numbers.\n"
        "4. NAME & DESIGNATION: Verify the person's name is in 'name' (do NOT put company name as person name), and role/title is in 'job_title'. Translate Urdu, Arabic, Sindhi, and other non-English text to English.\n"
        "5. MISSING VALUES: If a field is not visibly printed on this card, return empty string \"\" (do NOT invent or hallucinate).\n\n"
        "Return ONLY a JSON object with keys: name, job_title, address, telephone, mobile_phone, email, issues (array of strings explaining any corrections made).\n\n"
        "RAW EXTRACTED DATA TO VERIFY:\n" + json.dumps({k: extracted.get(k, "") for k in FIELDS}, ensure_ascii=False)
    )
    try:
        data = _call_gemini(prompt, image_data)
        result = dict(extracted)
        for key in FIELDS:
            if key in data:
                val = str(data.get(key) or "").strip()
                result[key] = val
        result["telephone"] = _numbers(result["telephone"])
        result["mobile_phone"] = _numbers(result["mobile_phone"])
        issues = data.get("issues") or []
        if isinstance(issues, str):
            issues = [issues]
        return result, [str(x).strip() for x in issues if str(x).strip()]
    except Exception as exc:
        log.warning("Gemini verification call failed: %s", exc)
        return extracted, [f"Verification fallback: {exc}"]


def check_connectivity() -> dict:
    """Run a small authenticated request to verify Gemini access."""
    result = info()
    if not settings.gemini_api_key:
        return result | {"ok": False, "error": "GEMINI_API_KEY is not configured"}
    try:
        resp = _call_gemini('{"status": "ok"} Return only {"status": "ok"}')
        return result | {"ok": True, "error": None}
    except Exception as exc:
        return result | {"ok": False, "error": str(exc)}
