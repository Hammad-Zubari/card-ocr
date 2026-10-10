"""NVIDIA vision extraction for one isolated business-card image."""
from __future__ import annotations

import base64
import json
import logging
from urllib import error, request

from config import settings
from ocr.types import ExtractedFields
from services.qwen_extractor import _FIELDS, _numbers, _parse_object, VisionResult

log = logging.getLogger(__name__)

FALLBACK_MODELS = (
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
)


def info() -> dict:
    return {
        "configured": bool(settings.nvidia_api_key),
        "model": settings.nvidia_model,
        "provider": "nvidia",
        "base_url": settings.nvidia_base_url,
    }


def extract_image(image) -> VisionResult:
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA is not configured: set NVIDIA_API_KEY in .env")
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
        "This image contains exactly ONE business card. Return only a compact JSON object "
        "with keys: name, company_name, job_title, address, telephone, mobile_phone, email. Use empty strings "
        "for missing values. Translate Urdu, Arabic, Sindhi, or other non-English person, company, "
        "designation, and address text into clear English. Keep names close to their pronunciation; "
        "never translate phone numbers or email addresses. Extract only visible text and do not invent details. "
        "telephone is office/landline; mobile_phone is mobile/cell. Multiple numbers must be comma-separated."
    )

    models_to_try = [settings.nvidia_model]
    for fm in FALLBACK_MODELS:
        if fm not in models_to_try:
            models_to_try.append(fm)

    last_error = None
    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "temperature": 0.1,
            "max_tokens": 700,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        req = request.Request(
            settings.nvidia_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + settings.nvidia_api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "Visitors-OCR/1.0",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=settings.nvidia_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            data = _parse_object(content)
            values = {key: str(data.get(key) or "").strip() for key in _FIELDS}
            values["telephone"] = _numbers(values["telephone"])
            values["mobile_phone"] = _numbers(values["mobile_phone"])
            return VisionResult(
                ExtractedFields(**values),
                company=str(data.get("company_name") or data.get("company") or "").strip(),
            )
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            last_error = f"NVIDIA API ({model_name}) returned HTTP {exc.code}: {detail}"
            log.warning("NVIDIA model %s failed: %s, trying fallback...", model_name, last_error)
            continue
        except Exception as exc:
            last_error = f"NVIDIA API ({model_name}) error: {exc}"
            log.warning("NVIDIA model %s error: %s, trying fallback...", model_name, last_error)
            continue

    raise RuntimeError(f"NVIDIA extraction failed: {last_error}")


def check_connectivity() -> dict:
    """Run a small authenticated request so the GUI can verify NVIDIA access."""
    result = info()
    if not settings.nvidia_api_key:
        return result | {"ok": False, "error": "NVIDIA_API_KEY is not configured"}

    payload = {
        "model": settings.nvidia_model,
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
        "messages": [{"role": "user", "content": "Reply with: OK"}],
    }
    req = request.Request(
        settings.nvidia_base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + settings.nvidia_api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Visitors-OCR/1.0",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=min(20, settings.nvidia_timeout)) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if not str(content).strip():
            raise RuntimeError("NVIDIA returned an empty response")
        return result | {"ok": True, "error": None}
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return result | {"ok": False, "error": f"NVIDIA API returned HTTP {exc.code}: {detail}"}
    except Exception as exc:
        return result | {"ok": False, "error": f"NVIDIA connectivity failed: {exc}"}
