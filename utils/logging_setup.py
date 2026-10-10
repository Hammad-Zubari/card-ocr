"""Logging configuration with secret redaction.

Log lines identify the PDF / page / card via the `ctx()` helper, e.g.
    log.info("%s OCR done in %.2fs", ctx(pdf, page, card), dt)
"""
from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler

from config import settings

# user:password@host in connection strings, and key=value style secrets.
_URL_PASSWORD = re.compile(r"(\w+://[^:/\s@]+:)([^@\s]+)(@)")
_KEY_VALUE = re.compile(r"(?i)\b(password|passwd|pwd|api[_-]?key|secret|token)(\s*[=:]\s*)(\S+)")


class RedactSecretsFilter(logging.Filter):
    """Masks passwords / API keys that might end up in a log message."""

    def __init__(self) -> None:
        super().__init__()
        # Literal secret values from the environment that must never be logged.
        self._literals = [
            v for k, v in os.environ.items()
            if v and len(v) >= 8 and re.search(r"(?i)(password|secret|key|token|database_url)", k)
        ]

    def _redact(self, text: str) -> str:
        text = _URL_PASSWORD.sub(r"\1***\3", text)
        text = _KEY_VALUE.sub(r"\1\2***", text)
        for lit in self._literals:
            text = text.replace(lit, "***")
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # malformed args: leave the record untouched
            return True
        redacted = self._redact(msg)
        if redacted != msg:
            record.msg, record.args = redacted, None
        return True


def ctx(pdf: str | None = None, page: int | None = None, card: int | None = None) -> str:
    parts = []
    if pdf:
        parts.append(f"pdf={pdf}")
    if page is not None:
        parts.append(f"page={page}")
    if card is not None:
        parts.append(f"card={card}")
    return "[" + " ".join(parts) + "]" if parts else ""


_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    redact = RedactSecretsFilter()

    file_handler = RotatingFileHandler(
        settings.log_dir / "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    console = logging.StreamHandler()
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    for h in (file_handler, console):
        h.setFormatter(fmt)
        h.addFilter(redact)
        root.addHandler(h)

    # Third-party libraries are chatty at INFO.
    for noisy in ("ppocr", "paddle", "paddlex", "werkzeug", "urllib3", "httpx", "httpx2", "httpcore",
                  "huggingface_hub", "modelscope", "filelock", "PIL", "waitress"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True
