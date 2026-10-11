# """Central configuration loaded from environment variables / .env."""
# from __future__ import annotations

# import os
# from dataclasses import dataclass, field
# from pathlib import Path

# from dotenv import load_dotenv

# BASE_DIR = Path(__file__).resolve().parent
# load_dotenv(BASE_DIR / ".env")

# # Windows Smart App Control can block SQLAlchemy's unsigned Cython DLLs; the
# # pure-Python implementation is functionally identical and fast enough here.
# os.environ.setdefault("DISABLE_SQLALCHEMY_CEXT_RUNTIME", "1")


# def _bool(name: str, default: bool) -> bool:
#     raw = os.getenv(name)
#     if raw is None or raw.strip() == "":
#         return default
#     return raw.strip().lower() in {"1", "true", "yes", "on"}


# def _int(name: str, default: int) -> int:
#     raw = os.getenv(name, "").strip()
#     try:
#         return int(raw) if raw else default
#     except ValueError:
#         return default


# def _float(name: str, default: float) -> float:
#     raw = os.getenv(name, "").strip()
#     try:
#         return float(raw) if raw else default
#     except ValueError:
#         return default


# def _database_url() -> str:
#     """Return a usable SQLAlchemy URL.

#     Supabase project URLs (https://...) are not valid database connection strings.
#     If the value is missing or invalid, keep the app running locally with SQLite
#     instead of crashing at import time.
#     """
#     url = os.getenv("DATABASE_URL", "").strip()
#     if not url:
#         return f"sqlite:///{(BASE_DIR / 'data' / 'business_cards.db').as_posix()}"
#     if url.startswith(("http://", "https://")):
#         return f"sqlite:///{(BASE_DIR / 'data' / 'business_cards.db').as_posix()}"
#     for prefix in ("postgres://", "postgresql://"):
#         if url.startswith(prefix):
#             return "postgresql+psycopg://" + url[len(prefix):]
#     if url.startswith("postgresql+psycopg://"):
#         return url
#     return f"sqlite:///{(BASE_DIR / 'data' / 'business_cards.db').as_posix()}"


# @dataclass(frozen=True)
# class Settings:
#     # --- web ---
#     host: str = os.getenv("APP_HOST", "127.0.0.1")
#     port: int = _int("APP_PORT", 5000)
#     max_upload_mb: int = _int("MAX_UPLOAD_MB", 200)
#     secret_key: str = os.getenv("FLASK_SECRET_KEY", "") or os.urandom(24).hex()

#     # --- paths ---
#     upload_dir: Path = BASE_DIR / "uploads"
#     log_dir: Path = BASE_DIR / "logs"
#     data_dir: Path = BASE_DIR / "data"

#     # --- database ---
#     database_url: str = field(default_factory=_database_url)
#     db_table: str = os.getenv("DB_TABLE", "business_cards")
#     supabase_url: str = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
#     supabase_key: str = os.getenv("SUPABASE_ANON_KEY", "").strip()

#     # --- PDF / images ---
#     render_dpi: int = _int("RENDER_DPI", 250)
#     max_pages_per_pdf: int = _int("MAX_PAGES_PER_PDF", 500)
#     # Only tiny crops (long side, px) are upscaled; over-enlarging text makes the
#     # detector split lines into single words.
#     min_card_long_side: int = _int("MIN_CARD_LONG_SIDE", 700)
#     # Crops larger than this are downscaled (keeps OCR fast without hurting accuracy).
#     max_card_long_side: int = _int("MAX_CARD_LONG_SIDE", 2200)

#     # --- optional LLM extraction ---
#     extraction_provider: str = os.getenv("EXTRACTION_PROVIDER", "qwen").strip().lower()
#     qwen_api_key: str = os.getenv("OPENROUTER_API_KEY", os.getenv("GROQ_API_KEY", "")).strip()
#     qwen_model: str = os.getenv("QWEN_MODEL", "qwen/qwen-2.5-vl-72b-instruct").strip()
#     qwen_base_url: str = os.getenv(
#         "QWEN_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
#     ).strip()
#     qwen_timeout: int = _int("QWEN_TIMEOUT", 45)
#     qwen_rate_limit_retries: int = _int("QWEN_RATE_LIMIT_RETRIES", 2)
#     nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "").strip()
#     nvidia_model: str = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct").strip()
#     nvidia_base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions").strip()
#     nvidia_timeout: int = _int("NVIDIA_TIMEOUT", 60)
#     gemini_api_key: str = os.getenv("GEMINI_API_KEY", os.getenv("Gemini_API_KEY", "")).strip()
#     gemini_model: str = os.getenv("GEMINI_MODEL", os.getenv("model", "gemini-3.5-flash-lite")).strip()
#     gemini_timeout: int = _int("GEMINI_TIMEOUT", 45)

#     # --- extraction ---
#     # ISO region used to parse phone numbers written without a "+country" prefix,
#     # e.g. "PK", "US", "CN". Leave empty to accept them without region validation.
#     default_phone_region: str = os.getenv("DEFAULT_PHONE_REGION", "").strip().upper()
#     review_confidence: float = _float("REVIEW_CONFIDENCE", 0.80)

#     # --- pipeline ---
#     preprocess_workers: int = _int("PREPROCESS_WORKERS", min(4, os.cpu_count() or 2))
#     max_jobs_in_memory: int = _int("MAX_JOBS_IN_MEMORY", 10)

#     log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

#     @property
#     def is_sqlite(self) -> bool:
#         return self.database_url.startswith("sqlite")

#     @property
#     def has_supabase_rest(self) -> bool:
#         return bool(self.supabase_url and self.supabase_key and self.supabase_key != "******")


# settings = Settings()

# for _d in (settings.upload_dir, settings.log_dir, settings.data_dir):
#     _d.mkdir(parents=True, exist_ok=True)


"""Central configuration loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Windows compatibility for SQLAlchemy's Cython extensions.
os.environ.setdefault("DISABLE_SQLALCHEMY_CEXT_RUNTIME", "1")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _database_url() -> str:
    """Return a usable SQLAlchemy URL."""
    url = os.getenv("DATABASE_URL", "").strip()

    if not url or url.startswith(("http://", "https://")):
        return f"sqlite:///{(BASE_DIR / 'data' / 'business_cards.db').as_posix()}"

    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]

    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]

    if url.startswith("postgresql+psycopg://"):
        return url

    return f"sqlite:///{(BASE_DIR / 'data' / 'business_cards.db').as_posix()}"


# Vercel functions have a read-only application directory.
# Use /tmp for temporary files on Vercel.
IS_VERCEL = bool(os.getenv("VERCEL"))

if IS_VERCEL:
    RUNTIME_DIR = Path("/tmp/business_cards_app")
else:
    RUNTIME_DIR = BASE_DIR


@dataclass(frozen=True)
class Settings:
    # --- web ---
    host: str = os.getenv("APP_HOST", "127.0.0.1")
    port: int = _int("APP_PORT", 5000)
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 200)
    secret_key: str = os.getenv("FLASK_SECRET_KEY", "") or os.urandom(24).hex()

    # --- paths ---
    upload_dir: Path = RUNTIME_DIR / "uploads"
    log_dir: Path = RUNTIME_DIR / "logs"
    data_dir: Path = RUNTIME_DIR / "data"

    # --- database ---
    database_url: str = field(default_factory=_database_url)
    db_table: str = os.getenv("DB_TABLE", "business_cards")
    supabase_url: str = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    supabase_key: str = os.getenv("SUPABASE_ANON_KEY", "").strip()

    # --- PDF / images ---
    render_dpi: int = _int("RENDER_DPI", 250)
    max_pages_per_pdf: int = _int("MAX_PAGES_PER_PDF", 500)
    min_card_long_side: int = _int("MIN_CARD_LONG_SIDE", 700)
    max_card_long_side: int = _int("MAX_CARD_LONG_SIDE", 2200)

    # --- optional LLM extraction ---
    extraction_provider: str = os.getenv(
        "EXTRACTION_PROVIDER", "qwen"
    ).strip().lower()
    qwen_api_key: str = os.getenv(
        "OPENROUTER_API_KEY", os.getenv("GROQ_API_KEY", "")
    ).strip()
    qwen_model: str = os.getenv(
        "QWEN_MODEL", "qwen/qwen-2.5-vl-72b-instruct"
    ).strip()
    qwen_base_url: str = os.getenv(
        "QWEN_BASE_URL",
        "https://openrouter.ai/api/v1/chat/completions",
    ).strip()
    qwen_timeout: int = _int("QWEN_TIMEOUT", 45)
    qwen_rate_limit_retries: int = _int("QWEN_RATE_LIMIT_RETRIES", 2)

    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "").strip()
    nvidia_model: str = os.getenv(
        "NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct"
    ).strip()
    nvidia_base_url: str = os.getenv(
        "NVIDIA_BASE_URL",
        "https://integrate.api.nvidia.com/v1/chat/completions",
    ).strip()
    nvidia_timeout: int = _int("NVIDIA_TIMEOUT", 60)

    gemini_api_key: str = os.getenv(
        "GEMINI_API_KEY", os.getenv("Gemini_API_KEY", "")
    ).strip()
    gemini_model: str = os.getenv(
        "GEMINI_MODEL", os.getenv("model", "gemini-3.5-flash-lite")
    ).strip()
    gemini_timeout: int = _int("GEMINI_TIMEOUT", 45)

    # --- extraction ---
    default_phone_region: str = os.getenv(
        "DEFAULT_PHONE_REGION", ""
    ).strip().upper()
    review_confidence: float = _float("REVIEW_CONFIDENCE", 0.80)

    # --- pipeline ---
    preprocess_workers: int = _int(
        "PREPROCESS_WORKERS", min(4, os.cpu_count() or 2)
    )
    max_jobs_in_memory: int = _int("MAX_JOBS_IN_MEMORY", 10)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def has_supabase_rest(self) -> bool:
        return bool(
            self.supabase_url
            and self.supabase_key
            and self.supabase_key != "******"
        )


settings = Settings()

# Create runtime directories only in writable storage.
for _d in (settings.upload_dir, settings.log_dir, settings.data_dir):
    _d.mkdir(parents=True, exist_ok=True)
