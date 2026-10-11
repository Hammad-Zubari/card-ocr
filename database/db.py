"""Database access for the Supabase/PostgreSQL business-card table."""
from __future__ import annotations

import logging
import json
import threading
import urllib.parse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, func, or_, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from database.models import Base, BusinessCard

log = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker | None = None
_init_lock = threading.Lock()

SAVE_FIELDS = ("name", "company_name", "job_title", "address", "telephone", "mobile_phone", "email",
               "business_type", "business_confidence")


class DatabaseError(Exception):
    """A user-presentable database failure (no credentials in the message)."""


def safe_url() -> str:
    return make_url(settings.database_url).render_as_string(hide_password=True)


def backend_name() -> str:
    if settings.has_supabase_rest:
        return "supabase-rest"
    return "sqlite" if settings.is_sqlite else make_url(settings.database_url).get_backend_name()


def _supabase_request(method: str, path: str, payload: dict | None = None) -> list[dict]:
    url = f"{settings.supabase_url}/rest/v1/{settings.db_table}{path}"
    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Accept": "application/json",
        "Prefer": "return=representation",
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    try:
        with urlopen(Request(url, data=data, headers=headers, method=method), timeout=15) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else []
    except (HTTPError, URLError, TimeoutError) as exc:
        detail = ""
        if isinstance(exc, HTTPError):
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise DatabaseError(f"Supabase save failed: {detail or exc}") from exc


def _save_cards_supabase(rows: list[dict]) -> list[int]:
    ids: list[int] = []
    for row in rows:
        values = {k: row.get(k) for k in SAVE_FIELDS}
        try:
            if row.get("db_id"):
                result = _supabase_request(
                    "PATCH", f"?id=eq.{int(row['db_id'])}", values
                )
            else:
                result = _supabase_request("POST", "", values)
        except DatabaseError as exc:
            # Older Supabase tables may not have the optional score column yet.
            # Keep card saving working until the migration is applied.
            if "business_confidence" not in str(exc):
                raise
            values.pop("business_confidence", None)
            if row.get("db_id"):
                result = _supabase_request("PATCH", f"?id=eq.{int(row['db_id'])}", values)
            else:
                result = _supabase_request("POST", "", values)
        if not result or not result[0].get("id"):
            raise DatabaseError("Supabase accepted no row for the save request")
        ids.append(int(result[0]["id"]))
    return ids


def _list_records_supabase(
    limit: int, offset: int, search: str, status_filter: str,
    date_from: str, date_to: str, sort_by: str,
) -> tuple[list[dict], int]:
    rows = _supabase_request("GET", "?select=*&order=id.desc&limit=1000")
    if search:
        needle = search.lower()
        fields = ("name", "company_name", "job_title", "address", "telephone",
                  "mobile_phone", "email", "business_type")
        rows = [r for r in rows if any(needle in str(r.get(f) or "").lower() for f in fields)]
    if date_from:
        rows = [r for r in rows if str(r.get("created_at") or "")[:10] >= date_from]
    if date_to:
        rows = [r for r in rows if str(r.get("created_at") or "")[:10] <= date_to]
    if sort_by == "id_asc":
        rows.sort(key=lambda r: int(r.get("id") or 0))
    elif sort_by == "name_asc":
        rows.sort(key=lambda r: str(r.get("name") or "").lower())
    elif sort_by == "name_desc":
        rows.sort(key=lambda r: str(r.get("name") or "").lower(), reverse=True)
    total = len(rows)
    return rows[offset:offset + limit], total


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is not None:
        return _engine
    with _init_lock:
        if _engine is None:
            kwargs: dict = {"pool_pre_ping": True, "future": True}
            if settings.is_sqlite:
                kwargs["connect_args"] = {"check_same_thread": False, "timeout": 15}
            else:
                # prepare_threshold=None keeps psycopg compatible with Supabase's
                # transaction pooler (pgbouncer), which can't use prepared statements.
                kwargs["connect_args"] = {"prepare_threshold": None, "connect_timeout": 10}
                kwargs.update(pool_size=5, max_overflow=5, pool_recycle=1800)
            _engine = create_engine(settings.database_url, **kwargs)
            _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
            log.info("Database engine created for %s", safe_url())
    return _engine


def init_db() -> None:
    """Create the table/indexes if missing and migrate SQLite schema to ensure all fields are nullable."""
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        if settings.is_sqlite:
            with engine.begin() as conn:
                info = conn.execute(text("PRAGMA table_info(business_cards)")).fetchall()
                col_map = {row[1]: row for row in info}
                
                # Check if old table has obsolete NOT NULL constraints on pdf_name, card_number, or ocr_status
                has_strict_not_null = any(
                    col_map.get(col) and col_map[col][3] == 1  # notnull == 1
                    for col in ("pdf_name", "card_number", "ocr_status")
                )
                schema_replaced = False
                
                if has_strict_not_null:
                    log.info("Migrating SQLite schema to relax obsolete NOT NULL constraints...")
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS business_cards_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            company_name TEXT,
                            address TEXT,
                            telephone TEXT,
                            mobile_phone TEXT,
                            email TEXT,
                            job_title TEXT,
                            business_type TEXT,
                            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                    # Copy matching columns if any existing data
                    existing_cols = set(col_map.keys()) & {"id", "name", "company_name", "address", "telephone", "mobile_phone", "email", "job_title", "business_type", "business_confidence", "created_at", "updated_at"}
                    if existing_cols:
                        cols_str = ", ".join(existing_cols)
                        conn.execute(text(f"INSERT OR IGNORE INTO business_cards_new ({cols_str}) SELECT {cols_str} FROM business_cards"))
                    conn.execute(text("DROP TABLE business_cards"))
                    conn.execute(text("ALTER TABLE business_cards_new RENAME TO business_cards"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_business_cards_email ON business_cards (email)"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_business_cards_telephone ON business_cards (telephone)"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_business_cards_mobile_phone ON business_cards (mobile_phone)"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_business_cards_name ON business_cards (name)"))
                    log.info("SQLite schema migrated: all fields are now safely nullable.")
                    schema_replaced = True
                elif "mobile_phone" not in col_map:
                    conn.execute(text("ALTER TABLE business_cards ADD COLUMN mobile_phone TEXT"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_business_cards_mobile_phone ON business_cards (mobile_phone)"))
                    log.info("Migrated SQLite schema: added mobile_phone")

                # Migrate newly added card fields on existing SQLite databases.
                if not schema_replaced and "company_name" not in col_map:
                    conn.execute(text("ALTER TABLE business_cards ADD COLUMN company_name TEXT"))
                    log.info("Migrated SQLite schema: added company_name")
                if not schema_replaced and "business_type" not in col_map:
                    conn.execute(text("ALTER TABLE business_cards ADD COLUMN business_type TEXT"))
                    log.info("Migrated SQLite schema: added business_type")
                if not schema_replaced and "business_confidence" not in col_map:
                    conn.execute(text("ALTER TABLE business_cards ADD COLUMN business_confidence INTEGER"))
        else:
            # create_all does not alter an already-existing Supabase/PostgreSQL table.
            with engine.begin() as conn:
                conn.execute(text(
                    f"ALTER TABLE {settings.db_table} ADD COLUMN IF NOT EXISTS company_name TEXT"
                ))
                conn.execute(text(
                    f"ALTER TABLE {settings.db_table} ADD COLUMN IF NOT EXISTS business_type TEXT"
                ))
                conn.execute(text(
                    f"ALTER TABLE {settings.db_table} ADD COLUMN IF NOT EXISTS business_confidence INTEGER"
                ))

        log.info("Database schema ready (table %s)", settings.db_table)
    except Exception as exc:
        log.error("Database initialisation failed: %s", _describe(exc))
        raise DatabaseError(f"Database initialisation failed: {_describe(exc)}") from exc


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _describe(exc: Exception) -> str:
    msg = str(getattr(exc, "orig", None) or exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return msg[:300]


def health() -> dict:
    if settings.has_supabase_rest:
        try:
            _supabase_request("GET", "?select=id&limit=1")
            return {"ok": True, "backend": "supabase-rest", "url": settings.supabase_url, "table": settings.db_table}
        except DatabaseError as exc:
            return {"ok": False, "backend": "supabase-rest", "url": settings.supabase_url, "table": settings.db_table, "error": str(exc)}
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "backend": backend_name(), "url": safe_url(), "table": settings.db_table}
    except Exception as exc:
        return {"ok": False, "backend": backend_name(), "url": safe_url(), "error": _describe(exc)}


def save_cards(rows: list[dict]) -> list[int]:
    """Insert new rows / update rows that carry a db_id. All-or-nothing transaction.
    Returns the database ids in the same order as `rows`."""
    try:
        if settings.has_supabase_rest:
            return _save_cards_supabase(rows)
        ids: list[int] = []
        with session_scope() as s:
            objs = []
            for r in rows:
                values = {k: r.get(k) for k in SAVE_FIELDS}
                obj = s.get(BusinessCard, r["db_id"]) if r.get("db_id") else None
                if obj is None:
                    obj = BusinessCard(**values)
                    s.add(obj)
                else:
                    for k, v in values.items():
                        setattr(obj, k, v)
                objs.append(obj)
            s.flush()
            ids = [o.id for o in objs]
        return ids
    except Exception as exc:
        log.exception("Database save failed for %d row(s)", len(rows))
        raise DatabaseError(f"Database save failed: {_describe(exc)}") from exc


def dedupe_candidates(emails: set[str], phone_keys: set[str], names: set[str]) -> list[dict]:
    """Fetch existing rows that could duplicate the given keys (cheap, indexed filters
    for email/name; phones are compared in Python since formats differ)."""
    try:
        with session_scope() as s:
            conds = []
            # The email column can hold several "; "-separated addresses, hence LIKE.
            for e in list(emails)[:200]:
                conds.append(func.lower(BusinessCard.email).like(f"%{e}%"))
            if names:
                conds.append(func.lower(BusinessCard.name).in_(list(names)))
            for p in list(phone_keys)[:200]:
                # Coarse filter: the last 4 digits are contiguous in every common format;
                # exact comparison happens in duplicate_service.
                conds.append(BusinessCard.telephone.like(f"%{p[-4:]}%"))
                conds.append(BusinessCard.mobile_phone.like(f"%{p[-4:]}%"))
            if not conds:
                return []
            q = select(BusinessCard.id, BusinessCard.name, BusinessCard.address, BusinessCard.telephone,
                       BusinessCard.email, BusinessCard.mobile_phone).where(or_(*conds)).limit(5000)
            return [dict(r._mapping) for r in s.execute(q)]
    except Exception as exc:
        raise DatabaseError(f"Duplicate lookup failed: {_describe(exc)}") from exc


def list_records(
    limit: int = 200,
    offset: int = 0,
    search: str = "",
    status_filter: str = "",
    date_from: str = "",
    date_to: str = "",
    sort_by: str = "id_desc",
) -> tuple[list[dict], int]:
    if settings.has_supabase_rest:
        try:
            return _list_records_supabase(limit, offset, search, status_filter, date_from, date_to, sort_by)
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Could not read Supabase records: {exc}") from exc
    try:
        with session_scope() as s:
            q = select(BusinessCard)
            cq = select(func.count()).select_from(BusinessCard)
            conds = []

            if search:
                like = f"%{search.lower()}%"
                conds.append(
                    or_(*(func.lower(c).like(like) for c in (
                        BusinessCard.name,
                        BusinessCard.email,
                        BusinessCard.telephone,
                        BusinessCard.mobile_phone,
                        BusinessCard.address,
                        BusinessCard.company_name,
                        BusinessCard.job_title,
                        BusinessCard.business_type,
                    )))
                )

            if date_from:
                conds.append(BusinessCard.created_at >= date_from)
            if date_to:
                conds.append(BusinessCard.created_at <= f"{date_to} 23:59:59")

            if conds:
                q = q.where(*conds)
                cq = cq.where(*conds)

            total = s.execute(cq).scalar_one()

            if sort_by == "id_asc":
                order = BusinessCard.id.asc()
            elif sort_by == "name_asc":
                order = BusinessCard.name.asc()
            elif sort_by == "name_desc":
                order = BusinessCard.name.desc()
            else:
                order = BusinessCard.id.desc()

            rows = s.execute(q.order_by(order).limit(limit).offset(offset)).scalars().all()
            return [r.to_dict() for r in rows], total
    except Exception as exc:
        raise DatabaseError(f"Could not read records: {_describe(exc)}") from exc


def get_record(card_id: int) -> dict | None:
    try:
        with session_scope() as s:
            obj = s.get(BusinessCard, card_id)
            return obj.to_dict() if obj else None
    except Exception as exc:
        raise DatabaseError(f"Could not fetch record #{card_id}: {_describe(exc)}") from exc


def update_record(card_id: int, values: dict) -> dict:
    """Update editable fields of an existing card in the database."""
    try:
        with session_scope() as s:
            obj = s.get(BusinessCard, card_id)
            if obj is None:
                raise DatabaseError(f"Record #{card_id} not found")
            for k in SAVE_FIELDS:
                if k in values:
                    setattr(obj, k, values[k] if values[k] is not None else "")
            s.flush()
            return obj.to_dict()
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Could not update record #{card_id}: {_describe(exc)}") from exc


def delete_record(card_id: int) -> bool:
    """Delete a record by ID."""
    try:
        with session_scope() as s:
            obj = s.get(BusinessCard, card_id)
            if obj is None:
                return False
            s.delete(obj)
            return True
    except Exception as exc:
        raise DatabaseError(f"Could not delete record #{card_id}: {_describe(exc)}") from exc


def bulk_delete_records(card_ids: list[int]) -> int:
    """Delete multiple records by ID."""
    if not card_ids:
        return 0
    try:
        with session_scope() as s:
            stmt = select(BusinessCard).where(BusinessCard.id.in_(card_ids))
            objs = s.execute(stmt).scalars().all()
            count = len(objs)
            for obj in objs:
                s.delete(obj)
            return count
    except Exception as exc:
        raise DatabaseError(f"Could not bulk delete records: {_describe(exc)}") from exc
