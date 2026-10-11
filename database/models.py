"""SQLAlchemy model for the business_cards table (PostgreSQL / Supabase / SQLite)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import settings


class Base(DeclarativeBase):
    pass


class BusinessCard(Base):
    __tablename__ = settings.db_table

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(Text)
    company_name: Mapped[str | None] = mapped_column(Text)
    address: Mapped[str | None] = mapped_column(Text)
    telephone: Mapped[str | None] = mapped_column(Text)
    mobile_phone: Mapped[str | None] = mapped_column(Text)
    email: Mapped[str | None] = mapped_column(Text)

    job_title: Mapped[str | None] = mapped_column(Text)
    business_type: Mapped[str | None] = mapped_column(Text)
    business_confidence: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index(f"ix_{settings.db_table}_email", "email"),
        Index(f"ix_{settings.db_table}_telephone", "telephone"),
        Index(f"ix_{settings.db_table}_mobile_phone", "mobile_phone"),
        Index(f"ix_{settings.db_table}_name", "name"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name or "", "address": self.address or "",
            "company_name": self.company_name or "",
            "telephone": self.telephone or "", "mobile_phone": self.mobile_phone or "",
            "email": self.email or "", "job_title": self.job_title or "",
            "business_type": self.business_type or "",
            "business_confidence": self.business_confidence or 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
