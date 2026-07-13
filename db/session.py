# db/session.py
"""Database engine and session factory. Import `get_session` everywhere
a DB connection is needed -- never create engines outside this module."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import structlog
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from db.models import Base

logger = structlog.get_logger(__name__)

_engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # detect stale connections
    pool_size=5,
    max_overflow=10,
)

_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


def create_tables(retries: int = 5, delay: float = 3.0) -> None:
    """Create all tables, retrying if Postgres isn't ready yet."""
    import time

    for attempt in range(1, retries + 1):
        try:
            Base.metadata.create_all(_engine)
            logger.info("db_tables_ready")
            return
        except Exception as exc:
            logger.warning("db_tables_retry", attempt=attempt, error=str(exc))
            if attempt == retries:
                raise
            time.sleep(delay)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional DB session, rolling back on error."""
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
