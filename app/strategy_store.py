"""Strategy rows as the web app sees them: queue a request, read results.

Deliberately imports no LLM client. The worker (app/strategy.py) does the generating.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Strategy

OPEN = ("pending", "running")


def open_request(s: Session, prompt_id: int) -> Strategy | None:
    return s.scalar(select(Strategy).where(Strategy.prompt_id == prompt_id,
                                           Strategy.status.in_(OPEN)).limit(1))


def request_strategy(s: Session, prompt_id: int, user: str | None) -> tuple[Strategy, bool]:
    """Queue one strategy for the worker. Returns (row, created). A second request while one
    is still pending or running returns the existing row instead of queueing a duplicate."""
    existing = open_request(s, prompt_id)
    if existing:
        return existing, False
    row = Strategy(prompt_id=prompt_id, status="pending", requested_by=user)
    s.add(row)
    s.commit()
    return row, True


def recent(s: Session, prompt_id: int, limit: int = 6) -> list[Strategy]:
    return list(s.scalars(select(Strategy).where(Strategy.prompt_id == prompt_id)
                          .order_by(Strategy.id.desc()).limit(limit)))
