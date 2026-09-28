"""JSON API. Reads stored results only; never calls an LLM."""

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth import api_user
from app.config import get_config
from app.db import get_session
from app.models import User
from app.scoring import Filters, ScoringError, gap_report, latest_batch, mention_rate

router = APIRouter()


@router.get("/healthz")
def healthz(s: Session = Depends(get_session)) -> dict:
    s.execute(text("SELECT 1"))
    return {"ok": True, "latest_batch": latest_batch(s)}


@router.get("/mention-rate")
def get_mention_rate(
    brand: str | None = None,
    provider: str | None = None,
    prompt_id: str | None = None,
    bucket: Literal["day", "week", "batch"] = "day",
    since: date | None = None,
    until: date | None = None,
    s: Session = Depends(get_session),
    _user: User = Depends(api_user),
) -> dict:
    try:
        return mention_rate(s, get_config(), brand, bucket,
                            Filters(provider, prompt_id, since, until))
    except ScoringError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/gap-report")
def get_gap_report(
    provider: str | None = None,
    prompt_id: str | None = None,
    since: date | None = None,
    until: date | None = None,
    limit: int = 20,
    s: Session = Depends(get_session),
    _user: User = Depends(api_user),
) -> dict:
    try:
        return gap_report(s, get_config(), Filters(provider, prompt_id, since, until), limit)
    except ScoringError as e:
        raise HTTPException(400, str(e)) from e
