"""Prompt versioning. A prompt row is never edited in place: an edit stops the old
version and inserts version+1, so trends never mix two different wordings."""

import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Config
from app.models import Prompt, utcnow

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class PromptError(ValueError):
    pass


def active_prompts(s: Session) -> list[Prompt]:
    return list(s.scalars(select(Prompt).where(Prompt.status == "active").order_by(Prompt.slug)))


def add_prompt(s: Session, slug: str, text: str, intent: str | None, user: str | None) -> Prompt:
    slug, text = slug.strip().lower(), text.strip()
    if not _SLUG.match(slug):
        raise PromptError("slug must be lowercase letters, digits and dashes")
    if not text:
        raise PromptError("prompt text is empty")
    if s.scalar(select(func.count()).where(Prompt.slug == slug)):
        raise PromptError(f"slug '{slug}' already exists; edit it instead")
    p = Prompt(slug=slug, version=1, text=text, intent=intent or None, created_by=user)
    s.add(p)
    s.commit()
    return p


def edit_prompt(s: Session, prompt_id: int, text: str, user: str | None) -> Prompt:
    """Stop the current version and start the next one, in one transaction."""
    old = s.get(Prompt, prompt_id)
    if old is None or old.status != "active":
        raise PromptError("only an active prompt can be edited")
    text = text.strip()
    if not text:
        raise PromptError("prompt text is empty")
    if text == old.text:
        return old
    latest = s.scalar(select(func.max(Prompt.version)).where(Prompt.slug == old.slug)) or 0
    old.status, old.stopped_at = "stopped", utcnow()
    new = Prompt(slug=old.slug, version=latest + 1, text=text, intent=old.intent, created_by=user)
    s.add(new)
    s.commit()
    return new


def stop_prompt(s: Session, prompt_id: int) -> Prompt:
    p = s.get(Prompt, prompt_id)
    if p is None:
        raise PromptError("prompt not found")
    if p.status == "active":
        p.status, p.stopped_at = "stopped", utcnow()
        s.commit()
    return p


def seed_prompts(s: Session, cfg: Config) -> int:
    """Load config.yaml seed_prompts, only when the prompts table is empty."""
    if s.scalar(select(func.count()).select_from(Prompt)):
        return 0
    for sp in cfg.seed_prompts:
        s.add(Prompt(slug=sp.slug, version=1, text=sp.text, intent=sp.intent, created_by="seed"))
    s.commit()
    return len(cfg.seed_prompts)
