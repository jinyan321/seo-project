"""Read-only queries for the AI Answers page. Never calls an LLM."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import Config
from app.models import Batch, Citation, Mention, Prompt, Run

PROVIDER_LABELS = {"openai": "ChatGPT search", "anthropic": "Claude"}
PROVIDER_ORDER = ["openai", "anthropic"]


def provider_label(name: str) -> str:
    return PROVIDER_LABELS.get(name, name)


def local_time(dt: datetime, tz: str) -> str:
    """Stored times are UTC (SQLite drops the tzinfo); show them in the schedule timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ZoneInfo(tz)).strftime("%b %d, %Y %H:%M")


def _rank(m: Mention) -> int:
    return m.rank_list if m.rank_list is not None else m.rank_first


def _provider_sort(names) -> list[str]:
    return sorted(names, key=lambda p: (PROVIDER_ORDER.index(p) if p in PROVIDER_ORDER else 99, p))


def prompt_nav(s: Session) -> list[int]:
    """Prompts that have at least one run: active first, then stopped; by slug, version."""
    with_runs = select(Run.prompt_id).distinct()
    rows = s.execute(
        select(Prompt.id, Prompt.status, Prompt.slug, Prompt.version)
        .where(Prompt.id.in_(with_runs))
    ).all()
    rows.sort(key=lambda r: (r.status != "active", r.slug, r.version))
    return [r.id for r in rows]


@dataclass
class ShownRun:
    run: Run
    provider: str
    label: str
    mentions: list[Mention]
    own: Mention | None
    cited: list[Citation]
    retrieved: list[Citation]


@dataclass
class SidebarItem:
    provider: str
    label: str
    named: int  # samples that named the own brand
    ok: int  # ok samples
    best_rank: int | None
    samples: dict[int, bool | None] = field(default_factory=dict)  # idx -> named? (None=error)


@dataclass
class AnswerPage:
    prompt: Prompt
    position: int
    total: int
    prev_id: int
    next_id: int
    days: list[date]
    day: date
    sidebar: list[SidebarItem]
    provider: str
    sample: int
    samples: list[int]
    shown: list[ShownRun]


def _shown(run: Run) -> ShownRun:
    mentions = sorted(run.mentions, key=_rank)
    cits = sorted(run.citations, key=lambda c: c.position)
    cited = [c for c in cits if c.kind == "cited"]
    cited_urls = {c.url for c in cited}
    return ShownRun(
        run=run, provider=run.provider, label=provider_label(run.provider), mentions=mentions,
        own=next((m for m in mentions if m.is_own), None), cited=cited,
        retrieved=[c for c in cits if c.kind == "retrieved" and c.url not in cited_urls],
    )


def answer_page(s: Session, cfg: Config, prompt_id: int, day: date | None = None,
                provider: str | None = None, sample: int = 0,
                compare: bool = False) -> AnswerPage | None:
    prompt = s.get(Prompt, prompt_id)
    nav = prompt_nav(s)
    if prompt is None or prompt_id not in nav:
        return None
    pos = nav.index(prompt_id)

    days = list(s.scalars(
        select(Batch.scheduled_for).join(Run, Run.batch_id == Batch.id)
        .where(Run.prompt_id == prompt_id).distinct().order_by(Batch.scheduled_for.desc())
    ))
    if day not in days:
        day = days[0]

    runs = list(s.scalars(
        select(Run).join(Batch, Run.batch_id == Batch.id)
        .where(Run.prompt_id == prompt_id, Batch.scheduled_for == day)
        .options(selectinload(Run.mentions), selectinload(Run.citations))
        .order_by(Run.provider, Run.sample_idx)
    ))
    by_provider: dict[str, dict[int, Run]] = defaultdict(dict)
    for r in runs:
        by_provider[r.provider][r.sample_idx] = r
    providers = _provider_sort(by_provider)

    sidebar = []
    for p in providers:
        item = SidebarItem(provider=p, label=provider_label(p), named=0, ok=0, best_rank=None)
        for idx, r in sorted(by_provider[p].items()):
            own = next((m for m in r.mentions if m.is_own), None) if r.status == "ok" else None
            item.samples[idx] = None if r.status != "ok" else own is not None
            if r.status == "ok":
                item.ok += 1
            if own:
                item.named += 1
                rank = _rank(own)
                item.best_rank = rank if item.best_rank is None else min(item.best_rank, rank)
        sidebar.append(item)

    if provider not in by_provider:
        provider = providers[0]
    samples = sorted(by_provider[provider])
    if sample not in by_provider[provider]:
        sample = samples[0]

    shown_providers = providers if compare else [provider]
    shown = [
        _shown(by_provider[p].get(sample) or next(iter(by_provider[p].values())))
        for p in shown_providers
    ]
    return AnswerPage(
        prompt=prompt, position=pos + 1, total=len(nav),
        prev_id=nav[pos - 1], next_id=nav[(pos + 1) % len(nav)],
        days=days, day=day, sidebar=sidebar, provider=provider, sample=sample,
        samples=samples, shown=shown,
    )


@dataclass
class HistoryRow:
    day: date
    cells: dict[str, tuple[int, int, int | None]]  # provider -> (named, ok, best rank)


def history(s: Session, prompt_id: int, limit: int = 30) -> tuple[list[str], list[HistoryRow]]:
    days = list(s.scalars(
        select(Batch.scheduled_for).join(Run, Run.batch_id == Batch.id)
        .where(Run.prompt_id == prompt_id).distinct()
        .order_by(Batch.scheduled_for.desc()).limit(limit)
    ))
    if not days:
        return [], []
    rows = s.execute(
        select(Batch.scheduled_for, Run.id, Run.provider, Run.status)
        .join(Batch, Run.batch_id == Batch.id)
        .where(Run.prompt_id == prompt_id, Batch.scheduled_for.in_(days))
    ).all()
    own_rank = {
        m.run_id: _rank(m)
        for m in s.scalars(
            select(Mention).join(Run, Mention.run_id == Run.id)
            .where(Run.prompt_id == prompt_id, Mention.is_own.is_(True))
        )
    }
    cells: dict[date, dict[str, list]] = defaultdict(lambda: defaultdict(lambda: [0, 0, None]))
    for d, run_id, provider, status in rows:
        c = cells[d][provider]
        if status != "ok":
            continue
        c[1] += 1
        if run_id in own_rank:
            c[0] += 1
            c[2] = own_rank[run_id] if c[2] is None else min(c[2], own_rank[run_id])
    providers = _provider_sort({p for d in cells for p in cells[d]})
    return providers, [
        HistoryRow(day=d, cells={p: tuple(cells[d][p]) for p in providers if p in cells[d]})
        for d in days
    ]


def brand_terms(cfg: Config) -> list[dict]:
    """Names to highlight in the browser: own brand first."""
    return [{"name": n, "own": b is cfg.brand, "cs": b.case_sensitive}
            for b in cfg.all_brands for n in b.names]
