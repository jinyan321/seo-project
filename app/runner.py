"""Daily batch lifecycle, resume, cost guard and re-extraction.

API calls run in a thread pool; all database writes happen on the calling thread.
"""

import json
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from app import alerts, db
from app.config import Config, get_config
from app.extract import EXTRACTOR_VERSION, extract_citations, extract_mentions
from app.models import Batch, Citation, Mention, Prompt, Run, utcnow
from app.prompts import active_prompts
from app.providers import Provider, ProviderError, build_providers, parse_raw
from app.sentiment import ClaudeLabeler, FakeLabeler, Labeler
from app.settings import Settings, get_settings

log = logging.getLogger("runner")

ADVISORY_LOCK_KEY = 7_204_331_901  # arbitrary, fixed: "the daily batch"
_local_lock = threading.Lock()  # non-Postgres fallback (single process only)


def today(cfg: Config) -> date:
    return datetime.now(ZoneInfo(cfg.schedule.timezone)).date()


@contextmanager
def batch_lock(engine: Engine) -> Iterator[bool]:
    """Postgres advisory lock held for the whole batch, so only one runner works at a time."""
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            got = bool(conn.execute(text("SELECT pg_try_advisory_lock(:k)"),
                                    {"k": ADVISORY_LOCK_KEY}).scalar())
            try:
                yield got
            finally:
                if got:
                    conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": ADVISORY_LOCK_KEY})
                conn.commit()
    else:
        got = _local_lock.acquire(blocking=False)
        try:
            yield got
        finally:
            if got:
                _local_lock.release()


@dataclass
class Task:
    prompt_id: int
    prompt_text: str
    provider: Provider
    sample_idx: int

    @property
    def key(self) -> tuple[int, str, int]:
        return (self.prompt_id, self.provider.name, self.sample_idx)


@dataclass
class Outcome:
    """Everything computed off-thread for one task."""

    raw: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: int = 0
    text: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    mentions: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    label_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class BatchResult:
    batch_id: int
    status: str
    ok: int = 0
    error: int = 0
    skipped: int = 0
    cost_usd: float = 0.0


def _analyse(provider_name: str, raw: dict[str, Any], cfg: Config, out: Outcome,
             labeler: Labeler | None) -> None:
    """parse -> extract -> sentiment. Raises ProviderError if the answer is unusable."""
    parsed = parse_raw(provider_name, raw)
    out.text, out.usage = parsed.text, parsed.usage
    out.mentions = extract_mentions(parsed.text, cfg.brand, cfg.competitors)
    out.citations = extract_citations(parsed.text, parsed.cited, parsed.retrieved)
    if labeler and out.mentions:
        try:
            brands = [m.brand for m in out.mentions]
            out.labels, out.label_usage = labeler.label(parsed.text, brands)
        except Exception as e:  # sentiment is optional; never fail the run for it
            log.warning("sentiment failed: %s", e)


def _execute(task: Task, day: date, cfg: Config, labeler: Labeler | None) -> Outcome:
    out = Outcome()
    t0 = time.monotonic()
    try:
        out.raw = task.provider.ask(task.prompt_text, seed=f"{day.isoformat()}|{task.sample_idx}")
    except Exception as e:  # SDK already retried transient errors
        out.error = f"{type(e).__name__}: {e}"[:2000]
        return out
    finally:
        out.latency_ms = int((time.monotonic() - t0) * 1000)
    try:
        _analyse(task.provider.name, out.raw, cfg, out, labeler)
    except ProviderError as e:
        out.error = f"ProviderError: {e}"
    return out


def _replace_children(s: Session, run: Run, out: Outcome) -> None:
    run.mentions.clear()
    run.citations.clear()
    s.flush()
    for m in out.mentions:
        run.mentions.append(Mention(
            brand=m.brand, is_own=m.is_own, count=m.count, rank_first=m.rank_first,
            rank_list=m.rank_list, snippet=m.snippet, sentiment=out.labels.get(m.brand),
        ))
    for c in out.citations:
        run.citations.append(Citation(url=c.url, domain=c.domain, kind=c.kind, position=c.position))


def _save(s: Session, batch: Batch, run: Run, task: Task, out: Outcome, cfg: Config) -> None:
    run.model, run.raw, run.latency_ms = task.provider.model, out.raw, out.latency_ms
    run.created_at = utcnow()
    if out.error:
        run.status, run.error, run.answer_text = "error", out.error, None
        run.mentions.clear()
        run.citations.clear()
        run.cost_usd = 0.0
    else:
        run.status, run.error, run.answer_text = "ok", None, out.text
        run.input_tokens = out.usage.get("input_tokens", 0)
        run.output_tokens = out.usage.get("output_tokens", 0)
        run.search_calls = out.usage.get("search_calls", 0)
        run.cost_usd = cfg.call_cost(task.provider.name, task.provider.model, out.usage)
        if out.label_usage:
            run.cost_usd += cfg.call_cost("sentiment", cfg.sentiment.model, out.label_usage)
        run.extractor_version = EXTRACTOR_VERSION
        _replace_children(s, run, out)
    batch.cost_usd = round((batch.cost_usd or 0) + run.cost_usd, 6)
    s.commit()
    log.info(json.dumps({
        "event": "run", "batch_id": batch.id, "prompt_id": task.prompt_id,
        "provider": task.provider.name, "sample": task.sample_idx, "status": run.status,
        "latency_ms": out.latency_ms, "cost_usd": run.cost_usd, "error": run.error,
    }))


def _snapshot(cfg: Config, prompts: list[Prompt], providers: list[Provider]) -> dict[str, Any]:
    return {
        "brand": cfg.brand.model_dump(),
        "competitors": [c.model_dump() for c in cfg.competitors],
        "prompts": [{"id": p.id, "slug": p.slug, "version": p.version, "text": p.text}
                    for p in prompts],
        "models": {p.name: p.model for p in providers},
        "samples_per_prompt": cfg.samples_per_prompt,
        "location": cfg.location.model_dump(),
        "extractor_version": EXTRACTOR_VERSION,
    }


def _default_labeler(cfg: Config, settings: Settings) -> Labeler | None:
    if not cfg.sentiment.enabled:
        return None
    if settings.provider_mode == "fake":
        return FakeLabeler()
    if settings.anthropic_api_key:
        return ClaudeLabeler(cfg, settings.anthropic_api_key)
    return None


def run_batch(
    day: date | None = None,
    *,
    providers: list[Provider] | None = None,
    labeler: Labeler | None = None,
    cfg: Config | None = None,
    settings: Settings | None = None,
    use_default_labeler: bool = True,
) -> BatchResult | None:
    """Run (or resume) the batch for `day`. Returns None if another runner holds the lock."""
    cfg = cfg or get_config()
    settings = settings or get_settings()
    day = day or today(cfg)
    if providers is None:
        providers = build_providers(cfg, settings)
    if labeler is None and use_default_labeler:
        labeler = _default_labeler(cfg, settings)

    with batch_lock(db.get_engine()) as got, db.session() as s:
        if not got:
            log.info("another batch runner holds the lock; exiting")
            return None

        batch = s.scalar(select(Batch).where(Batch.scheduled_for == day))
        if batch and batch.status == "complete":
            log.info("batch %s already complete", day)
            return _result(s, batch)
        prompts = active_prompts(s)
        if batch is None:
            batch = Batch(scheduled_for=day, status="running")
            s.add(batch)
        batch.status, batch.finished_at = "running", None
        batch.config_snapshot = _snapshot(cfg, prompts, providers)
        s.commit()

        if not prompts:
            batch.status, batch.finished_at = "failed", utcnow()
            s.commit()
            alerts.alert(f"batch {day}: no active prompts")
            alerts.ping(False)
            return _result(s, batch)

        existing = {(r.prompt_id, r.provider, r.sample_idx): r
                    for r in s.scalars(select(Run).where(Run.batch_id == batch.id))}
        tasks = [
            Task(p.id, p.text, prov, i)
            for p in prompts for prov in providers for i in range(cfg.samples_per_prompt)
            if (r := existing.get((p.id, prov.name, i))) is None or r.status != "ok"
        ]

        def row_for(task: Task) -> Run:
            run = existing.get(task.key)
            if run is None:
                run = Run(batch_id=batch.id, prompt_id=task.prompt_id, provider=task.provider.name,
                          model=task.provider.model, sample_idx=task.sample_idx, status="error")
                s.add(run)
                existing[task.key] = run
            return run

        capped = False
        inflight: dict[Future[Outcome], Task] = {}

        def save_finished() -> None:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                task = inflight.pop(fut)
                _save(s, batch, row_for(task), task, fut.result(), cfg)

        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            for n, task in enumerate(tasks):
                while len(inflight) >= cfg.concurrency:
                    save_finished()
                committed = batch.cost_usd + len(inflight) * cfg.estimate_per_call_usd
                if committed + cfg.estimate_per_call_usd > settings.max_daily_usd:
                    capped = True
                    for skipped in tasks[n:]:
                        run = row_for(skipped)
                        run.status, run.error = "skipped", "daily cost cap reached"
                    s.commit()
                    break
                inflight[pool.submit(_execute, task, day, cfg, labeler)] = task
            while inflight:
                save_finished()

        res = _result(s, batch)
        attempted = res.ok + res.error
        if capped:
            status = "cost_capped"
        elif attempted == 0 or res.error > attempted / 2:
            status = "failed"
        elif res.error:
            status = "partial"
        else:
            status = "complete"
        batch.status, batch.finished_at = status, utcnow()
        s.commit()
        res.status = status

    log.info(json.dumps({"event": "batch", **res.__dict__, "day": day.isoformat()}))
    if status != "complete":
        alerts.alert(f"batch {day} finished {status}: ok={res.ok} error={res.error} "
                     f"skipped={res.skipped} cost=${res.cost_usd:.2f}")
    alerts.ping(status == "complete")
    return res


def _result(s: Session, batch: Batch) -> BatchResult:
    statuses = list(s.scalars(select(Run.status).where(Run.batch_id == batch.id)))
    return BatchResult(
        batch_id=batch.id, status=batch.status, ok=statuses.count("ok"),
        error=statuses.count("error"), skipped=statuses.count("skipped"),
        cost_usd=round(batch.cost_usd or 0.0, 4),
    )


def reextract(*, with_sentiment: bool = False, labeler: Labeler | None = None,
              cfg: Config | None = None) -> dict[str, int]:
    """Rebuild mentions and citations from stored raw responses. No API calls unless
    `with_sentiment`. A run whose raw now parses becomes ok, and vice versa."""
    cfg = cfg or get_config()
    if with_sentiment and labeler is None:
        labeler = _default_labeler(cfg, get_settings())
    counts = {"runs": 0, "ok": 0, "error": 0}
    with db.session() as s:
        run_ids = list(s.scalars(select(Run.id).where(Run.raw.is_not(None)).order_by(Run.id)))
        for run_id in run_ids:
            run = s.get(Run, run_id)
            assert run is not None and run.raw is not None
            old_labels = {m.brand: m.sentiment for m in run.mentions}
            out = Outcome(raw=run.raw)
            try:
                _analyse(run.provider, run.raw, cfg, out, labeler if with_sentiment else None)
            except ProviderError as e:
                run.status, run.error, run.answer_text = "error", f"ProviderError: {e}", None
                run.mentions.clear()
                run.citations.clear()
                counts["error"] += 1
            else:
                if not with_sentiment:
                    out.labels = {k: v for k, v in old_labels.items() if v}
                run.status, run.error, run.answer_text = "ok", None, out.text
                _replace_children(s, run, out)
                counts["ok"] += 1
            run.extractor_version = EXTRACTOR_VERSION
            counts["runs"] += 1
            s.commit()
    return counts
