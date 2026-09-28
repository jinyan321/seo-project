"""Read-only metrics over stored runs. The denominator is always runs with status='ok'."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.config import Brand, Config
from app.models import Batch, Citation, Mention, Prompt, Run

Z95 = 1.959964


class ScoringError(ValueError):
    pass


@dataclass(frozen=True)
class Filters:
    provider: str | None = None
    prompt_id: str | None = None  # numeric row id, or a slug (= newest version of that slug)
    since: date | None = None
    until: date | None = None


@dataclass
class RunRow:
    id: int
    batch_id: int
    day: date
    provider: str
    prompt_id: int
    slug: str
    version: int
    text: str
    prompt_status: str


def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 2) if xs else None


def _period(day: date, bucket: str, batch_id: int) -> str:
    if bucket == "week":
        return (day - timedelta(days=day.weekday())).isoformat()  # Monday of ISO week
    if bucket == "batch":
        return f"{day.isoformat()}#{batch_id}"
    return day.isoformat()


def _resolve_prompt(s: Session, prompt_id: str) -> int:
    if prompt_id.isdigit():
        return int(prompt_id)
    pid = s.scalar(select(Prompt.id).where(Prompt.slug == prompt_id)
                   .order_by(Prompt.version.desc()).limit(1))
    if pid is None:
        raise ScoringError(f"unknown prompt '{prompt_id}'")
    return pid


def _filtered(stmt: Select, s: Session, f: Filters) -> Select:
    stmt = stmt.join(Batch, Run.batch_id == Batch.id).where(Run.status == "ok")
    if f.provider:
        stmt = stmt.where(Run.provider == f.provider)
    if f.prompt_id:
        stmt = stmt.where(Run.prompt_id == _resolve_prompt(s, f.prompt_id))
    if f.since:
        stmt = stmt.where(Batch.scheduled_for >= f.since)
    if f.until:
        stmt = stmt.where(Batch.scheduled_for <= f.until)
    return stmt


def _load(s: Session, f: Filters):
    rows = s.execute(_filtered(
        select(Run.id, Run.batch_id, Batch.scheduled_for, Run.provider, Run.prompt_id,
               Prompt.slug, Prompt.version, Prompt.text, Prompt.status)
        .select_from(Run).join(Prompt, Run.prompt_id == Prompt.id), s, f,
    ).order_by(Batch.scheduled_for, Run.id)).all()
    runs = [RunRow(*r) for r in rows]

    mentions: dict[int, dict[str, Mention]] = defaultdict(dict)
    for m in s.scalars(_filtered(select(Mention).join(Run, Mention.run_id == Run.id), s, f)):
        mentions[m.run_id][m.brand] = m

    cited: dict[int, set[str]] = defaultdict(set)
    for run_id, domain in s.execute(_filtered(
        select(Citation.run_id, Citation.domain).join(Run, Citation.run_id == Run.id)
        .where(Citation.kind == "cited"), s, f,
    )):
        cited[run_id].add(domain)
    return runs, mentions, cited


def _rank(m: Mention) -> int:
    return m.rank_list if m.rank_list is not None else m.rank_first


def _domain_match(domain: str, owned: list[str]) -> bool:
    return any(domain == d.lower() or domain.endswith("." + d.lower()) for d in owned)


def _stats(runs: list[RunRow], mentions: dict[int, dict[str, Mention]], brand: str) -> dict:
    hits = [mentions[r.id][brand] for r in runs if brand in mentions.get(r.id, {})]
    n, k = len(runs), len(hits)
    return {
        "runs": n,
        "mentioned": k,
        "mention_rate": _rate(k, n),
        "ci95": wilson(k, n),
        "avg_rank": _mean([_rank(m) for m in hits]),
        "top1_rate": _rate(sum(1 for m in hits if _rank(m) == 1), n),
    }


def mention_rate(
    s: Session, cfg: Config, brand_name: str | None = None, bucket: str = "day",
    filters: Filters = Filters(),
) -> dict[str, Any]:
    if bucket not in ("day", "week", "batch"):
        raise ScoringError("bucket must be day, week or batch")
    brand = cfg.find_brand(brand_name) if brand_name else cfg.brand
    if brand is None:
        raise ScoringError(f"'{brand_name}' is not a tracked brand")
    runs, mentions, cited = _load(s, filters)
    name = brand.name

    by_period: dict[str, list[RunRow]] = defaultdict(list)
    for r in runs:
        by_period[_period(r.day, bucket, r.batch_id)].append(r)
    series = [
        {"period": p, "provider": filters.provider or "all", **_stats(rs, mentions, name)}
        for p, rs in by_period.items()
    ]

    overall = _stats(runs, mentions, name)
    trend = None
    if len(series) >= 2 and series[-1]["mention_rate"] is not None:
        a, b = series[-2], series[-1]
        trend = {"from": a["period"], "to": b["period"],
                 "delta": round(b["mention_rate"] - a["mention_rate"], 4)}

    by_provider: dict[str, list[RunRow]] = defaultdict(list)
    by_prompt: dict[int, list[RunRow]] = defaultdict(list)
    for r in runs:
        by_provider[r.provider].append(r)
        by_prompt[r.prompt_id].append(r)

    summary = {
        "runs": overall["runs"],
        "mentioned": overall["mentioned"],
        "mention_rate": overall["mention_rate"],
        "ci95": overall["ci95"],
        "avg_rank_when_mentioned": overall["avg_rank"],
        "top1_rate": overall["top1_rate"],
        "own_domain_cited_rate": _own_domain_rate(runs, cited, brand),
        "sample_agreement": _agreement(runs, mentions, name),
        "trend": trend,
        "by_provider": {p: _stats(rs, mentions, name) for p, rs in sorted(by_provider.items())},
        "by_prompt": sorted(
            (
                {"prompt_id": pid, "slug": rs[0].slug, "version": rs[0].version,
                 "status": rs[0].prompt_status, "text": rs[0].text,
                 **_stats(rs, mentions, name)}
                for pid, rs in by_prompt.items()
            ),
            key=lambda d: (d["slug"], d["version"]),
        ),
        "share_of_voice": _share_of_voice(runs, mentions, cfg),
        "top_cited_domains": [
            {"domain": d, "runs": n}
            for d, n in Counter(d for r in runs for d in cited.get(r.id, ())).most_common(10)
        ],
        "sentiment": dict(Counter(
            mentions[r.id][name].sentiment or "unlabeled"
            for r in runs if name in mentions.get(r.id, {})
        )),
        "latest_snippets": [
            {"date": r.day.isoformat(), "provider": r.provider, "prompt": r.slug,
             "rank": _rank(mentions[r.id][name]), "sentiment": mentions[r.id][name].sentiment,
             "snippet": mentions[r.id][name].snippet}
            for r in reversed(runs) if name in mentions.get(r.id, {})
        ][:5],
    }
    return {
        "brand": name,
        "bucket": bucket,
        "filters": {"provider": filters.provider, "prompt_id": filters.prompt_id,
                    "since": filters.since and filters.since.isoformat(),
                    "until": filters.until and filters.until.isoformat()},
        "series": series,
        "summary": summary,
    }


def _own_domain_rate(runs: list[RunRow], cited: dict[int, set[str]], brand: Brand):
    if not brand.domains:
        return None
    hits = sum(1 for r in runs if any(_domain_match(d, brand.domains) for d in cited.get(r.id, ())))
    return _rate(hits, len(runs))


def _agreement(runs: list[RunRow], mentions: dict[int, dict[str, Mention]], name: str):
    """Share of (batch, prompt, provider) groups whose samples all agree on mentioned-or-not."""
    groups: dict[tuple, list[bool]] = defaultdict(list)
    for r in runs:
        groups[(r.batch_id, r.prompt_id, r.provider)].append(name in mentions.get(r.id, {}))
    multi = [g for g in groups.values() if len(g) >= 2]
    return _rate(sum(1 for g in multi if len(set(g)) == 1), len(multi))


def _share_of_voice(runs: list[RunRow], mentions: dict[int, dict[str, Mention]], cfg: Config):
    counts = {b.name: sum(1 for r in runs if b.name in mentions.get(r.id, {}))
              for b in cfg.all_brands}
    total = sum(counts.values())
    return sorted(
        ({"brand": b.name, "is_own": b is cfg.brand, "share": _rate(counts[b.name], total),
          "mention_rate": _rate(counts[b.name], len(runs))} for b in cfg.all_brands),
        key=lambda d: -(d["share"] or 0),
    )


def gap_report(s: Session, cfg: Config, filters: Filters = Filters(), limit: int = 20) -> dict:
    """Domains the AIs cite when they name competitors but not when they name us."""
    runs, mentions, cited = _load(s, filters)
    own = cfg.brand.name
    competitor_names = {c.name for c in cfg.competitors}
    brand_domains = [d for b in cfg.all_brands for d in b.domains]

    stats: dict[str, dict[str, Any]] = {}
    for r in runs:
        named = set(mentions.get(r.id, {}))
        comps = named & competitor_names
        for domain in cited.get(r.id, ()):
            if _domain_match(domain, brand_domains):
                continue
            st = stats.setdefault(domain, {"runs_with_competitor": 0, "runs_with_own": 0,
                                           "competitors": Counter()})
            if comps:
                st["runs_with_competitor"] += 1
                st["competitors"].update(comps)
            if own in named:
                st["runs_with_own"] += 1

    rows = [
        {"domain": d, "gap": st["runs_with_competitor"] - st["runs_with_own"],
         "runs_with_competitor": st["runs_with_competitor"], "runs_with_own": st["runs_with_own"],
         "competitors": [c for c, _ in st["competitors"].most_common()]}
        for d, st in stats.items()
    ]
    rows = [r for r in rows if r["gap"] > 0]
    rows.sort(key=lambda r: (-r["gap"], -r["runs_with_competitor"], r["domain"]))
    return {"brand": own, "runs": len(runs), "gaps": rows[:limit]}


def latest_batch(s: Session) -> dict[str, Any] | None:
    b = s.scalar(select(Batch).order_by(Batch.scheduled_for.desc()).limit(1))
    if b is None:
        return None
    n = s.scalar(select(func.count()).select_from(Run).where(Run.batch_id == b.id))
    return {"date": b.scheduled_for.isoformat(), "status": b.status, "runs": n,
            "cost_usd": round(b.cost_usd or 0, 4)}
