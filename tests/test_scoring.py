from datetime import date, timedelta

import pytest
from sqlalchemy import select

from app import db
from app.models import Batch, Citation, Mention, Prompt, Run
from app.scoring import Filters, ScoringError, gap_report, mention_rate, wilson


def test_wilson_known_values():
    assert wilson(0, 0) is None
    assert wilson(14, 30) == [0.3023, 0.6386]  # checked by hand: centre 0.4704 ± 0.1681
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.28


def add_run(s, batch, prompt, provider, sample, brands=(), cited=(), status="ok", rank=None):
    r = Run(batch_id=batch.id, prompt_id=prompt.id, provider=provider, model="m",
            sample_idx=sample, status=status, answer_text="x")
    s.add(r)
    s.flush()
    for i, b in enumerate(brands, 1):
        s.add(Mention(run_id=r.id, brand=b, is_own=b == "Mochi HRMS", count=1, rank_first=i,
                      rank_list=rank, snippet=f"{b} snippet"))
    for i, d in enumerate(cited, 1):
        s.add(Citation(run_id=r.id, url=f"https://{d}/x", domain=d, kind="cited", position=i))
    return r


@pytest.fixture
def data(seeded):
    with db.session() as s:
        p = s.scalar(select(Prompt).where(Prompt.slug == "best-sme"))
        d0 = date(2026, 9, 7)  # a Monday
        b1 = Batch(scheduled_for=d0, status="complete")
        b2 = Batch(scheduled_for=d0 + timedelta(days=7), status="complete")
        s.add_all([b1, b2])
        s.flush()
        # week 1: 1 of 3 mention us (and agree=False); competitor cited from g2 without us
        add_run(s, b1, p, "anthropic", 0, ["Mochi HRMS", "Talenox"], ["g2.com"])
        add_run(s, b1, p, "anthropic", 1, ["Talenox"], ["g2.com", "reddit.com"])
        add_run(s, b1, p, "anthropic", 2, ["Talenox"], ["g2.com"])
        add_run(s, b1, p, "openai", 0, status="error")  # excluded
        # week 2: 2 of 2, first both times
        add_run(s, b2, p, "openai", 0, ["Mochi HRMS"], ["talenox.com"], rank=1)
        add_run(s, b2, p, "openai", 1, ["Mochi HRMS", "Swingvy"], [], rank=1)
        s.commit()
    return p


def test_mention_rate_summary_and_series(data, cfg):
    with db.session() as s:
        out = mention_rate(s, cfg, bucket="week")
    sm = out["summary"]
    assert (sm["runs"], sm["mentioned"], sm["mention_rate"]) == (5, 3, 0.6)
    assert [p["period"] for p in out["series"]] == ["2026-09-07", "2026-09-14"]
    assert sm["trend"]["delta"] == round(1.0 - 1 / 3, 4)
    assert sm["top1_rate"] == 0.6  # rank_first 1 in week 1, rank_list 1 twice in week 2
    assert sm["sample_agreement"] == 0.5  # week-1 group disagrees, week-2 group agrees
    assert sm["own_domain_cited_rate"] is None  # config has no own domains
    sov = {b["brand"]: b["share"] for b in sm["share_of_voice"]}
    assert sov["Talenox"] == round(3 / 7, 4) and sov["Mochi HRMS"] == round(3 / 7, 4)
    assert sm["top_cited_domains"][0] == {"domain": "g2.com", "runs": 3}


def test_filters_and_competitor_brand(data, cfg):
    with db.session() as s:
        assert mention_rate(s, cfg, filters=Filters(provider="openai"))["summary"]["runs"] == 2
        assert mention_rate(s, cfg, "talenox")["summary"]["mentioned"] == 3
        assert mention_rate(s, cfg, filters=Filters(prompt_id="best-sme"))["summary"]["runs"] == 5
        with pytest.raises(ScoringError):
            mention_rate(s, cfg, "NotABrand")


def test_gap_report(data, cfg):
    with db.session() as s:
        gaps = {g["domain"]: g for g in gap_report(s, cfg)["gaps"]}
    assert gaps["g2.com"]["gap"] == 2 and gaps["g2.com"]["competitors"] == ["Talenox"]
    assert gaps["reddit.com"]["gap"] == 1
    assert "talenox.com" not in gaps  # a tracked brand's own site is not a PR target
