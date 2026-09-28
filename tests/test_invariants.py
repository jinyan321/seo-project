"""One test per CLAUDE.md invariant. A change that breaks one of these is not done."""

import subprocess
import sys
import threading
import time
from datetime import date

from sqlalchemy import func, select

from app import db, extract
from app.models import Batch, Mention, Prompt, Run
from app.prompts import add_prompt, edit_prompt
from app.providers.fake import FailingProvider, FakeProvider
from app.runner import reextract, run_batch
from app.scoring import mention_rate
from app.settings import Settings
from tests.conftest import ROOT

DAY = date(2026, 9, 1)


def fakes(cfg):
    return [FakeProvider(cfg, "anthropic", own_bias=0.5), FakeProvider(cfg, "openai", own_bias=0.5)]


def run(cfg, day=DAY, providers=None, **kw):
    return run_batch(day, providers=providers or fakes(cfg), cfg=cfg,
                     use_default_labeler=False, **kw)


def count(model, *where):
    with db.session() as s:
        return s.scalar(select(func.count()).select_from(model).where(*where))


# 1. The web app never calls an LLM.
def test_web_app_does_not_import_llm_code():
    code = ("import sys, app.web.main; bad = [m for m in ('anthropic', 'openai', 'app.runner', "
            "'app.providers', 'app.sentiment') if m in sys.modules]; print(bad); "
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# 2. One batch per day, and only one runner at a time.
def test_one_batch_per_day_and_concurrent_runner_does_nothing(seeded, cfg):
    gate, results = threading.Event(), []

    class SlowFake(FakeProvider):
        def ask(self, prompt, seed=""):
            gate.wait(5)
            return super().ask(prompt, seed)

    t = threading.Thread(target=lambda: results.append(
        run(cfg, providers=[SlowFake(cfg, "anthropic"), SlowFake(cfg, "openai")])))
    t.start()
    deadline = time.monotonic() + 5
    while count(Batch) == 0:  # wait until the first runner holds the lock and made the batch
        assert time.monotonic() < deadline, "first runner never started"
        time.sleep(0.01)
    assert run(cfg) is None  # second runner backs off
    gate.set()
    t.join()
    assert results[0].status == "complete"
    assert count(Batch) == 1
    assert run(cfg).status == "complete"  # rerun of a complete day is a no-op
    assert count(Run) == 30


# 3. Batches are resumable: a rerun only fills the gaps.
def test_resume_only_fills_missing_tasks(seeded, cfg):
    res = run(cfg, providers=[FakeProvider(cfg, "anthropic"), FailingProvider(cfg, "openai")])
    assert (res.status, res.ok, res.error) == ("partial", 15, 15)
    with db.session() as s:
        ok_before = {r.id: r.created_at for r in s.scalars(select(Run).where(Run.status == "ok"))}
    res = run(cfg)
    assert (res.status, res.ok, res.error) == ("complete", 30, 0)
    assert count(Run) == 30
    with db.session() as s:
        for rid, created in ok_before.items():
            assert s.get(Run, rid).created_at == created  # untouched


# 4. Failed runs are excluded from scoring, never counted as "not mentioned".
def test_failed_runs_excluded_from_denominator(seeded, cfg):
    run(cfg, providers=[FakeProvider(cfg, "anthropic"), FailingProvider(cfg, "openai")])
    with db.session() as s:
        data = mention_rate(s, cfg)
    assert data["summary"]["runs"] == 15
    assert set(data["summary"]["by_provider"]) == {"anthropic"}


# 5. The raw response is stored before parsing, even when parsing fails.
def test_raw_kept_when_parse_fails(seeded, cfg):
    class Empty(FakeProvider):
        def ask(self, prompt, seed=""):
            return {"fake": True, "text": ""}

    run(cfg, providers=[Empty(cfg, "anthropic")])
    with db.session() as s:
        r = s.scalars(select(Run)).first()
    assert r.status == "error" and "ProviderError" in r.error
    assert r.raw == {"fake": True, "text": ""}


# 6. Reextract rebuilds from raw with no API calls.
def test_reextract_uses_stored_raw_only(seeded, cfg, monkeypatch):
    run(cfg)
    before = count(Mention)

    def boom(*a, **k):
        raise AssertionError("API called")

    monkeypatch.setattr(FakeProvider, "ask", boom)
    with db.session() as s:
        s.query(Mention).delete()
        s.commit()
    assert reextract(cfg=cfg)["ok"] == 30
    assert count(Mention) == before


# 7. Prompts are never edited in place; scoring groups by the versioned prompt.
def test_prompt_edit_creates_new_version(seeded, cfg):
    with db.session() as s:
        p = s.scalar(select(Prompt).where(Prompt.slug == "best-sme"))
        old_id, old_text = p.id, p.text
        new = edit_prompt(s, p.id, "Which HR software do Malaysian SMEs trust most?", "tester")
        old = s.get(Prompt, old_id)
        assert (old.status, old.text, old.stopped_at is not None) == ("stopped", old_text, True)
        assert (new.slug, new.version, new.status) == ("best-sme", 2, "active")
        add_prompt(s, "new-one", "Is Mochi HRMS good?", None, "tester")
    run(cfg)
    with db.session() as s:
        used = set(s.scalars(select(Run.prompt_id).distinct()))
        by_prompt = mention_rate(s, cfg)["summary"]["by_prompt"]
    assert old_id not in used and new.id in used
    assert {(b["slug"], b["version"]) for b in by_prompt} >= {("best-sme", 2), ("new-one", 1)}


# 8. Cost cap: remaining tasks are skipped and the batch is cost_capped.
def test_cost_cap_stops_batch(seeded, cfg):
    settings = Settings(max_daily_usd=0.5, provider_mode="fake")  # ~5 calls at $0.08 estimate
    res = run(cfg, settings=settings)
    assert res.status == "cost_capped"
    assert res.skipped > 0 and res.ok + res.skipped == 30
    with db.session() as s:
        assert s.scalar(select(Batch)).cost_usd <= 0.5


# 9. Brand matching: "HRMS" alone never counts.
def test_hrms_alone_never_counts(cfg):
    assert extract.extract_mentions("The best HRMS in Malaysia is any HRMS.",
                                    cfg.brand, cfg.competitors) == []


# 10. Every schema change has a migration: models and migrations agree.
def test_migrations_match_models(tmp_path):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config as AlembicConfig
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine

    from app.models import Base

    url = f"sqlite:///{tmp_path / 'not-yet-created' / 'mig.db'}"  # fresh checkout: no data/ dir
    acfg = AlembicConfig(str(ROOT / "alembic.ini"))
    acfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(acfg, "head")
    eng = create_engine(url)
    with eng.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], diff
    command.downgrade(acfg, "base")
    eng.dispose()


# 6b. Reextract skips runs that never got a response (failed calls, cost-cap skips).
def test_reextract_ignores_runs_without_raw(seeded, cfg):
    run(cfg, providers=[FakeProvider(cfg, "anthropic"), FailingProvider(cfg, "openai")])
    with db.session() as s:
        assert s.scalar(select(func.count()).select_from(Run).where(Run.raw.is_(None))) == 15
    assert reextract(cfg=cfg) == {"runs": 15, "ok": 15, "error": 0}
