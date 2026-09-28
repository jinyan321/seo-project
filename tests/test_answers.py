from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app import db
from app.answers import history, prompt_nav
from app.auth import create_user
from app.models import Prompt, Run
from app.providers.fake import FailingProvider, FakeProvider
from app.render import render_answer
from app.runner import run_batch
from tests.test_web import csrf_from

D1, D2 = date(2026, 9, 1), date(2026, 9, 2)


def batch(cfg, day, providers=None):
    run_batch(day, providers=providers or [FakeProvider(cfg, "anthropic", own_bias=1.0),
                                           FakeProvider(cfg, "openai", own_bias=0.0)],
              cfg=cfg, use_default_labeler=False)


@pytest.fixture
def client(seeded, cfg):
    from app.web.main import create_app

    with db.session() as s:
        create_user(s, "viewer", "viewer-pass-123")
    batch(cfg, D1)
    batch(cfg, D2, [FakeProvider(cfg, "anthropic", own_bias=1.0), FailingProvider(cfg, "openai")])
    with TestClient(create_app()) as c:
        c.post("/login", data={"username": "viewer", "password": "viewer-pass-123",
                               "csrf": csrf_from(c.get("/login").text)})
        yield c


def pid(slug):
    with db.session() as s:
        return s.scalar(select(Prompt.id).where(Prompt.slug == slug))


def test_requires_login(seeded):
    from app.web.main import create_app

    with TestClient(create_app()) as c:
        r = c.get("/answers/1", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"


def test_answer_page_shows_prompt_chips_and_rendered_answer(client):
    r = client.get(f"/answers/{pid('best-sme')}?date={D1}&provider=anthropic")
    assert r.status_code == 200
    html = r.text
    assert "What is the best HR software for SMEs in Malaysia?" in html
    assert 'class="chip own"' in html and "Mochi HRMS" in html  # own_bias=1.0 always names us
    assert "<ol>" in html and "<li>" in html  # the fake answer's numbered list is rendered
    assert "Mentioned #" in html
    assert "Sources cited (2)" in html


def test_provider_sample_and_compare(client):
    base = f"/answers/{pid('best-sme')}?date={D1}"
    assert "Not mentioned" in client.get(base + "&provider=openai").text  # own_bias=0.0
    assert "sample 3" in client.get(base + "&provider=anthropic&sample=2").text
    both = client.get(base + "&compare=1").text
    assert "<h3>ChatGPT search</h3>" in both and "<h3>Claude</h3>" in both


def test_latest_day_default_and_error_run(client):
    r = client.get(f"/answers/{pid('statutory')}?provider=openai")
    assert f'value="{D2}" selected' in r.text  # defaults to the newest day
    assert "This call failed" in r.text and "simulated API failure" in r.text


def test_prev_next_wraps(client):
    with db.session() as s:
        nav = prompt_nav(s)
    first = client.get(f"/answers/{nav[0]}").text
    assert f"/answers/{nav[-1]}?" in first and f"/answers/{nav[1]}?" in first
    assert f"1 / {len(nav)}" in first


def test_unknown_prompt_404_and_index_redirect(client):
    assert client.get("/answers/9999").status_code == 404
    r = client.get("/answers", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/answers/")


def test_history_counts(client):
    with db.session() as s:
        providers, rows = history(s, pid("best-sme"))
    assert providers == ["openai", "anthropic"]
    by_day = {r.day: r.cells for r in rows}
    assert by_day[D1]["anthropic"][:2] == (3, 3) and by_day[D1]["openai"][:2] == (0, 3)
    assert by_day[D2]["openai"][:2] == (0, 0)  # failed runs don't count as ok samples
    assert "named in 3/3" in client.get(f"/answers/{pid('best-sme')}?tab=history").text


def test_untrusted_answer_is_escaped(client):
    evil = ("<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>\n\n"
            "[click](javascript:alert(1)) and [ok](https://example.com)")
    with db.session() as s:
        run = s.scalar(select(Run).where(Run.provider == "anthropic", Run.status == "ok"))
        run.answer_text = evil
        s.commit()
        url = f"/answers/{run.prompt_id}?date={D1}&provider=anthropic&sample={run.sample_idx}"
    html = client.get(url).text
    answer = html.split('<div class="answer">', 1)[1].split("</div>", 1)[0]
    assert "<script" not in answer and "<img" not in answer
    assert 'href="javascript:' not in answer
    assert '<a href="https://example.com" rel="nofollow noopener noreferrer" target="_blank">' \
        in answer


def test_render_answer_tables():
    html = render_answer("| Vendor | Price |\n|---|---|\n| Talenox | $ |")
    assert "<table>" in html and "<td>Talenox</td>" in html


def test_synced_time_is_local():
    from datetime import UTC, datetime

    from app.answers import local_time

    assert local_time(datetime(2026, 9, 26, 17, 0), "Asia/Kuala_Lumpur") == "Sep 27, 2026 01:00"
    assert local_time(datetime(2026, 9, 26, 17, 0, tzinfo=UTC), "Asia/Kuala_Lumpur") \
        == "Sep 27, 2026 01:00"


def test_single_sample_day_hides_sample_tabs(seeded, cfg):
    from app.web.main import create_app

    one = cfg.model_copy(update={"samples_per_prompt": 1})
    run_batch(D1, providers=[FakeProvider(one, "anthropic"), FakeProvider(one, "openai")],
              cfg=one, use_default_labeler=False)
    with db.session() as s:
        create_user(s, "solo", "solo-pass-123")
    with TestClient(create_app()) as c:
        c.post("/login", data={"username": "solo", "password": "solo-pass-123",
                               "csrf": csrf_from(c.get("/login").text)})
        html = c.get(f"/answers/{pid('best-sme')}").text
    assert "<h4>Sample</h4>" not in html and "Brands mentioned" in html
