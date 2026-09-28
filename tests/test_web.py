import re
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app import db
from app.auth import create_user
from app.models import Prompt
from app.providers.fake import FakeProvider
from app.runner import run_batch


@pytest.fixture
def client(seeded, cfg):
    from app.web.main import create_app

    with db.session() as s:
        create_user(s, "admin", "admin-pass-123", is_admin=True)
        create_user(s, "viewer", "viewer-pass-123")
    run_batch(date(2026, 9, 1), providers=[FakeProvider(cfg, "anthropic"),
                                           FakeProvider(cfg, "openai")],
              cfg=cfg, use_default_labeler=False)
    with TestClient(create_app()) as c:
        yield c


def csrf_from(html):
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


def login(c, user, pw):
    r = c.post("/login", data={"username": user, "password": pw,
                               "csrf": csrf_from(c.get("/login").text)},
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/dashboard"
    return csrf_from(c.get("/admin/prompts").text)  # the session (and token) rotate on login


def test_api_requires_basic_auth(client):
    assert client.get("/mention-rate").status_code == 401
    assert client.get("/mention-rate", auth=("admin", "wrong")).status_code == 401
    assert client.get("/gap-report").status_code == 401


def test_mention_rate_contract(client):
    r = client.get("/mention-rate?bucket=week", auth=("viewer", "viewer-pass-123"))
    assert r.status_code == 200
    body = r.json()
    assert body["brand"] == "Mochi HRMS"
    assert set(body["series"][0]) == {"period", "provider", "runs", "mentioned", "mention_rate",
                                      "ci95", "avg_rank", "top1_rate"}
    assert {"mention_rate", "ci95", "avg_rank_when_mentioned", "top1_rate",
            "own_domain_cited_rate", "sample_agreement", "trend", "by_provider", "by_prompt",
            "share_of_voice", "top_cited_domains", "latest_snippets"} <= set(body["summary"])
    assert body["summary"]["runs"] == 30


def test_bad_params_are_400(client):
    auth = ("viewer", "viewer-pass-123")
    assert client.get("/mention-rate?brand=Nope", auth=auth).status_code == 400
    assert client.get("/mention-rate?bucket=year", auth=auth).status_code == 422


def test_healthz_is_public(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["latest_batch"]["status"] == "complete"


def test_pages_need_login(client):
    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"


def test_dashboard_renders(client):
    login(client, "viewer", "viewer-pass-123")
    r = client.get("/dashboard")
    assert r.status_code == 200 and "Mention rate over time" in r.text


def test_prompt_edit_through_admin_page(client):
    token = login(client, "viewer", "viewer-pass-123")
    with db.session() as s:
        pid = s.scalar(select(Prompt.id).where(Prompt.slug == "statutory"))
    r = client.post(f"/admin/prompts/{pid}/edit", data={"text": "New wording?", "csrf": token},
                    follow_redirects=False)
    assert r.status_code == 303 and "version%202" in r.headers["location"]
    with db.session() as s:
        rows = list(s.scalars(select(Prompt).where(Prompt.slug == "statutory")
                              .order_by(Prompt.version)))
    assert [(p.version, p.status) for p in rows] == [(1, "stopped"), (2, "active")]


def test_csrf_required(client):
    login(client, "viewer", "viewer-pass-123")
    r = client.post("/admin/prompts", data={"slug": "x", "text": "y", "csrf": "forged"})
    assert r.status_code == 403


def test_users_page_admin_only(client):
    login(client, "viewer", "viewer-pass-123")
    assert client.get("/admin/users").status_code == 403
    client.cookies.clear()
    login(client, "admin", "admin-pass-123")
    assert client.get("/admin/users").status_code == 200
