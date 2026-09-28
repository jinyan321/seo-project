"""HTML pages: login, dashboard, prompt admin (with versioning) and user admin."""

from datetime import date
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.answers import (
    answer_page,
    brand_terms,
    history,
    local_time,
    prompt_nav,
    provider_label,
)
from app.auth import (
    admin_user,
    authenticate,
    check_csrf,
    create_user,
    csrf_token,
    page_user,
)
from app.config import get_config
from app.db import get_session
from app.models import Prompt, User
from app.prompts import PromptError, add_prompt, edit_prompt, stop_prompt
from app.render import render_answer
from app.scoring import Filters, ScoringError, gap_report, latest_batch, mention_rate
from app.strategy_store import open_request, recent, request_strategy

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _render(request: Request, name: str, user: User | None = None, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, name, {
        "user": user, "csrf": csrf_token(request),
        "msg": request.query_params.get("msg"), "err": request.query_params.get("err"), **ctx,
    })


def _back(path: str, *, msg: str | None = None, err: str | None = None) -> RedirectResponse:
    q = f"?msg={quote(msg)}" if msg else f"?err={quote(err)}" if err else ""
    return RedirectResponse(path + q, status_code=303)


@router.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse("/dashboard", status_code=303)


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_form(request: Request):
    return _render(request, "login.html")


@router.post("/login", include_in_schema=False)
def login(request: Request, username: str = Form(...), password: str = Form(...),
          csrf: str = Form(""), s: Session = Depends(get_session)):
    check_csrf(request, csrf)
    user = authenticate(s, username, password)
    if user is None:
        return _back("/login", err="Wrong username or password")
    request.session.clear()
    request.session["uid"] = user.id
    return RedirectResponse("/dashboard", status_code=303)


@router.post("/logout", include_in_schema=False)
def logout(request: Request, csrf: str = Form("")):
    check_csrf(request, csrf)
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard(request: Request, brand: str | None = None, bucket: str = "week",
              user: User = Depends(page_user), s: Session = Depends(get_session)):
    cfg = get_config()
    try:
        data = mention_rate(s, cfg, brand, bucket if bucket in ("day", "week") else "week")
    except ScoringError as e:
        return _back("/dashboard", err=str(e))
    return _render(request, "dashboard.html", user, data=data, bucket=bucket,
                   brands=[b.name for b in cfg.all_brands], gaps=gap_report(s, cfg, Filters(), 10),
                   latest=latest_batch(s))


# ---------- AI answers ----------


@router.get("/answers", include_in_schema=False)
def answers_index(user: User = Depends(page_user), s: Session = Depends(get_session)):
    nav = prompt_nav(s)
    if not nav:
        return _back("/dashboard", err="No answers yet. They appear after the first daily run.")
    return RedirectResponse(f"/answers/{nav[0]}", status_code=303)


@router.get("/answers/{prompt_id}", response_class=HTMLResponse, include_in_schema=False)
def answers_view(request: Request, prompt_id: int, date: date | None = None,
                 provider: str | None = None, sample: int = 0, compare: bool = False,
                 tab: str = "answer", sid: int | None = None,
                 user: User = Depends(page_user), s: Session = Depends(get_session)):
    cfg = get_config()
    page = answer_page(s, cfg, prompt_id, date, provider, sample, compare)
    if page is None:
        raise HTTPException(404, "No answers for this prompt")
    hist_providers, hist_rows = history(s, prompt_id) if tab == "history" else ([], [])
    strategies, shown_strategy, queued = [], None, None
    if tab == "strategy":
        strategies = recent(s, prompt_id)
        done = [st for st in strategies if st.status in ("ok", "error")]
        shown_strategy = (
            next((st for st in done if st.id == sid), None)
            or next((st for st in done if st.status == "ok"), None)
            or (done[0] if done else None)
        )
        queued = open_request(s, prompt_id)
    return _render(request, "answers.html", user, p=page, compare=compare, tab=tab,
                   strategies=strategies, st=shown_strategy, queued=queued,
                   render_answer=render_answer, label=provider_label,
                   synced=lambda dt: local_time(dt, cfg.schedule.timezone),
                   hist_providers=hist_providers, hist_rows=hist_rows,
                   brand_terms=brand_terms(cfg), own_brand=cfg.brand.name)


@router.post("/answers/{prompt_id}/strategy", include_in_schema=False)
def strategy_request(request: Request, prompt_id: int, csrf: str = Form(""),
                     user: User = Depends(page_user), s: Session = Depends(get_session)):
    """Queue a strategy. The worker writes it; the web app never calls an LLM."""
    check_csrf(request, csrf)
    if s.get(Prompt, prompt_id) is None:
        raise HTTPException(404, "Unknown prompt")
    _row, created = request_strategy(s, prompt_id, user.username)
    msg = ("Queued. The worker writes it within about a minute; refresh to see it."
           if created else "A strategy for this prompt is already being written.")
    return RedirectResponse(f"/answers/{prompt_id}?tab=strategy&msg={quote(msg)}",
                            status_code=303)


# ---------- prompts ----------


@router.get("/admin/prompts", response_class=HTMLResponse, include_in_schema=False)
def prompts_page(request: Request, user: User = Depends(page_user),
                 s: Session = Depends(get_session)):
    rows = list(s.scalars(select(Prompt).order_by(Prompt.slug, Prompt.version.desc())))
    groups: dict[str, list[Prompt]] = {}
    for p in rows:
        groups.setdefault(p.slug, []).append(p)
    return _render(request, "prompts.html", user, groups=groups)


@router.post("/admin/prompts", include_in_schema=False)
def prompts_add(request: Request, slug: str = Form(...), text: str = Form(...),
                intent: str = Form(""), csrf: str = Form(""),
                user: User = Depends(page_user), s: Session = Depends(get_session)):
    check_csrf(request, csrf)
    try:
        p = add_prompt(s, slug, text, intent, user.username)
    except PromptError as e:
        return _back("/admin/prompts", err=str(e))
    return _back("/admin/prompts", msg=f"Added '{p.slug}'. The next daily run will ask it.")


@router.post("/admin/prompts/{prompt_id}/edit", include_in_schema=False)
def prompts_edit(request: Request, prompt_id: int, text: str = Form(...), csrf: str = Form(""),
                 user: User = Depends(page_user), s: Session = Depends(get_session)):
    check_csrf(request, csrf)
    try:
        p = edit_prompt(s, prompt_id, text, user.username)
    except PromptError as e:
        return _back("/admin/prompts", err=str(e))
    return _back("/admin/prompts", msg=f"'{p.slug}' is now version {p.version}.")


@router.post("/admin/prompts/{prompt_id}/stop", include_in_schema=False)
def prompts_stop(request: Request, prompt_id: int, csrf: str = Form(""),
                 user: User = Depends(page_user), s: Session = Depends(get_session)):
    check_csrf(request, csrf)
    try:
        p = stop_prompt(s, prompt_id)
    except PromptError as e:
        return _back("/admin/prompts", err=str(e))
    return _back("/admin/prompts", msg=f"Stopped '{p.slug}' v{p.version}. History is kept.")


# ---------- users (admin only) ----------


@router.get("/admin/users", response_class=HTMLResponse, include_in_schema=False)
def users_page(request: Request, user: User = Depends(admin_user),
               s: Session = Depends(get_session)):
    users = list(s.scalars(select(User).order_by(User.username)))
    return _render(request, "users.html", user, users=users)


@router.post("/admin/users", include_in_schema=False)
def users_add(request: Request, username: str = Form(...), password: str = Form(...),
              is_admin: bool = Form(False), csrf: str = Form(""),
              user: User = Depends(admin_user), s: Session = Depends(get_session)):
    check_csrf(request, csrf)
    try:
        create_user(s, username, password, is_admin)
    except ValueError as e:
        return _back("/admin/users", err=str(e))
    return _back("/admin/users", msg=f"Created '{username}'.")
