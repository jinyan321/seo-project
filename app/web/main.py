"""FastAPI app. Run with: uvicorn app.web.main:app"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from app import db
from app.auth import LoginRequired, bootstrap_admin
from app.settings import get_settings
from app.web import api, pages


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    with db.session() as s:
        if bootstrap_admin(s, settings.app_username, settings.app_password):
            logging.getLogger("web").info("created first admin '%s'", settings.app_username)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    if settings.session_secret == "change-me" and settings.cookie_secure:
        raise RuntimeError("set SESSION_SECRET to a long random value")
    app = FastAPI(title="AI Brand-Mention Tracker", lifespan=lifespan)
    app.add_middleware(
        SessionMiddleware, secret_key=settings.session_secret, https_only=settings.cookie_secure,
        same_site="lax", max_age=7 * 24 * 3600,
    )

    @app.exception_handler(LoginRequired)
    async def _to_login(_request: Request, _exc: LoginRequired):
        return RedirectResponse("/login", status_code=303)

    app.include_router(api.router)
    app.include_router(pages.router)
    return app


app = create_app()
