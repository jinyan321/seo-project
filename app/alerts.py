"""Failure alerts (Slack-compatible webhook) and the healthchecks.io dead-man ping.

Alerting must never crash a batch, so every error here is logged and swallowed.
"""

import logging

import httpx

from app.settings import get_settings

log = logging.getLogger(__name__)


def alert(message: str) -> None:
    url = get_settings().alert_webhook_url
    log.warning("ALERT: %s", message)
    if not url:
        return
    try:
        httpx.post(url, json={"text": f"[brand-tracker] {message}"}, timeout=10)
    except httpx.HTTPError as e:
        log.error("alert webhook failed: %s", e)


def ping(ok: bool) -> None:
    url = get_settings().hc_ping_url
    if not url:
        return
    try:
        httpx.get(url if ok else f"{url.rstrip('/')}/fail", timeout=10)
    except httpx.HTTPError as e:
        log.error("healthcheck ping failed: %s", e)
