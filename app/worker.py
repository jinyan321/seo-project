"""The only process that calls the LLM APIs. Run exactly one replica: python -m app.worker"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select

from app import alerts, db, logs
from app.config import get_config
from app.models import Batch
from app.runner import run_batch
from app.strategy import run_pending, run_weekly

log = logging.getLogger("worker")


def job() -> None:
    try:
        run_batch()
    except Exception as e:
        log.exception("batch crashed")
        alerts.alert(f"batch crashed: {type(e).__name__}: {e}")
        alerts.ping(False)


def strategy_job(fn) -> None:
    try:
        fn()
    except Exception as e:
        log.exception("strategy job crashed")
        alerts.alert(f"strategy job crashed: {type(e).__name__}: {e}")


def missed_today() -> bool:
    """True if today's scheduled time has passed and today's batch isn't complete."""
    sch = get_config().schedule
    now = datetime.now(ZoneInfo(sch.timezone))
    if (now.hour, now.minute) < (sch.hour, sch.minute):
        return False
    with db.session() as s:
        b = s.scalar(select(Batch).where(Batch.scheduled_for == now.date()))
    return b is None or b.status != "complete"


def main() -> None:
    logs.setup()
    sch = get_config().schedule
    scheduler = BlockingScheduler(timezone=ZoneInfo(sch.timezone))
    scheduler.add_job(
        job, CronTrigger(hour=sch.hour, minute=sch.minute, timezone=ZoneInfo(sch.timezone)),
        id="daily-batch", misfire_grace_time=3600, coalesce=True, max_instances=1,
    )
    st = get_config().strategy
    if st.enabled:
        scheduler.add_job(
            strategy_job, CronTrigger(day_of_week=st.weekday, hour=st.hour,
                                      timezone=ZoneInfo(sch.timezone)),
            args=[run_weekly], id="weekly-strategy", misfire_grace_time=6 * 3600,
            coalesce=True, max_instances=1,
        )
        # "Generate now" requests from the web app are queued rows; pick them up quickly.
        scheduler.add_job(strategy_job, "interval", minutes=1, args=[run_pending],
                          id="pending-strategies", coalesce=True, max_instances=1)
    if sch.catch_up_on_start and missed_today():
        log.info("today's batch was missed; running it now")
        scheduler.add_job(job, id="catch-up", max_instances=1)
    log.info("worker started; daily batch at %02d:%02d %s", sch.hour, sch.minute, sch.timezone)
    scheduler.start()


if __name__ == "__main__":
    main()
