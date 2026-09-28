"""python -m app.cli run | reextract | demo | seed | create-user"""

import argparse
import getpass
import json
import sys
from datetime import date, timedelta

from app import db, logs
from app.auth import create_user
from app.config import get_config
from app.prompts import seed_prompts
from app.providers.fake import FakeProvider
from app.runner import reextract, run_batch, today
from app.sentiment import FakeLabeler
from app.settings import get_settings


def cmd_run(args: argparse.Namespace) -> int:
    day = date.fromisoformat(args.date) if args.date else None
    res = run_batch(day)
    if res is None:
        print("another batch is running; nothing done")
        return 1
    print(json.dumps(res.__dict__))
    return 0 if res.status == "complete" else 2


def cmd_reextract(args: argparse.Namespace) -> int:
    print(json.dumps(reextract(with_sentiment=args.sentiment)))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Fill the database with fake batches for the last N days. No API cost."""
    cfg = get_config()
    with db.session() as s:
        seed_prompts(s, cfg)
    providers = [FakeProvider(cfg, "anthropic", own_bias=0.35),
                 FakeProvider(cfg, "openai", own_bias=0.2)]
    end = today(cfg)
    for i in range(args.days - 1, -1, -1):
        res = run_batch(end - timedelta(days=i), providers=providers, labeler=FakeLabeler())
        print(json.dumps(res.__dict__ if res else None))
    return 0


def cmd_strategy(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from app.models import Prompt
    from app.prompts import active_prompts
    from app.strategy import queue, run_pending

    with db.session() as s:
        if args.prompt:
            ids = list(s.scalars(select(Prompt.id).where(Prompt.slug == args.prompt,
                                                         Prompt.status == "active")))
            if not ids:
                print(f"error: no active prompt '{args.prompt}'", file=sys.stderr)
                return 1
        else:
            ids = [p.id for p in active_prompts(s)]
    queue(ids, "cli")
    print(f"generated {run_pending()} strategies")
    return 0


def cmd_seed(_args: argparse.Namespace) -> int:
    with db.session() as s:
        n = seed_prompts(s, get_config())
    print(f"seeded {n} prompts" if n else "prompts table not empty; nothing seeded")
    return 0


def cmd_create_user(args: argparse.Namespace) -> int:
    password = args.password or getpass.getpass("Password: ")
    with db.session() as s:
        try:
            u = create_user(s, args.username, password, args.admin)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    print(f"created {'admin' if u.is_admin else 'user'} '{u.username}'")
    return 0


def main(argv: list[str] | None = None) -> int:
    logs.setup()
    p = argparse.ArgumentParser(prog="app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run or resume one batch now (REAL API COST unless fake mode)")
    r.add_argument("--date", help="YYYY-MM-DD; defaults to today in the schedule timezone")
    r.set_defaults(fn=cmd_run)

    x = sub.add_parser("reextract", help="rebuild mentions/citations from stored raw answers")
    x.add_argument("--sentiment", action="store_true", help="also re-label sentiment (API cost)")
    x.set_defaults(fn=cmd_reextract)

    d = sub.add_parser("demo", help="fake data for the last N days")
    d.add_argument("--days", type=int, default=7)
    d.set_defaults(fn=cmd_demo)

    st = sub.add_parser("strategy", help="write strategies now (REAL API COST unless fake mode)")
    st.add_argument("--prompt", help="slug of one active prompt; default: all active prompts")
    st.set_defaults(fn=cmd_strategy)

    sub.add_parser("seed", help="load seed_prompts if the table is empty").set_defaults(fn=cmd_seed)

    u = sub.add_parser("create-user")
    u.add_argument("username")
    u.add_argument("--admin", action="store_true")
    u.add_argument("--password", help="omit to be prompted")
    u.set_defaults(fn=cmd_create_user)

    args = p.parse_args(argv)
    get_settings()  # fail early on bad env
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
