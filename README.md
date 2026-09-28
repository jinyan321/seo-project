# AI Brand-Mention Tracker: HR software in Malaysia

Every day this service asks ChatGPT (OpenAI) and Claude (Anthropic) the same buyer questions, with web search on and location set to Malaysia. It records:
- whether your brand (`brand` in `config.yaml`, currently **Kakitangan**) appears in the answers
- which of 8 competitors appear
- which websites the AIs cite

The results are served as mention rate over time, plus related scores.

> This repository contains the application code: the worker, the web app, database migrations and tests. The dependency list (`requirements.txt`), the Caddy HTTPS config and the internal design notes are kept in the private deployment repo, so it does not build on its own.

## How it runs

```
Internet ─HTTPS─> Caddy ─> web (FastAPI)       reads only: /mention-rate, /gap-report, /dashboard, /admin
                               │
                           Postgres  <── backup (daily pg_dump)
                               ▲
worker (1 replica) ────────────┘  09:07 Malaysia time: 5 prompts × 2 AIs × 1 sample = 10 calls
```

- **Only the worker calls the AIs.** The web app only reads stored results.
- **Every raw answer is stored.** After changing brand names, `reextract` rechecks all history for free.
- **Failed calls are recorded but left out of the scores.** They never count as "not mentioned".
- **Prompts live in the database and are versioned.** Editing a prompt creates v2 and stops v1, so a trend never mixes two different questions.

## Run locally (no Docker, no API cost)

Create a `.env` in the project folder. It is git-ignored and the app reads it automatically, so this works the same in PowerShell and bash:

```
PROVIDER_MODE=fake
APP_USERNAME=admin
APP_PASSWORD=<choose-a-local-password>
SESSION_SECRET=local-dev-secret
COOKIE_SECURE=false
DATABASE_URL=sqlite:///./data/tracker.db
```

**Local Postgres (recommended, matches production).** Start it once. It restarts with Docker, and its data persists in a volume:

```
docker run -d --name tracker-dev-db --restart unless-stopped -p 127.0.0.1:5432:5432 -e POSTGRES_USER=tracker -e POSTGRES_PASSWORD=<pick-one> -e POSTGRES_DB=tracker -v tracker-dev-pgdata:/var/lib/postgresql/data postgres:16-alpine
```

Then set `DATABASE_URL=postgresql+psycopg://tracker:<pick-one>@127.0.0.1:5432/tracker` in `.env`. Use `127.0.0.1`, not `localhost`: on Windows, `localhost` tries IPv6 first and each connection stalls for seconds.

Then run these from the project folder (macOS/Linux: use `.venv/bin/` instead of `.venv\Scripts\`):

```
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt   # not included in this repo
.venv\Scripts\alembic upgrade head
.venv\Scripts\python -m app.cli demo --days 14     # fake data, no API calls
.venv\Scripts\uvicorn app.web.main:app --reload    # open http://localhost:8000, log in with APP_USERNAME / APP_PASSWORD
.venv\Scripts\python -m pytest -q                  # starts a throwaway Postgres in Docker; set TEST_DB=sqlite for a quick run
```

## Deploy (one VPS)

```bash
cp .env.example .env        # fill in keys, APP_PASSWORD, SESSION_SECRET, POSTGRES_PASSWORD, DOMAIN
docker compose up -d --build
```

This starts `db`, runs `migrate` (migrations, then seeds prompts from `config.yaml`), and starts `web`, `worker`, `caddy` (automatic HTTPS for `DOMAIN`) and `backup` (a daily dump into `./backups`). Copy `./backups` off the server, for example with rclone from a host cron job.

Before trusting the numbers, do the **first real run** by hand and check it. This costs real money, and `MAX_DAILY_USD` caps it:

```bash
docker compose exec worker python -m app.cli run
```

## Commands

| Command | What it does |
|---|---|
| `python -m app.cli run [--date YYYY-MM-DD]` | Run or resume one batch now. Only the missing calls are made |
| `python -m app.cli reextract [--sentiment]` | Rebuild mentions and citations from stored answers. No API cost unless you add `--sentiment` |
| `python -m app.cli demo --days N` | N days of fake data |
| `python -m app.cli seed` | Load `seed_prompts` from `config.yaml` if there are no prompts yet |
| `python -m app.cli create-user NAME [--admin]` | Add a login |
| `python -m app.cli strategy [--prompt SLUG]` | Write strategies now (Claude, `strategy.model` in config). Costs money unless in fake mode |
| `python -m app.worker` | The scheduler. Run exactly one |

## API (HTTP Basic auth)

`GET /mention-rate?brand=&provider=&prompt_id=&bucket=day|week|batch&since=&until=`

- `brand`: any tracked brand. Defaults to your own brand (Kakitangan).
- `prompt_id`: a prompt row id, or a slug. A slug means the newest version of that prompt.

```bash
curl -u admin:pass "https://your-host/mention-rate?bucket=week"
```

The response has two parts:
- `series`: one row per period, with `runs`, `mentioned`, `mention_rate`, `ci95` (95% Wilson range), `avg_rank` and `top1_rate`
- `summary`: the same scores over the whole range, plus `avg_rank_when_mentioned`, `own_domain_cited_rate`, `sample_agreement`, `trend`, `by_provider`, `by_prompt` (per version), `share_of_voice`, `top_cited_domains`, `sentiment` and `latest_snippets`

`GET /gap-report`: websites the AIs cite when they name competitors, but not when they name your brand. These are content and PR targets.

`GET /healthz`: public. Returns the latest batch status.

## Pages (login form)

- `/dashboard`: the mention-rate chart with its 95% range, the scores, share of voice and the gap report
- `/answers/{prompt_id}`: what each AI actually answered for a prompt, per day and sample. It shows the brands mentioned with their rank, the answer with brand names highlighted, cited sources and "found but not used" sources. It also has side-by-side compare, find-in-answers and a History tab
- `/answers/{prompt_id}?tab=strategy`: an AI-written strategy for that question, built from the last 7 days of answers. It covers where you stand, prioritised actions (each citing its evidence) and why competitors get picked. It's refreshed every Monday; **Generate now** queues one, and the worker writes it within about a minute
- `/admin/prompts`: add, edit (creates a new version) or stop prompts
- `/admin/users`: admins only

## Configuration

- **`config.yaml`:** brand, competitors, aliases and domains, models, schedule, prices and seed prompts.
  - Set `brand.domains` to get `own_domain_cited_rate`.
  - After editing brands, run `reextract`.
- **`.env`:** secrets and ops settings. See [.env.example](.env.example).

## Limits

- **Not yet run against the real APIs.** The parsers follow each API's documented format and are tested with hand-built samples in `tests/fixtures/`. After the first real run, save real responses there and read about 30 answers by hand against what the extractor recorded.
- **The API is close to the apps, but not the same.** The API with web search is not identical to the ChatGPT and Claude apps, which add personalisation.
- **Daily numbers are noisy.** 30 calls a day gives wide margins of error, so use `bucket=week` for decisions.
- **Check prices.** Prices in `config.yaml` drive the cost cap. Check them against the current price pages.
