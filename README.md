# erised

A real-time ad serving platform with a learned click-through-rate model in the
bid path. A publisher's page calls `/v1/bid`, the gateway runs a three-stage
auction over live inventory, and returns a creative in single-digit
milliseconds. Every impression logs its own feature vector, which is what makes
the next model trainable.

Not a toy: the serving loop, the event pipeline, and the training pipeline are
all wired end to end, and a trained model has passed the promotion gates on
simulated traffic. Also not a product — see [Scope](#scope) for what is
deliberately missing.

---

## Architecture

```
publisher page
      │  static/adtag.js
      ▼
┌─────────────────────┐      ┌──────────────┐
│  FastAPI gateway    │─────▶│  Redis       │  budget counters, rate limits
│                     │      └──────────────┘
│  1. eligibility     │      ┌──────────────┐
│  2. CTR scoring     │◀─────│  Postgres    │  campaigns, ads, api keys,
│  3. second-price    │      └──────────────┘  impressions
│     auction         │      ┌──────────────┐
└──────────┬──────────┘─────▶│  Redpanda    │  impressions, clicks,
           │                 └──────┬───────┘  conversions
           │ creative + signed             │ Kafka table engine
           │ click URL                     ▼
           ▼                        ┌──────────────┐
      browser ──── click ─────────▶ │  ClickHouse  │  90-day TTL, CTR aggregates
                                    └──────┬───────┘
                                           │
                                    train_ctr.py ──▶ models/current
                                           │              │
                                           └── gates ─────┘ hot-reloaded by
                                                            the gateway
```

### The bid path

**Stage 1 — eligibility.** A hard filter against an in-memory inventory
snapshot refreshed from Postgres: device targeting, active flag, flight dates,
floor price, and remaining budget. No model involvement.

**Stage 2 — scoring.** XGBoost predicts CTR per eligible ad from 17 features,
then `bid_value = predicted_ctr × target_cpm`. That multiplication is what lets
relevance beat budget: an ad bidding $8 at 0.4% CTR loses to one bidding $4 at
1.2%, because the second is worth more per impression to everyone involved.

**Stage 3 — auction.** Second-price against the runner-up's bid value, floored
at the winner's floor price. A configurable fraction of auctions
(`EXPLORATION_EPSILON`, default 8%) is decided randomly instead — without it the
model only ever observes clicks on ads it already favours, and a genuinely
better ad it happens to underrate never gets served, never accumulates data,
and never corrects the model. The bias is self-reinforcing and no amount of
modelling recovers from it. Serve propensity is logged per impression so
training can correct for it.

### The model

17 features (`adplatform/ml/features.py`): time-of-day and weekday, device
one-hots, keyword overlap and overlap ratio, ad and page keyword counts, target
CPM, smoothed CTR priors at ad / placement / pair level, log pair impression
count, ad age, and budget pacing.

Training (`adplatform/ml/train_ctr.py`) does a time-ordered split, downsamples
negatives with a corrected intercept, fits XGBoost with early stopping,
applies isotonic calibration on a held-out split, and then refuses to promote
unless all three gates pass:

| Gate | Threshold |
|---|---|
| Log loss | must beat the smoothed-prior baseline |
| Calibration ratio | within `[0.85, 1.15]` |
| AUC | above `0.55` |

A blocked run still writes its versioned directory so it can be inspected.
Promotion is an atomic-ish swap of `models/current`, and the serving process
picks it up by watching `metadata.json`'s mtime — no restart.

Feature vectors are logged at serve time rather than recomputed at train time,
which is the standard defence against train/serve skew: if the two ever
disagree, the model learns from inputs it will never see in production.

The same care applies to *which trees* run. Early stopping leaves the booster
holding rounds past the optimum, and training scores, calibrates, and gates only
the `best_iteration` prefix — so serving records that prefix in `metadata.json`
and predicts with it. Serving the whole booster would put a model in production
that nothing ever measured, differing precisely in the rounds early stopping
identified as overfitting.

---

## Quickstart

Requires Docker with Compose v2. Commands below are shell; on Windows use
PowerShell equivalents.

```bash
./scripts/preflight.sh          # checks docker, ports, image tags, files
cp .env.example .env            # then edit — see Configuration
make up                         # build, start, wait for health, verify
make seed                       # advertisers, campaigns, ads
```

`make up` finishes by running `scripts/verify_stack.sh`, which asserts all four
stores are connected and the Kafka topics exist. If that passes, open:

- <http://localhost:8000/demo> — a demo publisher page rendering a live ad
- <http://localhost:8000/docs> — OpenAPI

### Train a model

The gateway serves on a smoothed-prior heuristic until a model is promoted;
`/health` reports `"ctr_model": "baseline"` in that state. To change it you need
volume — the gates reject a model trained on too few rows.

```bash
# raise BID_RATE_LIMIT to 10000/minute in .env first, then recreate the gateway
docker compose run --rm -e SIM_API_KEY="<key>" bootstrap \
  python -m scripts.simulate_traffic --impressions 100000 \
  --base-url http://gateway:8000 --clickhouse-host clickhouse

docker compose run --rm bootstrap \
  python -m adplatform.ml.train_ctr --days 30 --out /app/models \
  --dsn clickhouse://default@clickhouse:8123/default
```

At 20k impressions the calibration gate sits at roughly 1.2 sigma of pure
sampling noise and fails a large fraction of the time on a perfectly calibrated
model. 100k puts it near 2.6 sigma, where the gate measures calibration rather
than luck.

---

## Configuration

`.env` is gitignored; `.env.example` is the template. The values that matter:

| Variable | Notes |
|---|---|
| `API_KEY_PEPPER` | Required. Also the HMAC key for click signatures — rotating it invalidates every API key **and** every outstanding click URL. |
| `ADMIN_TOKEN` | Guards `/admin/*`. |
| `PUBLIC_BASE_URL` | Stamped into click URLs. Must be reachable by whoever follows them — `localhost` is wrong from inside a container. |
| `POSTGRES_PASSWORD` | Baked into the volume on first `up`; changing it later does not change the database password. |
| `BID_RATE_LIMIT` | Default `120/minute`. Raise only while simulating. |
| `EXPLORATION_EPSILON` | Fraction of auctions decided randomly. |

The gateway refuses to start in production with a default pepper, an empty
admin token, or other insecure config — `settings.validate_for_production()`
runs in the lifespan and aborts boot rather than logging a warning nobody reads.

---

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /v1/bid` | publisher key | Run an auction, return a creative and a signed click URL |
| `GET /v1/click` | signature | Verify HMAC, record the click, 302 to the destination |
| `GET /v1/impression` | — | Tracking pixel |
| `POST /v1/conversion` | publisher key | Post-click conversion |
| `GET /v1/stats` | publisher key | Publisher reporting |
| `POST /v1/campaigns`, `POST /v1/ads` | advertiser key | Create inventory |
| `GET /v1/advertiser/stats` | advertiser key | Advertiser reporting |
| `/admin/*` | admin token | Provision publishers, advertisers, keys |
| `GET /health` | — | Add `?deep=true` for per-store status |

API keys are hashed with HMAC-SHA256 under the server pepper — deterministic,
so the hash doubles as an O(1) lookup index rather than forcing a linear scan
across every stored hash on the hot path. Keys carry an `owner_type`, checked
in the dependency rather than in each handler, so a publisher key cannot call
an advertiser endpoint and a new endpoint cannot forget to check.

Click URLs are signed over `impression_id:expiry`, truncated to 128 bits,
base64url-encoded. Unsigned, expired, and tampered URLs all get the same vague
403 — the log records which, the response never does, because telling a forger
whether the expiry or the signature failed tells them which half to work on.

---

## Development

```bash
pip install -r requirements.txt
pytest -q                         # 244 tests
python -m scripts.check_imports   # every module must import
```

The eight tests in `tests/test_rust_parity.py` need the Rust extension
(`pip install ./erised-core`) and skip without it. CI's `parity` job builds it
and fails if the module is missing, so the skip cannot go green unnoticed.

CI runs both on 3.11 and 3.12 plus a Docker build. `check_imports` exists
because a cleanup commit once deleted a `from dataclasses import ...` while
leaving the decorator in place: every file still compiled, a linter would have
passed, and the gateway would not boot. It covers modules no test imports.

```
make up        make down       make reset      # reset DELETES volumes
make seed      make verify     make train
make logs      make shell-pg   make shell-ch
```

---

## Scope

Built and working: real-time serving, three-stage auction with exploration,
campaign-level budget enforcement, HMAC click signing, the Kafka → ClickHouse
event pipeline, CTR training with promotion gates and hot model reload, a
publisher ad tag, and a demo page.

Deliberately not built:

- **Billing.** `cost_usd` is computed and logged. Nothing invoices it, charges a
  card, or handles prepaid balances or disputes. Budgets are enforced in Redis;
  money is not.
- **Invalid traffic detection.** No bot filtering, click-fraud detection, IP or
  UA reputation, or viewability measurement.
- **Creative sandboxing.** `creative_html` is injected into publisher pages
  verbatim, which is arbitrary JavaScript on someone else's domain. A real
  deployment needs moderation and an iframe sandbox.
- **Human auth and dashboards.** API keys and a shared admin token are enough
  for machines, not for people.
- **Privacy compliance.** `user_id`, `page_url`, and device data are stored with
  no consent handling or deletion path.

Two honesty notes about the model results: simulated clicks are drawn from a
formula, so a passing AUC means the model rediscovered that formula — it
validates the pipeline, not model quality on real traffic. And the simulator
writes labels directly to ClickHouse, so `/v1/click` is exercised by tests and
by hand but not under load.
