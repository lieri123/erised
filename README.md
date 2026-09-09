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

**Stage 2 — scoring.** XGBoost predicts CTR per eligible ad from 23 features
(17 hand-written, 6 read out of a learned ad × placement embedding table), then
`bid_value = predicted_ctr × target_cpm`. That multiplication is what lets
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

23 features (`adplatform/ml/features.py`), in two blocks that are never
interleaved.

The hand-written block (17): time-of-day and weekday, device one-hots, keyword
overlap and overlap ratio, ad and page keyword counts, target CPM, smoothed CTR
priors at ad / placement / pair level, log pair impression count, ad age, and
budget pacing.

The learned block (6): output of an embedding table fit by gradient descent on
the click log. See [Embeddings](#embeddings) below.

Training (`adplatform/ml/train_ctr.py`) does a time-ordered split, fits the
embedding table on the training split, downsamples negatives with a corrected
intercept, fits XGBoost with early stopping,
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

### Embeddings

Every hand-written feature is a request attribute or a count. `pair_ctr_prior`
is the strongest of them and it does not generalise across entities: the prior
for (`ad_9134`, `plc_4`) uses the impressions of that pair and nothing else, so
on thin pairs it shrinks back toward the global rate. Nothing tells the model
that `ad_9134` resembles the other outdoor-gear ads that do have data on
`plc_4`, and no amount of smoothing fixes that.

`adplatform/ml/embeddings.py` fits a dense vector per ad and per placement by
minibatch Adam on the click log, under a factorisation machine:

```
logit(click) = b + b_ad + b_placement + <e_ad, e_placement>
```

The inner product makes that a low-rank model of the pair matrix, so an ad with
fifty impressions borrows from ads sitting near it in the space. Six columns go
to XGBoost: the FM's own logit, the bare affinity term, both learned biases,
`||e_ad||`, and a cold flag for ids that fell back to the shared
out-of-vocabulary row. Raw latent coordinates are deliberately not fed to the
trees, since no single axis of an arbitrarily rotated embedding means anything
by itself.

None of this replaces the booster. Trees are good on dense axis-aligned inputs
and hopeless at learning a dot product over sparse ids; the FM knows nothing
about keywords or pacing. The same promotion gates decide whether the pair
ships.

From `tests/test_train_ctr_embeddings.py`, the same pipeline run with and
without the block on synthetic traffic:

| Inventory (60k impressions) | Log loss without | with | AUC without → with |
|---|---|---|---|
| 40 ads × 6 placements — ~250 impressions/pair | 0.1731 | 0.1728 | 0.759 → 0.759 |
| 250 ads × 20 placements — ~12 impressions/pair | 0.1987 | 0.1955 | 0.693 → 0.735 |

Log loss, lower is better; both rows are the same code and seed, differing only
in `--no-embeddings`. The first row is a null result and the expected one — with
hundreds of impressions per pair the count prior is already a good estimate and
there is little left to generalise. The gap opens where counting stops working,
which is the regime any real inventory is in.

The table is fit on the training split, never on the calibration split that
gates promotion, and saved as `embeddings.npz` inside the same immutable version
directory as `model.json`. It cannot be updated independently, because those six
columns only mean anything to the trees fit on the exact numbers they produced.
Serving loads both out of one artifact and holds them in a single object, so a
mid-request hot reload cannot mix versions, and refuses any artifact whose
`metadata.json` declares `uses_embeddings` without shipping the table. Fitting
runs in float64 and the table is narrowed to float32 before it is used at all,
so the vectors that built the training matrix are bit-identical to the ones the
gateway loads back off disk.

`--no-embeddings` trains the control: the block is filled with constants, the
artifact keeps the width serving expects, and the two runs are comparable.

Adding the block bumped `FEATURE_VERSION` to 3, which would normally make older
impressions untrainable. It does not here, because v3 appends to an unchanged v2
base: a v2 vector is a v3 vector's first 17 columns. The logged learned columns
are discarded on every row anyway, since they came from whichever table was live
at serve time rather than the one being fit. `train_ctr.py` recomputes each
row's block from the table it is about to ship, so v2 and v3 impressions train
side by side (`TRAINABLE_FEATURE_VERSIONS`).

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

The run logs what the embedding table learned and writes the same numbers to
`metadata.json` under `embedding.fit` and `metrics.embeddings_only`. The latter
is what the table scores alone on the held-out calibration rows, which is how
you tell whether the vectors learned anything or the trees are routing around
noise. To measure the block's contribution directly, train the control and
compare:

```bash
docker compose run --rm bootstrap \
  python -m adplatform.ml.train_ctr --days 30 --out /app/models \
  --dsn clickhouse://default@clickhouse:8123/default --no-embeddings --dry-run
```

Ads and placements with fewer than `--embedding-min-count` impressions (default
20) share one out-of-vocabulary row rather than getting a vector fit to four
impressions and one click.

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
pytest -q                         # 269 tests
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

`make train` writes into `./models`, which both the gateway and the bootstrap
job bind-mount. The image runs as uid 10001, and a bind mount takes the host
directory's ownership rather than the image's — so on a Linux host `./models`
has to be writable by that uid. Docker Desktop on macOS and Windows papers over
this and needs nothing.

---

## Scope

Built and working: real-time serving, three-stage auction with exploration,
campaign-level budget enforcement, HMAC click signing, the Kafka → ClickHouse
event pipeline, CTR training with promotion gates and hot model reload, a
publisher ad tag, and a demo page.

Deliberately not built:

- **Billing.** `cost_usd` is computed and logged. Nothing invoices it, charges a
  card, or handles prepaid balances or disputes. Budgets are enforced in Redis;
  money is not. Reconciliation corrects a spend counter upward only — a lower
  total from ClickHouse is the shape a Kafka outage makes, not evidence of an
  overcount, and lowering the counter would hand back every dropped
  impression's budget. Reconciling from the durable copy in Postgres instead
  would need a schema change: `impressions` stores `ad_id`, and budgets are
  per campaign.
- **Invalid traffic detection.** No bot filtering, click-fraud detection, IP or
  UA reputation, or viewability measurement.
- **Creative moderation.** The ad tag already sandboxes: `static/adtag.js`
  renders every creative in an iframe with `allow-scripts` and
  `allow-same-origin` deliberately withheld, so advertiser markup cannot reach
  the publisher's origin. What is missing is everything *before* that.
  `creative_html` is stored and served unsanitised and unreviewed, and
  `/v1/bid` returns it as `ad_markup` to a publisher who is free to ignore the
  tag and assign it to `innerHTML`. The sandbox is defence in depth against a
  creative that turns malicious after review; it is not a substitute for the
  review, and it protects nobody who does not use the tag.
- **Publisher key exposure.** `adtag.js` carries the publisher's API key in the
  browser, where it cannot be a secret. Anyone who views source can mint bid
  requests attributed to that publisher, bounded only by `BID_RATE_LIMIT`. CORS
  does not help — it is enforced by browsers, and `curl` ignores it. A real
  exchange has the publisher's own server sign the request, or issues
  short-lived origin-bound tokens. This is also what makes the missing invalid
  traffic detection above matter more than it otherwise would.
- **Human auth and dashboards.** API keys and a shared admin token are enough
  for machines, not for people.
- **Privacy compliance.** `user_id`, `page_url`, and device data are stored with
  no consent handling or deletion path.

Three honesty notes about the model results. Simulated clicks are drawn from a
formula, so a passing AUC means the model rediscovered that formula: it
validates the pipeline, not model quality on real traffic. That goes double for
the embeddings, since the simulator's `World` generates its ad × placement
affinity as a low-rank inner product, which is the structure a factorisation
machine represents. Recovering it shows the pipeline works and says nothing
about whether real inventory is low-rank. And the simulator writes labels
directly to ClickHouse, so `/v1/click` is exercised by tests and by hand but not
under load.
