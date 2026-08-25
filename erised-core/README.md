# erised-core

Rust port of the Erised matching core. This is **step one of four** — the
auction only.

| Stage | Python source | Ported |
|---|---|---|
| 1. Eligibility filter | `rtb.get_eligible_ads` | no |
| 2a. Feature extraction | `ml/features.extract_features` | no |
| 2b. CTR inference | `ml/ctr_model.predict_batch` | no |
| 2c. Bid value | `ml/rtb_integration.score_ads` | **yes** |
| 3. Auction and pricing | `ml/rtb_integration.run_auction` | **yes** |

The auction went first because it is pure arithmetic with no dependencies, it is
the piece where a bug shows up on an advertiser's invoice, and it already had a
Python test suite to diff against.

## Layout

```
erised-core/
  crates/erised-auction/   pure Rust, zero dependencies, 28 unit tests
    src/round.rs           CPython-compatible round(x, 6)
    src/rng.rs             SplitMix64 behind an Rng trait
    src/auction.rs         bid value, clearing price, epsilon-greedy
  crates/erised-py/        PyO3 bindings (module name: erised_core)
  pyproject.toml           maturin build config
```

## Build

```bash
cd erised-core
cargo test                              # Rust unit tests, no Python needed
pip install maturin
maturin develop --release                # installs erised_core into the venv
```

Then from the repo root:

```bash
python -m pytest tests/test_rust_parity.py -v
```

Copy `test_rust_parity.py` into the repo's `tests/` directory. It imports
`adplatform.ml.rtb_integration`, so it relies on the same `pythonpath = ["."]`
setting in `pyproject.toml` that the rest of the suite does.

## What the parity harness actually checks

Rust and Python are compared on inputs neither was written against:

- **100,000 rounding cases**, deliberately clustered on exact halfway values at
  the sixth decimal and their one-ulp neighbours. The naive translation
  (`(x * 1e6).round() / 1e6`) fails ~10% of these. `py_round` passes all of them.
- **20,000 clearing prices** across the full CTR clamp range, including CTRs
  below the `1e-9` divisor floor and floors misconfigured above `target_cpm`.
  Asserted with `==`, not `approx`.
- **2,000 full auctions** at `epsilon=0.0`, comparing winner identity, clearing
  price, `cost_usd`, and propensity exactly.
- **60,000 auctions at `epsilon=0.10`**, confirming that each ad's *observed*
  serve frequency matches its *logged* propensity. This is the check that would
  have caught the original `serve_propensity` bug.

Exploration decisions are compared statistically rather than stream-for-stream:
the two implementations use different RNGs on purpose, so diffing individual
draws would measure the RNGs rather than the auction.

## One thing to confirm before you run this

`run_auction` here returns `(1 - eps) + eps/n` when the exploration branch
happens to land on the ad the model would have picked anyway. The copy of
`rtb_integration.py` I was working from returns `eps/n` unconditionally in that
branch — the bug you already found and fixed. If your working tree still has
the unfixed version, `test_auction_matches_with_exploration_disabled` will pass
(it runs at `epsilon=0.0`) and `test_serve_propensity_matches` will fail. That
failure means the Python needs the fix, not the Rust.

## CI

Add alongside the existing `test` and `build` jobs. It must **fail**, not skip,
when the extension is missing — a green build that silently skipped the only
test comparing the two pricing implementations is worse than no test:

```yaml
  parity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: dtolnay/rust-toolchain@stable
      - name: Rust unit tests
        run: cargo test --manifest-path erised-core/Cargo.toml
      - name: Build extension
        run: |
          pip install maturin pytest
          maturin develop --release -m erised-core/crates/erised-py/Cargo.toml
      - name: Fail if the extension is missing
        run: python -c "import erised_core"
      - name: Parity
        run: python -m pytest tests/test_rust_parity.py -v
```

## Notes for the next stage

- The PyO3 boundary currently marshals candidates as tuples of primitives. That
  is right for proving parity and wrong for serving traffic — once features and
  inference move over, the inventory snapshot should live on the Rust side and
  the boundary should carry ad *ids*, not ad *objects*.
- `features.rs` is the next port and the harder one. The traps, in order of how
  likely they are to bite: feature vectors are computed in f64 then narrowed to
  `float32` before the booster sees them, so the narrowing has to happen at the
  same point; keyword normalisation needs `BTreeSet<String>` to match
  `sorted({...})`; and ad keywords get `.lower()` but not `.strip()` while page
  keywords get both — replicate the inconsistency rather than fixing it, or
  `n_ad_keywords` drifts.
- `model.rs` needs `train_ctr.py` to dump `calibrator.json` (the isotonic knots)
  alongside `calibrator.pkl` first. Rust cannot read the pickle. Do that export
  before starting the model crate, not during.
