"""Differential tests: Rust matching core vs. the Python implementation.

This is the file that makes the port safe. Every other test asserts that the
Rust code does what someone *thinks* the Python code does; this one asserts that
it does what the Python code *actually* does, on inputs neither implementation
was written against.

Skips cleanly when the extension is not built, so the suite still runs for
anyone who has not installed a Rust toolchain. Do not let that skip become
permanent in CI — a green build that silently skipped the only test comparing
the two pricing implementations is worse than no test at all. The `parity` job
in the workflow should fail if the module is missing.

Build with:
    maturin develop --release -m erised-core/crates/erised-py/Cargo.toml
"""

import math
import random
from dataclasses import dataclass

import pytest

erised_core = pytest.importorskip(
    "erised_core", reason="Rust extension not built; run `maturin develop --release`"
)

from adplatform.ml.rtb_integration import (  # noqa: E402
    ScoredAd,
    clearing_cpm,
    run_auction,
)

SEED = 20260819
N_CASES = 20_000


@dataclass
class StubAd:
    """Duck-typed stand-in for rtb.Ad.

    Only the three fields the auction reads. Following the convention in
    features.py, which duck-types `ad` for the same reason: the training path
    reconstructs ads from ClickHouse rows, not from live Ad objects.
    """

    ad_id: str
    target_cpm: float
    floor_price: float


def py_scored(ad_id, target_cpm, floor_price, ctr):
    ad = StubAd(ad_id, target_cpm, floor_price)
    return ScoredAd(
        ad=ad,
        predicted_ctr=ctr,
        bid_value=round(ctr * target_cpm, 6),
        features=[],
    )


def random_candidate(rng, i):
    return (
        f"ad_{i:04d}",
        round(rng.uniform(0.10, 25.0), 4),   # target_cpm
        round(rng.uniform(0.0, 6.0), 4),     # floor_price, sometimes above target
        rng.choice(
            [
                rng.uniform(0.0001, 0.25),   # normal range
                0.0001,                      # MIN_CTR clamp
                0.25,                        # MAX_CTR clamp
                rng.uniform(0.0, 1e-8),      # near the 1e-9 divisor floor
            ]
        ),
    )


# ---------------------------------------------------------------------------
# The rounding primitive
# ---------------------------------------------------------------------------


def test_round6_matches_cpython_on_adversarial_values():
    """The single highest-risk line in the port.

    Values are drawn to cluster near exact halfway cases at the sixth decimal,
    because that is the only region where round-half-even and round-half-away
    disagree. Uniform sampling would almost never hit it.
    """
    rng = random.Random(SEED)
    values = []
    for _ in range(N_CASES):
        base = rng.randrange(0, 10_000_000)
        # x.xxxxxx5 exactly, plus one-ulp perturbations either side of it.
        halfway = (base * 10 + 5) / 1e7
        values.extend(
            [
                halfway,
                math.nextafter(halfway, math.inf),
                math.nextafter(halfway, -math.inf),
                rng.uniform(0, 100),
                -rng.uniform(0, 100),
            ]
        )

    mismatches = [
        (v, round(v, 6), erised_core.round6(v, 6))
        for v in values
        if erised_core.round6(v, 6) != round(v, 6)
    ]
    assert not mismatches, f"{len(mismatches)} of {len(values)} diverged, e.g. {mismatches[:5]}"


def test_bid_value_matches():
    rng = random.Random(SEED + 1)
    for _ in range(N_CASES):
        ctr = rng.uniform(0.0, 0.25)
        cpm = rng.uniform(0.0, 25.0)
        assert erised_core.bid_value(ctr, cpm) == round(ctr * cpm, 6)


# ---------------------------------------------------------------------------
# Clearing price
# ---------------------------------------------------------------------------


def test_clearing_price_matches():
    """Exact equality, not approximate.

    `pytest.approx` would be the wrong assertion here. This is a billing figure;
    a discrepancy in the sixth decimal across a billion impressions is real
    money, and more importantly it would mean the two implementations disagree
    about the rounding rule, which will eventually produce a much larger
    divergence somewhere else.
    """
    rng = random.Random(SEED + 2)
    for i in range(N_CASES):
        ad_id, cpm, floor, ctr = random_candidate(rng, i)
        runner_up = rng.choice([rng.uniform(0.0, 2.0), 0.0, ctr * cpm])

        expected = clearing_cpm(py_scored(ad_id, cpm, floor, ctr), runner_up)
        got = erised_core.clearing_price(ctr, cpm, floor, runner_up)

        assert got == expected, (
            f"clearing price diverged: ctr={ctr!r} cpm={cpm!r} floor={floor!r} "
            f"runner_up={runner_up!r} -> python {expected!r}, rust {got!r}"
        )


def test_clearing_price_never_exceeds_target_cpm():
    """Invariant both implementations must hold, checked independently.

    Parity alone would not catch a bug that both sides share.
    """
    rng = random.Random(SEED + 3)
    for i in range(N_CASES):
        ad_id, cpm, floor, ctr = random_candidate(rng, i)
        runner_up = rng.uniform(0.0, 5.0)
        assert erised_core.clearing_price(ctr, cpm, floor, runner_up) <= cpm


# ---------------------------------------------------------------------------
# Full auction
# ---------------------------------------------------------------------------


def test_auction_matches_with_exploration_disabled():
    """Compared at epsilon=0.0 on purpose.

    The two implementations use different RNGs by design, so exploration
    decisions cannot line up and comparing them would only measure the RNGs.
    Disabling exploration isolates what the port actually has to get right:
    ranking, tie-breaking, and pricing. Exploration is covered by
    test_serve_propensity_matches below and by the Rust unit tests.
    """
    rng = random.Random(SEED + 4)
    for trial in range(2_000):
        n = rng.randrange(1, 12)
        raw = [random_candidate(rng, i) for i in range(n)]

        py_ads = [py_scored(*c) for c in raw]
        expected = run_auction(py_ads, impression_id="imp", epsilon=0.0)
        got = erised_core.auction(raw, "imp", 0.0, 0)

        assert (expected is None) == (got is None)
        if expected is None:
            continue

        assert py_ads[got.winner_index] is expected.winner, (
            f"trial {trial}: different winner. python picked "
            f"{expected.winner.ad.ad_id}, rust picked {raw[got.winner_index][0]}"
        )
        assert got.win_price == expected.win_price, (
            f"trial {trial}: price diverged, python {expected.win_price!r} "
            f"vs rust {got.win_price!r} on {raw}"
        )
        assert got.cost_usd == expected.cost_usd
        assert got.serve_propensity == expected.serve_propensity
        assert got.is_exploration == expected.is_exploration


def test_empty_candidate_set_is_no_fill_in_both():
    assert run_auction([], impression_id="imp", epsilon=0.0) is None
    assert erised_core.auction([], "imp", 0.0, 0) is None


def test_serve_propensity_matches():
    """Both branches of the propensity formula, checked without an RNG.

    This is the arithmetic the IPS weighting depends on, and it is the one place
    the exploration branch can be verified across the boundary deterministically:
    force epsilon to 1.0 so Rust always explores, then check that whichever ad it
    lands on carries the propensity Python would have logged for that ad.
    """
    rng = random.Random(SEED + 5)
    for seed in range(2_000):
        n = rng.randrange(2, 10)
        raw = [random_candidate(rng, i) for i in range(n)]
        py_ads = [py_scored(*c) for c in raw]
        greedy = max(range(n), key=lambda i: (py_ads[i].bid_value, -i))

        got = erised_core.auction(raw, "imp", 1.0, seed)
        assert got.is_exploration

        eps = 1.0
        expected = (1.0 - eps) + eps / n if got.winner_index == greedy else eps / n
        assert got.serve_propensity == pytest.approx(expected, abs=1e-12)


def test_propensities_form_a_distribution():
    """Sum over all candidates must be 1.0, or IPS weights do not average to 1.

    Checked empirically rather than algebraically: run many auctions, count how
    often each index is served, and confirm the observed frequency matches the
    logged propensity. This catches a propensity that is self-consistent but
    describes the wrong policy.
    """
    raw = [
        ("a", 5.00, 1.00, 0.02),
        ("b", 8.00, 2.00, 0.01),
        ("c", 4.00, 0.50, 0.03),
        ("d", 6.00, 1.50, 0.015),
    ]
    eps = 0.10
    n = len(raw)
    trials = 60_000
    served = [0] * n

    # Derive the greedy index from the Python implementation rather than
    # hardcoding it. Hardcoding is how this test failed the first time it ran:
    # 'c' bids 0.03 x 4.00 = 0.12 and beats 'a' at 0.02 x 5.00 = 0.10, which is
    # the whole point of ranking on bid_value and is easy to eyeball wrong.
    py_ads = [py_scored(*c) for c in raw]
    greedy = max(range(n), key=lambda i: (py_ads[i].bid_value, -i))

    for seed in range(trials):
        served[erised_core.auction(raw, "imp", eps, seed).winner_index] += 1

    for i, count in enumerate(served):
        observed = count / trials
        expected = (1.0 - eps) + eps / n if i == greedy else eps / n
        assert abs(observed - expected) < 0.01, (
            f"index {i} served {observed:.4f} of the time but the policy "
            f"implies {expected:.4f}"
        )
