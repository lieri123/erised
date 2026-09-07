"""Time the auction-and-pricing stage: Python reference vs. the Rust core.

Scope, because the number this prints is easy to overclaim. It measures
`run_auction` against `erised_core.auction` and nothing else. It does not
measure a bid request: eligibility filtering, feature extraction, XGBoost
inference, the Redis budget check and the Kafka emit are all outside it, and
`run_rtb` still calls the Python `run_auction`, so no part of this is on the
serving path yet. Any claim built on this output has to say "auction and
pricing stage".

Three arms:

  python          `run_auction` over a prebuilt list of ScoredAd
  rust            `erised_core.auction` over a prebuilt list of tuples,
                  including extraction of every tuple across the FFI boundary
  rust_marshal    `erised_core.marshal_only` — the same extraction with no
                  auction after it

The third is not a result on its own. It exists so `rust - rust_marshal` can
attribute how much of the Rust arm is boundary crossing rather than auction,
which is the case for eventually moving the inventory snapshot to the Rust side.
The honest headline is the `rust` arm: it is what the port costs today, paid in
full.

Both arms are checked for identical output before either is timed. A speedup
over code that computes something else is not a speedup.

    python scripts/bench_auction.py --iterations 50000 --epsilon 0.08 --max-ads 40
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import erised_core
except ImportError:  # pragma: no cover
    sys.exit(
        "erised_core is not importable.\n"
        "Build it first:\n"
        "    maturin develop --release -m erised-core/crates/erised-py/Cargo.toml\n"
        "Exiting rather than running a Python-only benchmark, which would look "
        "like a result and mean nothing."
    )

from adplatform.ml.rtb_integration import ScoredAd, run_auction  # noqa: E402


@dataclass
class StubAd:
    """Only the three fields the auction reads.

    Same approach as tests/test_rust_parity.py. Constructing real `rtb.Ad`
    objects would drag in Postgres row types for no benefit — the auction never
    looks at the other fields.
    """

    ad_id: str
    target_cpm: float
    floor_price: float
    campaign_id: str = ""
    advertiser_id: str = ""


def make_candidate_sets(n_sets: int, max_ads: int, seed: int):
    """Pre-generate the inputs so neither arm pays for random number generation.

    Sets vary in size between max_ads//2 and max_ads rather than all being
    max_ads. A fixed candidate count would let the CPU's branch predictor learn
    the loop trip count, which flatters both arms but flatters the Rust one more.
    """
    rng = random.Random(seed)
    tuple_sets, scored_sets = [], []

    for _ in range(n_sets):
        n = rng.randint(max(1, max_ads // 2), max_ads)
        raw = []
        for i in range(n):
            target_cpm = round(rng.uniform(0.10, 25.0), 4)
            raw.append((
                f"ad_{i:04d}",
                target_cpm,
                round(target_cpm * rng.uniform(0.2, 0.6), 4),
                rng.uniform(0.0001, 0.25),
            ))
        tuple_sets.append(raw)
        scored_sets.append([
            ScoredAd(
                ad=StubAd(ad_id, cpm, floor),
                predicted_ctr=ctr,
                bid_value=round(ctr * cpm, 6),
                features=[],
            )
            for ad_id, cpm, floor, ctr in raw
        ])

    return tuple_sets, scored_sets


def assert_parity(tuple_sets, scored_sets) -> None:
    """Confirm the two arms agree before timing them.

    At epsilon=0 only. The two implementations use different RNGs by design, so
    at any other epsilon the exploration decisions diverge and this would fail
    for a reason that has nothing to do with correctness. tests/test_rust_parity.py
    covers the exploration branch properly.
    """
    for i, (raw, scored) in enumerate(zip(tuple_sets, scored_sets)):
        rust = erised_core.auction(raw, "bench", 0.0, 0)
        py = run_auction(scored, "bench", 0.0, random.Random(0))
        if (rust is None) != (py is None):
            sys.exit(f"set {i}: one arm filled and the other did not")
        if py is None:
            continue
        if scored[rust.winner_index] is not py.winner:
            sys.exit(f"set {i}: different winner; refusing to time divergent code")
        if rust.win_price != py.win_price or rust.cost_usd != py.cost_usd:
            sys.exit(
                f"set {i}: price diverged, python {py.win_price!r} "
                f"vs rust {rust.win_price!r}; refusing to time divergent code"
            )


def timer_overhead_ns(samples: int = 20_000) -> float:
    """What an empty measurement costs.

    The Rust arm lands close enough to this that ignoring it would be dishonest:
    if the floor is 60ns and the arm reads 900ns, roughly 7% of the reported
    figure is the clock, not the code.
    """
    perf = time.perf_counter_ns
    deltas = []
    for _ in range(samples):
        a = perf()
        b = perf()
        deltas.append(b - a)
    return statistics.median(deltas)


def summarise(durations_ns: list[int], overhead_ns: float) -> dict:
    ordered = sorted(durations_ns)
    n = len(ordered)

    def pct(p: float) -> float:
        # Nearest-rank. At 50k samples the gap from an interpolating percentile
        # is far below run-to-run variance on a loaded machine.
        return ordered[min(n - 1, int(p * n))] / 1000.0

    return {
        "n": n,
        "p50_us": round(pct(0.50), 4),
        "p95_us": round(pct(0.95), 4),
        "p99_us": round(pct(0.99), 4),
        "max_us": round(ordered[-1] / 1000.0, 4),
        "mean_us": round(statistics.fmean(ordered) / 1000.0, 4),
        "mean_us_less_timer": round(
            max(0.0, statistics.fmean(ordered) - overhead_ns) / 1000.0, 4
        ),
    }


def run_python_stage3_arm(scored_sets, iterations, epsilon, warmup, seed):
    """`run_auction` alone, over ScoredAd objects that already exist.

    Not the arm to compare against Rust — see run_python_full_arm. Reported
    because the gap between the two is the whole story of where the Rust arm's
    advantage comes from, and leaving it out would look like padding.
    """
    perf = time.perf_counter_ns
    rng = random.Random(seed)
    n_sets = len(scored_sets)

    for i in range(warmup):
        run_auction(scored_sets[i % n_sets], "bench", epsilon, rng)

    out = []
    for i in range(iterations):
        scored = scored_sets[i % n_sets]
        a = perf()
        run_auction(scored, "bench", epsilon, rng)
        out.append(perf() - a)
    return out


def run_python_full_arm(tuple_sets, iterations, epsilon, warmup, seed):
    """Bid-value construction plus the auction — the scope Rust actually covers.

    `erised_core.auction` calls `ScoredAd::new` on every candidate, which is
    where `bid_value = py_round(ctr * cpm, 6)` happens. In Python that line lives
    in `score_ads`, one stage earlier. Timing Python's `run_auction` against the
    Rust binding therefore compares n rounding operations against zero, and the
    Rust arm loses a race it was never in.

    This arm does what `score_ads` does to the candidate list and then runs the
    auction, so both sides round n times and sort once. It is the comparison the
    speedup figure should be quoted from. It excludes CTR inference, which is
    the rest of stage 2 and is not ported.
    """
    perf = time.perf_counter_ns
    rng = random.Random(seed)
    n_sets = len(tuple_sets)

    # The Ad objects are prebuilt on purpose. In the serving path they come out
    # of the inventory cache and are not reconstructed per request, so charging
    # Python for building them each iteration would inflate this arm — the
    # dataclass __init__ alone is comparable to everything else in it. Rust does
    # construct its `Ad` inside the call; that asymmetry runs against the Rust
    # arm, which is the direction to err in.
    def score_and_run(ads_and_ctrs):
        scored = [
            ScoredAd(
                ad=ad,
                predicted_ctr=ctr,
                bid_value=round(ctr * ad.target_cpm, 6),
                features=[],
            )
            for ad, ctr in ads_and_ctrs
        ]
        return run_auction(scored, "bench", epsilon, rng)

    inputs = [
        [(StubAd(ad_id, cpm, floor), ctr) for ad_id, cpm, floor, ctr in raw]
        for raw in tuple_sets
    ]

    for i in range(warmup):
        score_and_run(inputs[i % n_sets])

    out = []
    for i in range(iterations):
        batch = inputs[i % n_sets]
        a = perf()
        score_and_run(batch)
        out.append(perf() - a)
    return out


def run_rust_arm(tuple_sets, iterations, epsilon, warmup):
    perf = time.perf_counter_ns
    auction = erised_core.auction
    n_sets = len(tuple_sets)

    for i in range(warmup):
        auction(tuple_sets[i % n_sets], "bench", epsilon, i)

    out = []
    for i in range(iterations):
        raw = tuple_sets[i % n_sets]
        a = perf()
        auction(raw, "bench", epsilon, i)
        out.append(perf() - a)
    return out


def run_marshal_arm(tuple_sets, iterations, warmup):
    perf = time.perf_counter_ns
    marshal = erised_core.marshal_only
    n_sets = len(tuple_sets)

    for i in range(warmup):
        marshal(tuple_sets[i % n_sets])

    out = []
    for i in range(iterations):
        raw = tuple_sets[i % n_sets]
        a = perf()
        marshal(raw)
        out.append(perf() - a)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iterations", type=int, default=50_000)
    ap.add_argument("--warmup", type=int, default=2_000)
    ap.add_argument("--max-ads", type=int, default=40,
                    help="upper bound on candidates per auction; sets vary "
                         "between half this and this")
    ap.add_argument("--sets", type=int, default=512,
                    help="distinct candidate sets cycled through, so the "
                         "measurement is not one input in L1 cache forever")
    ap.add_argument("--epsilon", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the summary as JSON")
    ap.add_argument("--skip-parity", action="store_true",
                    help="time the arms without checking they agree first "
                         "(you should not need this)")
    args = ap.parse_args()

    if not 0.0 <= args.epsilon <= 1.0:
        sys.exit("--epsilon must be between 0 and 1")

    tuple_sets, scored_sets = make_candidate_sets(args.sets, args.max_ads, args.seed)

    if not args.skip_parity:
        assert_parity(tuple_sets, scored_sets)

    overhead = timer_overhead_ns()

    pyf = run_python_full_arm(tuple_sets, args.iterations, args.epsilon,
                              args.warmup, args.seed)
    py3 = run_python_stage3_arm(scored_sets, args.iterations, args.epsilon,
                                args.warmup, args.seed)
    rs = run_rust_arm(tuple_sets, args.iterations, args.epsilon, args.warmup)
    mar = run_marshal_arm(tuple_sets, args.iterations, args.warmup)

    arms = {
        "python_bidvalue_and_auction": summarise(pyf, overhead),
        "python_auction_only": summarise(py3, overhead),
        "rust": summarise(rs, overhead),
        "rust_marshal_only": summarise(mar, overhead),
    }

    py_mean = arms["python_bidvalue_and_auction"]["mean_us_less_timer"]
    py3_mean = arms["python_auction_only"]["mean_us_less_timer"]
    rs_mean = arms["rust"]["mean_us_less_timer"]
    mar_mean = arms["rust_marshal_only"]["mean_us_less_timer"]
    rs_compute = max(rs_mean - mar_mean, 1e-9)

    pyf_arm = arms["python_bidvalue_and_auction"]
    derived = {
        "matched_scope": "bid value construction + auction + second-price clearing",
        "speedup_mean": round(py_mean / max(rs_mean, 1e-9), 2),
        "speedup_p50": round(pyf_arm["p50_us"] / max(arms["rust"]["p50_us"], 1e-9), 2),
        "speedup_p99": round(pyf_arm["p99_us"] / max(arms["rust"]["p99_us"], 1e-9), 2),
        "rust_compute_us_mean": round(rs_compute, 4),
        "speedup_mean_excluding_marshalling": round(py_mean / rs_compute, 2),
        "marshalling_share_of_rust_arm": round(mar_mean / max(rs_mean, 1e-9), 3),
        # Reported so nobody has to derive it and discover it themselves: if the
        # boundary is crossed for stage 3 alone, the FFI cost is not repaid.
        "speedup_mean_auction_only_scope": round(py3_mean / max(rs_mean, 1e-9), 2),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "auction and pricing stage only; not an end-to-end bid request",
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "max_ads": args.max_ads,
            "sets": args.sets,
            "epsilon": args.epsilon,
            "seed": args.seed,
            "parity_checked": not args.skip_parity,
        },
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "timer_overhead_ns": round(overhead, 1),
        },
        "arms": arms,
        "derived": derived,
    }

    w = 30
    print()
    print(f"bid value + auction + pricing, {args.iterations:,} iterations, "
          f"up to {args.max_ads} candidates, epsilon={args.epsilon}")
    print(f"timer floor {overhead:.0f} ns; microseconds per auction")
    print()
    print(f"{'arm':<{w}}{'p50':>9}{'p95':>9}{'p99':>9}{'mean':>9}{'max':>10}")
    print("-" * (w + 46))
    for name, s in arms.items():
        print(f"{name:<{w}}{s['p50_us']:>9.2f}{s['p95_us']:>9.2f}"
              f"{s['p99_us']:>9.2f}{s['mean_us']:>9.2f}{s['max_us']:>10.2f}")
    print()
    print("  matched scope (bid value + auction + pricing), Rust vs Python:")
    print(f"    mean  {derived['speedup_mean']:>6.2f}x     "
          f"p50  {derived['speedup_p50']:>6.2f}x     "
          f"p99  {derived['speedup_p99']:>6.2f}x")
    print()
    print(f"  marshalling is {derived['marshalling_share_of_rust_arm'] * 100:.1f}% of the "
          f"Rust arm; without it the mean speedup would be "
          f"{derived['speedup_mean_excluding_marshalling']:.2f}x (derived)")
    print(f"  against the auction stage alone, Rust is "
          f"{derived['speedup_mean_auction_only_scope']:.2f}x — the boundary is not "
          f"repaid at that scope")
    print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
