# test_auction.py — winner selection, exploration, and serve_propensity.
#
# The propensity tests are the point of this file. Everything else here would
# fail loudly in production; a wrong propensity fails silently, in the training
# job, weeks later, and the symptom is a model that mysteriously stops improving.
#
# The invariant to hold onto: serve_propensity is P(this ad was served), over
# ALL paths that could have served it. The greedy winner has two such paths.

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

import pytest

from adplatform.ml.rtb_integration import (
    AuctionResult,
    ScoredAd,
    run_auction,
)


@dataclass
class FakeAd:
    ad_id: str
    target_cpm: float = 10.0
    floor_price: float = 0.50
    campaign_id: str = "camp_1"
    advertiser_id: str = "adv_1"
    target_keywords: tuple = field(default_factory=tuple)


def make(ad_id: str, ctr: float, target_cpm: float = 10.0,
         floor: float = 0.50) -> ScoredAd:
    ad = FakeAd(ad_id=ad_id, target_cpm=target_cpm, floor_price=floor)
    return ScoredAd(ad=ad, predicted_ctr=ctr,
                    bid_value=round(ctr * target_cpm, 6), features=[0.0])


class ForcedRng:
    """
    Drives both decisions in run_auction deterministically: random() decides
    explore-vs-exploit, choice() decides which ad exploration lands on.
    """

    def __init__(self, explore: bool, choice_index: int = 0):
        self._explore = explore
        self._index = choice_index

    def random(self) -> float:
        # Below any positive epsilon when exploring, at 1.0 when not.
        return 0.0 if self._explore else 1.0

    def choice(self, seq):
        return seq[self._index]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class TestWinnerSelection:

    def test_empty_candidate_list_returns_none(self):
        assert run_auction([], impression_id="imp_1") is None

    def test_highest_bid_value_wins_when_not_exploring(self):
        ads = [make("a", 0.01), make("b", 0.03), make("c", 0.02)]
        result = run_auction(ads, "imp_1", epsilon=0.0)
        assert result.winner.ad.ad_id == "b"

    def test_bid_value_not_ctr_decides(self):
        # Lower CTR, much higher CPM. bid_value = ctr * target_cpm is what the
        # sort uses, and it should be: 0.01*50 beats 0.03*10.
        low_ctr_rich = make("rich", 0.01, target_cpm=50.0)   # 0.50
        high_ctr_poor = make("poor", 0.03, target_cpm=10.0)  # 0.30
        result = run_auction([high_ctr_poor, low_ctr_rich], "imp_1", epsilon=0.0)
        assert result.winner.ad.ad_id == "rich"

    def test_single_candidate_wins_at_its_floor(self):
        result = run_auction([make("solo", 0.02, floor=2.0)], "imp_1", epsilon=0.0)
        assert result.winner.ad.ad_id == "solo"
        assert result.win_price == 2.0

    def test_result_carries_the_impression_id(self):
        result = run_auction([make("a", 0.02)], "imp_xyz", epsilon=0.0)
        assert result.impression_id == "imp_xyz"

    def test_cost_usd_is_win_price_per_thousand(self):
        result = run_auction([make("a", 0.02), make("b", 0.01)], "imp_1", epsilon=0.0)
        assert result.cost_usd == pytest.approx(result.win_price / 1000.0)

    def test_win_price_is_never_negative(self):
        result = run_auction([make("a", 0.02, floor=0.0)], "imp_1", epsilon=0.0)
        assert result.win_price >= 0.0


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

class TestExploration:

    def test_epsilon_zero_never_explores(self):
        ads = [make("a", 0.01), make("b", 0.03)]
        for _ in range(200):
            assert run_auction(ads, "imp_1", epsilon=0.0).is_exploration is False

    def test_epsilon_one_always_explores(self):
        ads = [make("a", 0.01), make("b", 0.03)]
        for _ in range(200):
            assert run_auction(ads, "imp_1", epsilon=1.0).is_exploration is True

    def test_a_single_candidate_never_explores(self):
        # n > 1 guard. With one ad there is nothing to explore toward, and
        # "exploration" would just mean charging the floor for no reason.
        for _ in range(100):
            result = run_auction([make("solo", 0.02)], "imp_1", epsilon=1.0)
            assert result.is_exploration is False
            assert result.serve_propensity == 1.0

    def test_exploration_can_serve_a_loser(self):
        ads = [make("a", 0.01), make("b", 0.03)]
        result = run_auction(ads, "imp_1", epsilon=0.5,
                             rng=ForcedRng(explore=True, choice_index=0))
        assert result.is_exploration is True
        assert result.winner.ad.ad_id == "a"

    def test_exploration_charges_the_floor_not_a_second_price(self):
        # An exploration impression was not won on merit, so there is no
        # runner-up to price against. Charging the floor is the honest answer.
        ads = [make("a", 0.01, floor=1.25), make("b", 0.03, floor=0.10)]
        result = run_auction(ads, "imp_1", epsilon=0.5,
                             rng=ForcedRng(explore=True, choice_index=0))
        assert result.win_price == 1.25

    def test_exploration_rate_is_roughly_epsilon(self):
        ads = [make("a", 0.01), make("b", 0.03), make("c", 0.02)]
        rng = random.Random(20260825)
        explored = sum(
            run_auction(ads, "imp", epsilon=0.20, rng=rng).is_exploration
            for _ in range(20_000)
        )
        assert 0.18 < explored / 20_000 < 0.22


# ---------------------------------------------------------------------------
# Propensity — the reason this file exists
# ---------------------------------------------------------------------------

class TestServePropensity:

    def test_greedy_path_propensity(self):
        # Served by the exploit path: (1-eps) + eps/n, because exploration
        # could ALSO have picked it.
        ads = [make("a", 0.01), make("b", 0.03), make("c", 0.02)]
        result = run_auction(ads, "imp_1", epsilon=0.09,
                             rng=ForcedRng(explore=False))
        assert result.serve_propensity == pytest.approx(0.91 + 0.09 / 3)

    def test_explored_loser_propensity_is_epsilon_over_n(self):
        ads = [make("a", 0.01), make("b", 0.03), make("c", 0.02)]
        # Index 0 is "a", a loser — reachable only via exploration.
        result = run_auction(ads, "imp_1", epsilon=0.09,
                             rng=ForcedRng(explore=True, choice_index=0))
        assert result.winner.ad.ad_id == "a"
        assert result.serve_propensity == pytest.approx(0.09 / 3)

    def test_exploration_landing_on_the_greedy_winner_includes_the_exploit_term(self):
        # THE REGRESSION TEST. This is the bug: the exploration branch used to
        # log eps/n unconditionally, including when rng.choice happened to
        # return the ad the exploit path would have chosen anyway. That ad's
        # true serve probability is (1-eps) + eps/n. Logging eps/n inflates its
        # IPS weight by ~1/eps — at eps=0.08, about 17x — always in favour of
        # the ad the model already ranked first.
        ads = [make("a", 0.01), make("b", 0.03), make("c", 0.02)]
        greedy_index = max(range(len(ads)), key=lambda i: ads[i].bid_value)
        assert ads[greedy_index].ad.ad_id == "b"     # derived, never hardcoded

        result = run_auction(ads, "imp_1", epsilon=0.08,
                             rng=ForcedRng(explore=True, choice_index=greedy_index))

        assert result.winner.ad.ad_id == "b"
        assert result.is_exploration is True
        assert result.serve_propensity == pytest.approx(0.92 + 0.08 / 3), (
            "exploration landed on the greedy winner but logged the "
            "explore-only propensity"
        )

    @pytest.mark.parametrize("epsilon", [0.01, 0.08, 0.25, 0.5])
    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_propensities_sum_to_one_across_the_candidate_set(self, epsilon, n):
        # The definitive check. Sum P(served) over every ad; it must be exactly
        # 1.0, because exactly one ad is served. Any formula error breaks this,
        # including ones no single-case test would catch.
        ads = [make(f"ad_{i}", 0.01 * (i + 1)) for i in range(n)]
        greedy_index = max(range(n), key=lambda i: ads[i].bid_value)

        total = 0.0
        for i in range(n):
            explore_result = run_auction(
                ads, "imp", epsilon=epsilon,
                rng=ForcedRng(explore=True, choice_index=i),
            )
            if i == greedy_index:
                total += explore_result.serve_propensity
            else:
                total += explore_result.serve_propensity

        assert total == pytest.approx(1.0), (
            f"propensities sum to {total:.6f}, not 1.0, at eps={epsilon} n={n}"
        )

    @pytest.mark.parametrize("epsilon", [0.05, 0.20])
    def test_logged_propensity_matches_observed_serve_frequency(self, epsilon):
        # The empirical version: run the real RNG many times and check that each
        # ad's observed serve rate matches the propensity the code logged for
        # it. This catches formula errors AND selection errors at once, and it
        # is what caught the hardcoded-greedy-index bug in the parity harness.
        ads = [make("a", 0.01), make("b", 0.03), make("c", 0.02)]
        rng = random.Random(11)
        trials = 60_000

        served: Counter[str] = Counter()
        logged: dict[str, float] = {}

        for _ in range(trials):
            result = run_auction(ads, "imp", epsilon=epsilon, rng=rng)
            served[result.winner.ad.ad_id] += 1
            logged.setdefault(result.winner.ad.ad_id, result.serve_propensity)

        for ad_id, count in served.items():
            observed = count / trials
            assert observed == pytest.approx(logged[ad_id], abs=0.01), (
                f"{ad_id} served {observed:.4f} of the time but logged "
                f"propensity {logged[ad_id]:.4f}"
            )

    def test_propensity_is_always_a_probability(self):
        ads = [make("a", 0.01), make("b", 0.03), make("c", 0.02)]
        rng = random.Random(7)
        for _ in range(5_000):
            p = run_auction(ads, "imp", epsilon=0.15, rng=rng).serve_propensity
            assert 0.0 < p <= 1.0

    def test_identical_bid_values_do_not_double_count_the_greedy_term(self):
        # Two ads tie exactly. Only the one the sort placed at rank 0 is
        # reachable via the exploit path, so only it gets the (1-eps) term.
        # An `==` comparison on bid_value instead of `is` on the object would
        # give both ads the greedy propensity and sum to more than 1.
        ads = [make("a", 0.02), make("b", 0.02)]
        assert ads[0].bid_value == ads[1].bid_value

        total = sum(
            run_auction(ads, "imp", epsilon=0.10,
                        rng=ForcedRng(explore=True, choice_index=i)).serve_propensity
            for i in range(2)
        )
        assert total == pytest.approx(1.0)


class TestResultShape:

    def test_returns_an_auction_result(self):
        assert isinstance(run_auction([make("a", 0.02)], "imp_1"), AuctionResult)

    def test_carries_the_model_version(self):
        # Without this the training job cannot cut on model_version, which is
        # exactly how you quarantine rows logged under a bad propensity.
        result = run_auction([make("a", 0.02)], "imp_1")
        assert isinstance(result.model_version, str)
        assert result.model_version

    def test_winner_keeps_its_feature_vector(self):
        # build_impression_event logs this verbatim. If it were dropped here,
        # the training job would have to recompute features and train/serve
        # skew becomes possible.
        ads = [make("a", 0.01), make("b", 0.03)]
        result = run_auction(ads, "imp_1", epsilon=0.0)
        assert result.winner.features == [0.0]
