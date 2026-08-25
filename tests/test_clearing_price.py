# test_clearing_price.py — what the advertiser is actually billed.
#
# clearing_cpm is four lines and every one of them is load-bearing. The ordering
# in particular: round happens BEFORE the target_cpm cap. Swap those two and
# a price of 1.29735 caps to 1.29735, rounds to 1.2973500000000001 in float, and
# bills above target_cpm — which is the one thing an advertiser is guaranteed
# never happens. That reordering is a one-line "cleanup" someone will make.

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import assume, given, settings as hyp_settings
from hypothesis import strategies as st

from adplatform.ml.rtb_integration import PRICE_TICK_CPM, ScoredAd, clearing_cpm


@dataclass
class FakeAd:
    ad_id: str = "ad_1"
    target_cpm: float = 10.0
    floor_price: float = 0.50


def scored(ctr: float, target_cpm: float = 10.0, floor: float = 0.50) -> ScoredAd:
    ad = FakeAd(target_cpm=target_cpm, floor_price=floor)
    return ScoredAd(ad=ad, predicted_ctr=ctr,
                    bid_value=round(ctr * target_cpm, 6), features=[])


class TestSecondPrice:
    """The economics: you pay what it took to beat the runner-up, not your bid."""

    def test_pays_runner_up_price_not_own_bid(self):
        # Winner would have paid 10.00 CPM at its own bid. The runner-up only
        # forced it to 5.00 + a tick.
        winner = scored(ctr=0.02, target_cpm=10.0)     # bid_value 0.20
        runner_up_bid = 0.10                            # half the winner's
        price = clearing_cpm(winner, runner_up_bid)

        assert price == pytest.approx(5.0 + PRICE_TICK_CPM)
        assert price < winner.ad.target_cpm

    def test_price_rises_with_the_runner_up(self):
        winner = scored(ctr=0.02, target_cpm=10.0)
        cheap = clearing_cpm(winner, 0.05)
        pricey = clearing_cpm(winner, 0.15)
        assert cheap < pricey

    def test_a_higher_ctr_winner_pays_less_for_the_same_rival_bid(self):
        # The bid_value -> CPM conversion divides by the winner's own CTR, so a
        # more relevant ad clears the same competition at a lower CPM. This is
        # the mechanism that makes relevance beat budget; if it inverts, the
        # auction is just a price auction with extra steps.
        low = clearing_cpm(scored(ctr=0.01, target_cpm=10.0), 0.05)
        high = clearing_cpm(scored(ctr=0.04, target_cpm=10.0), 0.05)
        assert high < low


class TestCapAndFloor:

    def test_never_bills_above_target_cpm(self):
        # Runner-up bid so high the raw price would blow past the cap.
        winner = scored(ctr=0.01, target_cpm=2.0)
        assert clearing_cpm(winner, 999.0) == 2.0

    def test_floor_price_is_respected_when_competition_is_weak(self):
        winner = scored(ctr=0.02, target_cpm=10.0, floor=3.0)
        # Runner-up is negligible; without the floor this clears near zero.
        assert clearing_cpm(winner, 0.0) == 3.0

    def test_cap_wins_when_floor_exceeds_target(self):
        # Misconfigured campaign — the CampaignRequest validator rejects this on
        # write, but inventory predating that validator can still hold it, and
        # billing above target_cpm is worse than billing below floor.
        winner = scored(ctr=0.02, target_cpm=1.0, floor=5.0)
        assert clearing_cpm(winner, 0.0) <= 1.0

    def test_rounding_happens_before_the_cap_not_after(self):
        # THE REGRESSION TEST. Pick a target_cpm that a 6dp round can push
        # upward past. If someone reorders to cap-then-round, this fires.
        target = 1.2973499
        winner = scored(ctr=0.01, target_cpm=target, floor=0.0)
        # Drive the raw price well above the cap so the cap is what binds.
        price = clearing_cpm(winner, 0.5)
        assert price <= target, (
            f"billed {price!r} against target_cpm {target!r} — "
            "clearing_cpm is rounding after the cap instead of before"
        )


class TestDegenerateInputs:

    def test_zero_ctr_does_not_divide_by_zero(self):
        # predicted_ctr is clipped to MIN_CTR in the model, but run_auction can
        # be called with hand-built ScoredAds and a crash here 500s a bid.
        price = clearing_cpm(scored(ctr=0.0, target_cpm=10.0), 0.05)
        assert price == 10.0          # huge raw price, capped

    def test_zero_runner_up_still_charges_at_least_the_floor(self):
        assert clearing_cpm(scored(ctr=0.02, floor=0.25), 0.0) == 0.25

    def test_zero_floor_and_zero_competition_clears_at_the_tick(self):
        # Documents, rather than endorses, the $0.00-ish auction. A campaign
        # with floor_price=0.0 and no competition pays essentially nothing.
        price = clearing_cpm(scored(ctr=0.02, floor=0.0), 0.0)
        assert price == pytest.approx(PRICE_TICK_CPM)


class TestProperties:
    """
    The invariant that matters is a boundary, not a formula: whatever the
    inputs, floor <= price <= target_cpm, unless those two are themselves
    contradictory, in which case the cap wins.
    """

    @given(
        ctr=st.floats(min_value=1e-6, max_value=0.25),
        target=st.floats(min_value=0.01, max_value=100.0),
        floor=st.floats(min_value=0.0, max_value=100.0),
        rival=st.floats(min_value=0.0, max_value=50.0),
    )
    @hyp_settings(max_examples=500, deadline=None)
    def test_price_never_exceeds_target_cpm(self, ctr, target, floor, rival):
        winner = scored(ctr=ctr, target_cpm=target, floor=floor)
        assert clearing_cpm(winner, rival) <= target

    @given(
        ctr=st.floats(min_value=1e-6, max_value=0.25),
        target=st.floats(min_value=0.01, max_value=100.0),
        floor=st.floats(min_value=0.0, max_value=100.0),
        rival=st.floats(min_value=0.0, max_value=50.0),
    )
    @hyp_settings(max_examples=500, deadline=None)
    def test_price_meets_floor_whenever_the_floor_is_reachable(
        self, ctr, target, floor, rival
    ):
        # Tolerance is half a tick of the 6dp rounding, and that is not slack —
        # it is the exact bound of a real effect. `round(price, 6)` runs AFTER
        # max(price, floor), so a floor with more than six decimals rounds
        # *below itself*: floor 0.0390625 clears at 0.039062. Hypothesis found
        # this; it is an underbill of 5e-10 dollars per impression, which is
        # not worth reordering money code to fix, but it is worth stating
        # rather than hiding behind a vague epsilon. If this ever fails by more
        # than 5e-7, something structural changed.
        assume(floor <= target)
        winner = scored(ctr=ctr, target_cpm=target, floor=floor)
        assert clearing_cpm(winner, rival) >= floor - 5e-7

    @given(
        ctr=st.floats(min_value=1e-4, max_value=0.25),
        rival_a=st.floats(min_value=0.0, max_value=10.0),
        rival_b=st.floats(min_value=0.0, max_value=10.0),
    )
    @hyp_settings(max_examples=300, deadline=None)
    def test_monotone_in_the_runner_up_bid(self, ctr, rival_a, rival_b):
        # A stronger rival never makes you pay less. Non-strict because the cap
        # and the floor both flatten the curve at the ends.
        assume(rival_a <= rival_b)
        winner = scored(ctr=ctr, target_cpm=1000.0, floor=0.0)
        assert clearing_cpm(winner, rival_a) <= clearing_cpm(winner, rival_b) + 1e-9

    @given(ctr=st.floats(min_value=1e-6, max_value=0.25),
           rival=st.floats(min_value=0.0, max_value=50.0))
    @hyp_settings(max_examples=300, deadline=None)
    def test_price_is_never_negative(self, ctr, rival):
        assert clearing_cpm(scored(ctr=ctr, floor=0.0), rival) >= 0.0
