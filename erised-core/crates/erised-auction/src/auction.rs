//! Port of `adplatform/ml/rtb_integration.py` stages 2b and 3.
//!
//! Scope: bid value construction, second-price clearing, and the epsilon-greedy
//! exploration branch. CTR prediction is *not* here — `ScoredAd` takes the
//! predicted CTR as an input, which keeps this crate free of the model and lets
//! it be tested exhaustively without an artifact on disk.
//!
//! Units are the whole difficulty of this file, exactly as in the Python:
//! ranking happens in `bid_value` (a dimensionless CTR x CPM product), billing
//! happens in CPM (dollars per thousand impressions), and `cost_usd` is dollars
//! for this single impression. Mixing any two of those three is the bug class
//! this module exists to prevent.

use std::cmp::Ordering;

use crate::rng::Rng;
use crate::round::py_round;

/// One cent. The winner pays the break-even CPM plus this, so it strictly beats
/// the runner-up rather than exactly tying it.
pub const PRICE_TICK_CPM: f64 = 0.01;

/// Floor on the CTR used as a divisor in `clearing_cpm`. Guards against a
/// division by zero producing an infinite clearing price.
const MIN_DIVISOR_CTR: f64 = 1e-9;

/// The subset of an `Ad` the auction actually reads.
///
/// Targeting fields, creative HTML and budget live upstream in the eligibility
/// filter and never reach here. Keeping this struct narrow means the auction
/// cannot accidentally start depending on something the training-time
/// reconstruction of an ad would not have.
#[derive(Debug, Clone, PartialEq)]
pub struct Ad {
    pub ad_id: String,
    pub target_cpm: f64,
    pub floor_price: f64,
}

/// An eligible ad with its model score attached.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredAd {
    pub ad: Ad,
    pub predicted_ctr: f64,
    /// `py_round(predicted_ctr * target_cpm, 6)`
    pub bid_value: f64,
    /// The exact vector that was scored, carried through so the caller can log
    /// the winner's input verbatim.
    pub features: Vec<f64>,
}

impl ScoredAd {
    /// Mirrors the `ScoredAd(...)` construction inside `score_ads`.
    ///
    /// `bid_value = predicted_ctr * target_cpm` is the line that makes relevance
    /// beat raw budget: an ad bidding $8 at 0.4% CTR loses to one bidding $4 at
    /// 1.2%, because the second is worth more per impression to everyone.
    pub fn new(ad: Ad, predicted_ctr: f64, features: Vec<f64>) -> Self {
        let bid_value = py_round(predicted_ctr * ad.target_cpm, 6);
        Self {
            ad,
            predicted_ctr,
            bid_value,
            features,
        }
    }
}

/// The outcome of one auction.
///
/// The winner is identified by its **index** into the input slice rather than
/// by a cloned `ScoredAd` or by `ad_id`. 
#[derive(Debug, Clone, PartialEq)]
pub struct AuctionResult {
    pub winner_index: usize,
    /// Clearing price in CPM — dollars per THOUSAND impressions.
    pub win_price: f64,
    pub cost_usd: f64,
    pub impression_id: String,
    pub is_exploration: bool,
    /// Probability that the serving policy would have chosen this ad for this
    /// request. 
    pub serve_propensity: f64,
}

/// Convert a second-price result from `bid_value` units back into a CPM.
///
/// Ranking happens on `bid_value = predicted_ctr * cpm`, but billing happens in
/// CPM. Charging the runner-up's `bid_value` directly as though it were already
/// a CPM undercharges by roughly `1/ctr` — a factor of 50 to 100 at realistic
/// click rates.
///
/// The correct clearing price is the lowest CPM at which the winner would still
/// have beaten the runner-up.
pub fn clearing_cpm(winner: &ScoredAd, runner_up_bid_value: f64) -> f64 {
    let ctr = winner.predicted_ctr.max(MIN_DIVISOR_CTR);
    let mut price = (runner_up_bid_value / ctr) + PRICE_TICK_CPM;
    price = price.max(winner.ad.floor_price);
    let price = py_round(price, 6);
    // A misconfigured ad can have floor_price above target_cpm; the
    // advertiser's own maximum always wins that argument.
    price.min(winner.ad.target_cpm)
}

/// The price paid when there is nobody to price against, and the price paid by
/// an ad that won the exploration lottery rather than the auction.
fn floor_cpm(ad: &Ad) -> f64 {
    ad.floor_price.min(ad.target_cpm)
}

fn build_result(
    winner_index: usize,
    cpm: f64,
    impression_id: &str,
    is_exploration: bool,
    serve_propensity: f64,
) -> AuctionResult {
    let cpm = cpm.max(0.0);
    AuctionResult {
        winner_index,
        win_price: cpm,
        cost_usd: cpm / 1000.0,
        impression_id: impression_id.to_owned(),
        is_exploration,
        serve_propensity,
    }
}

/// Probability the policy serves the greedy winner: 
fn greedy_propensity(epsilon: f64, n: usize) -> f64 {
    if n > 1 {
        (1.0 - epsilon) + (epsilon / n as f64)
    } else {
        1.0
    }
}

/// Second-price auction with an epsilon-greedy exploration branch.
///
/// Exploration serves a uniformly random eligible ad at its floor CPM. C
pub fn run_auction<R: Rng>(
    scored_ads: &[ScoredAd],
    impression_id: &str,
    epsilon: f64,
    rng: &mut R,
) -> Option<AuctionResult> {
    if scored_ads.is_empty() {
        return None;
    }
    let n = scored_ads.len();

    // Stable descending sort by bid_value, matching Python's
    // `sorted(..., reverse=True)`: `reverse` does not reverse ties, so equal
    // bids keep their original order.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scored_ads[b]
            .bid_value
            .partial_cmp(&scored_ads[a].bid_value)
            .unwrap_or(Ordering::Equal)
    });
    let greedy = order[0];

    if n > 1 && rng.next_f64() < epsilon {
        let picked = rng.choose(n);
        let propensity = if picked == greedy {
            greedy_propensity(epsilon, n)
        } else {
            epsilon / n as f64
        };
        return Some(build_result(
            picked,
            floor_cpm(&scored_ads[picked].ad),
            impression_id,
            true,
            propensity,
        ));
    }

    let winner = &scored_ads[greedy];
    let cpm = if n >= 2 {
        clearing_cpm(winner, scored_ads[order[1]].bid_value)
    } else {
        floor_cpm(&winner.ad)
    };

    Some(build_result(
        greedy,
        cpm,
        impression_id,
        false,
        greedy_propensity(epsilon, n),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    /// Drives the exploration branch deterministically. `next_f64` returns the
    /// scripted values in order; 
    struct ScriptedRng {
        floats: Vec<f64>,
        choice: usize,
        cursor: usize,
    }

    impl ScriptedRng {
        fn explore_picking(index: usize) -> Self {
            Self {
                floats: vec![0.0],
                choice: index,
                cursor: 0,
            }
        }
        fn never_explores() -> Self {
            Self {
                floats: vec![1.0],
                choice: 0,
                cursor: 0,
            }
        }
    }

    impl Rng for ScriptedRng {
        fn next_f64(&mut self) -> f64 {
            let v = self.floats[self.cursor.min(self.floats.len() - 1)];
            self.cursor += 1;
            v
        }
        fn choose(&mut self, _n: usize) -> usize {
            self.choice
        }
    }

    fn ad(id: &str, target_cpm: f64, floor_price: f64) -> Ad {
        Ad {
            ad_id: id.to_owned(),
            target_cpm,
            floor_price,
        }
    }

    fn scored(id: &str, target_cpm: f64, floor_price: f64, ctr: f64) -> ScoredAd {
        ScoredAd::new(ad(id, target_cpm, floor_price), ctr, vec![])
    }

    // bid value 

    #[test]
    fn bid_value_is_ctr_times_cpm_rounded() {
        let s = scored("a", 5.00, 1.0, 0.02);
        assert_eq!(s.bid_value, 0.1);
    }

    #[test]
    fn relevance_beats_raw_budget() {
        let a = scored("a", 5.00, 1.0, 0.02); // 0.10
        let b = scored("b", 8.00, 1.0, 0.01); // 0.08
        assert!(a.bid_value > b.bid_value);
    }

    // clearing price 

    #[test]
    fn clearing_price_converts_out_of_bid_value_units() {
        let winner = scored("a", 5.00, 1.0, 0.02);
        // Runner-up bid_value 0.08 / winner ctr 0.02 = 4.00, plus a tick.
        assert_eq!(clearing_cpm(&winner, 0.08), 4.01);
    }

    #[test]
    fn clearing_price_leaves_the_winner_still_winning() {
        let winner = scored("a", 5.00, 1.0, 0.02);
        let price = clearing_cpm(&winner, 0.08);
        assert!(price * winner.predicted_ctr > 0.08);
        assert!(price <= winner.ad.target_cpm);
    }

    #[test]
    fn clearing_price_respects_the_floor() {
        let winner = scored("a", 5.00, 2.50, 0.02);
        assert_eq!(clearing_cpm(&winner, 0.01), 2.50);
    }

    #[test]
    fn clearing_price_never_exceeds_target_cpm() {
        // The regression the round-before-cap ordering exists for.
        let winner = scored("a", 1.2973499, 0.0, 0.5);
        let price = clearing_cpm(&winner, 0.9);
        assert!(
            price <= winner.ad.target_cpm,
            "billed {price} against a cap of {}",
            winner.ad.target_cpm
        );
    }

    #[test]
    fn misconfigured_floor_above_target_still_caps_at_target() {
        let winner = scored("a", 2.00, 9.00, 0.02);
        assert_eq!(clearing_cpm(&winner, 0.05), 2.00);
    }

    #[test]
    fn zero_ctr_does_not_produce_an_infinite_price() {
        let winner = scored("a", 5.00, 0.0, 0.0);
        let price = clearing_cpm(&winner, 0.08);
        assert!(price.is_finite());
        assert_eq!(price, 5.00); // clamped to target_cpm
    }

    // auction 

    #[test]
    fn empty_candidate_set_yields_no_fill() {
        let mut rng = SplitMix64::new(1);
        assert!(run_auction(&[], "imp_1", 0.05, &mut rng).is_none());
    }

    #[test]
    fn highest_bid_value_wins() {
        let ads = vec![
            scored("a", 5.00, 1.0, 0.02),
            scored("b", 8.00, 1.0, 0.01),
            scored("c", 4.00, 1.0, 0.005),
        ];
        let mut rng = ScriptedRng::never_explores();
        let r = run_auction(&ads, "imp_1", 0.05, &mut rng).unwrap();
        assert_eq!(r.winner_index, 0);
        assert!(!r.is_exploration);
    }

    #[test]
    fn single_bidder_pays_its_floor() {
        let ads = vec![scored("a", 5.00, 1.25, 0.02)];
        let mut rng = ScriptedRng::never_explores();
        let r = run_auction(&ads, "imp_1", 0.05, &mut rng).unwrap();
        assert_eq!(r.win_price, 1.25);
        assert_eq!(r.serve_propensity, 1.0);
    }

    #[test]
    fn cost_usd_is_cpm_over_a_thousand() {
        let ads = vec![scored("a", 5.00, 1.25, 0.02)];
        let mut rng = ScriptedRng::never_explores();
        let r = run_auction(&ads, "imp_1", 0.05, &mut rng).unwrap();
        assert_eq!(r.cost_usd, r.win_price / 1000.0);
    }

    #[test]
    fn ties_break_toward_the_earlier_candidate() {
        let ads = vec![
            scored("a", 4.00, 1.0, 0.01),
            scored("b", 4.00, 1.0, 0.01), // identical bid_value
        ];
        let mut rng = ScriptedRng::never_explores();
        let r = run_auction(&ads, "imp_1", 0.05, &mut rng).unwrap();
        assert_eq!(r.winner_index, 0, "stable sort must keep input order on ties");
    }

    #[test]
    fn exploration_serves_the_chosen_ad_at_its_floor() {
        let ads = vec![
            scored("a", 5.00, 1.00, 0.02), // greedy winner
            scored("b", 8.00, 2.00, 0.01),
        ];
        let mut rng = ScriptedRng::explore_picking(1);
        let r = run_auction(&ads, "imp_1", 0.05, &mut rng).unwrap();
        assert!(r.is_exploration);
        assert_eq!(r.winner_index, 1);
        assert_eq!(r.win_price, 2.00);
    }

    #[test]
    fn exploring_a_loser_logs_the_exploration_propensity() {
        let ads = vec![
            scored("a", 5.00, 1.00, 0.02),
            scored("b", 8.00, 2.00, 0.01),
        ];
        let mut rng = ScriptedRng::explore_picking(1);
        let r = run_auction(&ads, "imp_1", 0.05, &mut rng).unwrap();
        assert!((r.serve_propensity - 0.05 / 2.0).abs() < 1e-12);
    }

    /// The regression that matters. When exploration lands on the ad the model
    /// would have served anyway, the row would have been logged either way.
    #[test]
    fn exploring_the_greedy_winner_logs_the_greedy_propensity() {
        let ads = vec![
            scored("a", 5.00, 1.00, 0.02), // greedy winner, index 0
            scored("b", 8.00, 2.00, 0.01),
        ];
        let mut rng = ScriptedRng::explore_picking(0);
        let r = run_auction(&ads, "imp_1", 0.05, &mut rng).unwrap();
        assert!(r.is_exploration);
        assert_eq!(r.winner_index, 0);

        let expected = (1.0 - 0.05) + 0.05 / 2.0; // 0.975
        assert!(
            (r.serve_propensity - expected).abs() < 1e-12,
            "propensity {} would give an IPS weight {:.0}x too large",
            r.serve_propensity,
            expected / r.serve_propensity
        );
    }

    #[test]
    fn propensities_sum_to_one_across_the_candidate_set() {
        // Every ad's serve probability, summed, must be exactly the total
        // probability mass. If this fails, IPS weights do not average to 1 and
        // the effective training-set size is wrong.
        let n = 4usize;
        let eps = 0.05;
        let total = greedy_propensity(eps, n) + (n - 1) as f64 * (eps / n as f64);
        assert!((total - 1.0).abs() < 1e-12, "propensities summed to {total}");
    }

    #[test]
    fn single_candidate_never_explores() {
        let ads = vec![scored("a", 5.00, 1.00, 0.02)];
        // Scripted to always take the exploration branch if it is reachable.
        let mut rng = ScriptedRng::explore_picking(0);
        let r = run_auction(&ads, "imp_1", 1.0, &mut rng).unwrap();
        assert!(!r.is_exploration);
    }

    #[test]
    fn exploration_rate_matches_epsilon_over_many_auctions() {
        let ads = vec![
            scored("a", 5.00, 1.00, 0.02),
            scored("b", 8.00, 2.00, 0.01),
            scored("c", 4.00, 0.50, 0.03),
        ];
        let mut rng = SplitMix64::new(20260819);
        let trials = 200_000;
        let explored = (0..trials)
            .filter(|_| {
                run_auction(&ads, "imp", 0.05, &mut rng)
                    .unwrap()
                    .is_exploration
            })
            .count();
        let rate = explored as f64 / trials as f64;
        assert!((rate - 0.05).abs() < 0.002, "exploration rate was {rate}");
    }

    #[test]
    fn price_is_never_negative_and_never_above_the_cap() {
        let mut rng = SplitMix64::new(99);
        for i in 0..10_000u64 {
            let ctr_a = (i % 250) as f64 / 1000.0 + 0.0001;
            let ctr_b = ((i * 7) % 250) as f64 / 1000.0 + 0.0001;
            let ads = vec![
                scored("a", 1.0 + (i % 900) as f64 / 100.0, (i % 300) as f64 / 100.0, ctr_a),
                scored("b", 1.0 + (i % 700) as f64 / 100.0, (i % 200) as f64 / 100.0, ctr_b),
            ];
            let r = run_auction(&ads, "imp", 0.05, &mut rng).unwrap();
            let cap = ads[r.winner_index].ad.target_cpm;
            assert!(r.win_price >= 0.0, "negative price {}", r.win_price);
            assert!(
                r.win_price <= cap,
                "billed {} against a cap of {cap}",
                r.win_price
            );
        }
    }
}
