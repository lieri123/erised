//! Python bindings for the Erised matching core.
//!
//!
//! First, the parity harness needs to feed this the same rows it feeds the
//! Python implementation, and those rows come out of ClickHouse as plain
//! numbers — round-tripping them through a `pyclass` mirror of `rtb.Ad` would
//! mean the test is partly exercising the marshalling rather than the auction.
//!
//! Second, marshalling cost is real. Converting forty `Ad` objects across the
//! FFI boundary per bid request would eat a meaningful share of the latency the
//! port is supposed to buy back. When this graduates from "prove parity" to
//! "serve traffic", the inventory snapshot should live on the Rust side and the
//! boundary should carry ad *ids*, not ad *objects*.

use erised_auction::{clearing_cpm, py_round, run_auction, Ad, ScoredAd, SplitMix64};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Mirrors `adplatform.ml.rtb_integration.AuctionResult`, minus `winner` and
/// `model_version`.
///
/// `winner_index` replaces the `winner` object: the caller already holds the
/// candidate list it passed in, so handing back an index avoids cloning the
/// feature vector back across the boundary. `model_version` is not the
/// auction's business — the Python caller stamps it on.
#[pyclass(get_all, frozen)]
#[derive(Debug, Clone)]
struct PyAuctionResult {
    winner_index: usize,
    win_price: f64,
    cost_usd: f64,
    impression_id: String,
    is_exploration: bool,
    serve_propensity: f64,
}

#[pymethods]
impl PyAuctionResult {
    fn __repr__(&self) -> String {
        format!(
            "AuctionResult(winner_index={}, win_price={}, cost_usd={}, \
             impression_id={:?}, is_exploration={}, serve_propensity={})",
            self.winner_index,
            self.win_price,
            self.cost_usd,
            self.impression_id,
            if self.is_exploration { "True" } else { "False" },
            self.serve_propensity,
        )
    }
}

/// `round(x, ndigits)` with CPython semantics, exposed so the parity harness can
/// fuzz the rounding primitive directly instead of only observing it through
/// clearing prices.
#[pyfunction]
#[pyo3(signature = (x, ndigits = 6))]
fn round6(x: f64, ndigits: usize) -> f64 {
    py_round(x, ndigits)
}

/// `bid_value = round(predicted_ctr * target_cpm, 6)`.
#[pyfunction]
fn bid_value(predicted_ctr: f64, target_cpm: f64) -> f64 {
    ScoredAd::new(
        Ad {
            ad_id: String::new(),
            target_cpm,
            floor_price: 0.0,
        },
        predicted_ctr,
        Vec::new(),
    )
    .bid_value
}

/// Second-price clearing, in CPM.
#[pyfunction]
fn clearing_price(
    predicted_ctr: f64,
    target_cpm: f64,
    floor_price: f64,
    runner_up_bid_value: f64,
) -> f64 {
    let winner = ScoredAd::new(
        Ad {
            ad_id: String::new(),
            target_cpm,
            floor_price,
        },
        predicted_ctr,
        Vec::new(),
    );
    clearing_cpm(&winner, runner_up_bid_value)
}

/// Run one auction.
///
/// `candidates` is a list of `(ad_id, target_cpm, floor_price, predicted_ctr)`
/// in the same order the Python `score_ads` produced them — order is
/// load-bearing, because ties break toward the earlier entry in both
/// implementations.
///
/// `seed` seeds a SplitMix64 rather than reusing Python's `random` module, so
/// exploration decisions will *not* line up with the Python implementation for
/// the same seed. 
///
/// Returns `None` for an empty candidate set, matching the Python `no_fill`
/// path.
#[pyfunction]
#[pyo3(signature = (candidates, impression_id, epsilon = 0.05, seed = 0))]
fn auction(
    candidates: &Bound<'_, PyList>,
    impression_id: &str,
    epsilon: f64,
    seed: u64,
) -> PyResult<Option<PyAuctionResult>> {
    let mut scored = Vec::with_capacity(candidates.len());
    for item in candidates.iter() {
        let (ad_id, target_cpm, floor_price, predicted_ctr): (String, f64, f64, f64) =
            item.extract()?;
        scored.push(ScoredAd::new(
            Ad {
                ad_id,
                target_cpm,
                floor_price,
            },
            predicted_ctr,
            Vec::new(),
        ));
    }

    let mut rng = SplitMix64::new(seed);
    Ok(
        run_auction(&scored, impression_id, epsilon, &mut rng).map(|r| PyAuctionResult {
            winner_index: r.winner_index,
            win_price: r.win_price,
            cost_usd: r.cost_usd,
            impression_id: r.impression_id,
            is_exploration: r.is_exploration,
            serve_propensity: r.serve_propensity,
        }),
    )
}

#[pymodule]
fn erised_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAuctionResult>()?;
    m.add_function(wrap_pyfunction!(round6, m)?)?;
    m.add_function(wrap_pyfunction!(bid_value, m)?)?;
    m.add_function(wrap_pyfunction!(clearing_price, m)?)?;
    m.add_function(wrap_pyfunction!(auction, m)?)?;
    m.add("PRICE_TICK_CPM", erised_auction::PRICE_TICK_CPM)?;
    Ok(())
}
