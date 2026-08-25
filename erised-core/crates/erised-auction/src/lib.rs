//! Erised matching core — auction stage.
//!
//! A port of stages 2b and 3 of `adplatform/ml/rtb_integration.py`: bid value
//! construction, second-price clearing, and epsilon-greedy exploration.
//!
//! Piece where a mistake shows up on an advertiser's invoice, so it should be auditable in
//! one sitting and buildable offline.

#![deny(missing_debug_implementations)]

mod auction;
mod rng;
mod round;

pub use auction::{clearing_cpm, run_auction, Ad, AuctionResult, ScoredAd, PRICE_TICK_CPM};
pub use rng::{Rng, SplitMix64};
pub use round::py_round;
