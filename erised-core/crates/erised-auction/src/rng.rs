//! Randomness for the exploration branch.
//!
//! Bit-for-bit RNG parity to allow for diff exploration decisions
//! between the two implementations, 
//! `random.random()`'s 53-bit assembly and `_randbelow`'s rejection loop — a
//! lot of surface area to maintain for a property no one downstream depends on.
//!


/// Minimal source of randomness for `run_auction`.
pub trait Rng {
    /// Uniform in `[0.0, 1.0)`. Equivalent to `random.random()`.
    fn next_f64(&mut self) -> f64;

    /// Uniform index in `[0, n)`. Equivalent to `random.choice` over a
    /// sequence of length `n`. Panics if `n == 0`.
    fn choose(&mut self, n: usize) -> usize;
}

/// SplitMix64. Small, fast, and reproducible across compiler versions, because auction decisions get replayed in the
//  parity harness.
//  Do not reuse this for the HMAC click
/// signing path.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Rng for SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        // Top 53 bits, scaled. Same construction CPython uses, so the value
        // distribution is identical even though the stream is not.
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    fn choose(&mut self, n: usize) -> usize {
        assert!(n > 0, "choose() called on an empty candidate set");
        let n64 = n as u64;
       
        let bound = (u64::MAX / n64) * n64;
        loop {
            let r = self.next_u64();
            if r < bound {
                return (r % n64) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Rng, SplitMix64};

    #[test]
    fn next_f64_stays_in_unit_interval() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..100_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "out of range: {x}");
        }
    }

    #[test]
    fn choose_covers_every_index() {
        let mut rng = SplitMix64::new(7);
        let n = 5;
        let mut seen = vec![0usize; n];
        for _ in 0..50_000 {
            seen[rng.choose(n)] += 1;
        }

        for (i, count) in seen.iter().enumerate() {
            assert!(*count > 8_000, "index {i} drawn only {count} times");
        }
    }

    #[test]
    fn choose_of_one_is_zero() {
        let mut rng = SplitMix64::new(1);
        assert_eq!(rng.choose(1), 0);
    }

    #[test]
    fn seeding_is_reproducible() {
        let draw = |seed| {
            let mut r = SplitMix64::new(seed);
            (0..8).map(|_| r.next_f64()).collect::<Vec<_>>()
        };
        assert_eq!(draw(42), draw(42));
        assert_ne!(draw(42), draw(43));
    }
}
