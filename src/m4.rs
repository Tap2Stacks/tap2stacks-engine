//! Odds Model M4: direct Monte Carlo simulation of the trade.xyz Hyperliquid mark-price
//! formula (`docs/odds-model-v2-m4-direction-and-cap.md` §0b).
//!
//! All state quantities evolve in cents on the 3-second oracle-update clock (using the
//! causal phase and regime from `crate::v2::RegimeDetector`):
//! - Book `B` (`midPx`): `dB[n] = phi * dB[n-1] + eta`, `eta ~ Normal(0, s_eta)`.
//! - Oracle `O` (`oraclePx`): `dO[n+1] = a * dB[n] + lam * (B[n] - O[n] - E[n]) + J`,
//!   where `J = +/- K` with probability `pj` and `K = 1 + Geom_0(geom_p)` on `{1, 2, ...}`.
//! - Basis `D[n] = B[n] - O[n]` and 150 s continuous-time EMA
//!   `E[n] = E[n-1] + (1 - exp(-3 / 150)) * (D[n] - E[n-1])`.
//! - Mark `X[n] = round_to_cent(O[n] + median(0, E[n], D[n])) = round_to_cent(median(O, O + E, B))`.
//! - Settlement matches v2 Method B: a cell `[lc, uc]` in window `[ws, we)` hits iff any
//!   segment `[min(X_{s-1}, X_s), max(X_{s-1}, X_s)]` for `s in ws..we` intersects `[lc, uc]`.
//!
//! All placeholder preset parameters below come from `logs/calibration/m4-trend/mark-model-sim3.txt`
//! with M3 per-preset `m_base` values. `price_grid_m4` applies the current regime's
//! `MarkFormulaParams` across every quoted window.

use std::f64::consts::TAU;

use crate::grid::{PriceBand, TimeWindow};
use crate::price::PriceTick;

/// Local Hyperliquid 3-second oracle-update clock pinned for M4 (trade.xyz mark formula only).
const HL_ORACLE_CLOCK_SECS: u64 = 3;
/// Continuous-time EMA time constant in seconds for the trade.xyz basis EMA `E`.
pub const MARK_EMA_TAU_SECS: f64 = 150.0;
/// Data gap threshold (milliseconds) above which `MarkFormulaTracker` resets `E` and `dB`.
pub const MARK_TRACKER_RESET_GAP_MS: u64 = 30_000;
/// Minimum elapsed time (ms) between distinct oracle-update book snapshots when triggered by
/// an observed `oraclePx`/`markPx` change (prevents sub-second duplicate frames from zeroing `dB`).
const UPDATE_MIN_INTERVAL_MS: u64 = 2_000;
/// Maximum elapsed time (ms) before sampling `B` on the 3-second oracle clock even if `O`/`X` stay flat.
const UPDATE_MAX_INTERVAL_MS: u64 = HL_ORACLE_CLOCK_SECS * 1_000;

/// Exact branchless median of three finite floats.
#[inline]
pub fn median3(a: f64, b: f64, c: f64) -> f64 {
    a.max(b.min(c)).min(b.max(c))
}

/// Computes the rounded integer-cent mark price `X = round(O + median(0, E, D))`,
/// which is identically `round(median(O, O + E, B))` since `D = B - O`.
#[inline]
pub fn mark_from_basis(oracle_cents: f64, ema_basis_cents: f64, basis_cents: f64) -> i64 {
    (oracle_cents + median3(0.0, ema_basis_cents, basis_cents)).round() as i64
}

/// Per-regime parameters for the M4 trade.xyz mark-formula simulator (`s_eta` in cents per 3 s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarkFormulaParams {
    /// Book AR(1) persistence on 3-second book moves `dB`.
    pub phi: f64,
    /// Book innovation std dev `s_eta` in cents per 3-second update.
    pub s_eta: f64,
    /// Lag-1 response of next oracle move `dO[n+1]` to prior book move `dB[n]`.
    pub a: f64,
    /// Pull of next oracle move `dO[n+1]` toward `B[n] - O[n] - E[n]`.
    pub lam: f64,
    /// Probability of an independent oracle own jump `J != 0` on a 3-second update.
    pub pj: f64,
    /// Geometric parameter `p` for `K = 1 + Geom_0(p)` on `{1, 2, ...}` (`E[K] = 1 / p` cents).
    pub geom_p: f64,
    /// Per-preset base margin (initialized from M3 `m_base`).
    pub m_base: f64,
}

/// Complete M4 parameter table for one ticker, ordered `[Quiet, Normal, Volatile, News]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickerM4Params {
    pub ticker: &'static str,
    pub presets: [MarkFormulaParams; 4],
}

/// NVDA M4 placeholder parameters from `logs/calibration/m4-trend/mark-model-sim3.txt`
/// with M3 `m_base` values.
pub static NVDA_M4_PARAMS: TickerM4Params = TickerM4Params {
    ticker: "NVDA",
    presets: [
        // Quiet
        MarkFormulaParams {
            phi: 0.139,
            s_eta: 2.24,
            a: 0.195,
            lam: 0.008,
            pj: 0.32,
            geom_p: 0.565,
            m_base: 0.025,
        },
        // Normal
        MarkFormulaParams {
            phi: 0.037,
            s_eta: 3.63,
            a: 0.436,
            lam: 0.056,
            pj: 0.76,
            geom_p: 0.461,
            m_base: 0.098,
        },
        // Volatile
        MarkFormulaParams {
            phi: 0.054,
            s_eta: 7.26,
            a: 0.451,
            lam: 0.150,
            pj: 0.87,
            geom_p: 0.280,
            m_base: 0.134,
        },
        // News
        MarkFormulaParams {
            phi: 0.143,
            s_eta: 20.40,
            a: 0.296,
            lam: 0.222,
            pj: 0.95,
            geom_p: 0.068,
            m_base: 0.263,
        },
    ],
};

/// TSLA M4 placeholder parameters from `logs/calibration/m4-trend/mark-model-sim3.txt`
/// with M3 `m_base` values.
pub static TSLA_M4_PARAMS: TickerM4Params = TickerM4Params {
    ticker: "TSLA",
    presets: [
        // Quiet
        MarkFormulaParams {
            phi: 0.087,
            s_eta: 2.80,
            a: 0.190,
            lam: 0.021,
            pj: 0.31,
            geom_p: 0.329,
            m_base: 0.025,
        },
        // Normal
        MarkFormulaParams {
            phi: 0.048,
            s_eta: 5.72,
            a: 0.353,
            lam: 0.150,
            pj: 0.86,
            geom_p: 0.241,
            m_base: 0.073,
        },
        // Volatile
        MarkFormulaParams {
            phi: 0.068,
            s_eta: 10.44,
            a: 0.358,
            lam: 0.283,
            pj: 0.92,
            geom_p: 0.164,
            m_base: 0.197,
        },
        // News
        MarkFormulaParams {
            phi: 0.095,
            s_eta: 26.05,
            a: 0.311,
            lam: 0.268,
            pj: 0.96,
            geom_p: 0.073,
            m_base: 0.185,
        },
    ],
};

/// Returns M4 parameters for tickers with fitted presets (`NVDA`, `TSLA`).
/// Other tickers (`AAPL`, `SP500`) return `None` and stay on v1.
pub fn m4_ticker_params(ticker: &str) -> Option<&'static TickerM4Params> {
    let symbol = ticker.split(':').next_back().unwrap_or(ticker);
    if symbol.eq_ignore_ascii_case("NVDA") {
        Some(&NVDA_M4_PARAMS)
    } else if symbol.eq_ignore_ascii_case("TSLA") {
        Some(&TSLA_M4_PARAMS)
    } else {
        None
    }
}

/// Snapshot of live causal state (all prices/basis in cents) used to seed M4 Monte Carlo paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarkFormulaState {
    pub oracle_cents: f64,
    pub book_cents: f64,
    pub ema_basis_cents: f64,
    pub last_db_cents: f64,
    pub mark_cents: i64,
}

/// Strictly causal per-ticker tracker for the 150 s continuous-time basis EMA `E` and the
/// 3-second update book move `dB`.
#[derive(Debug, Clone, Default)]
pub struct MarkFormulaTracker {
    last_ts_ms: Option<u64>,
    last_oracle_cents: Option<f64>,
    last_book_cents: Option<f64>,
    last_mark_cents: Option<i64>,
    ema_basis_cents: Option<f64>,
    prev_update_book_cents: Option<f64>,
    curr_update_book_cents: Option<f64>,
    last_update_ts_ms: Option<u64>,
    last_db_cents: f64,
}

impl MarkFormulaTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.last_oracle_cents = None;
        self.last_book_cents = None;
        self.last_mark_cents = None;
        self.ema_basis_cents = None;
        self.prev_update_book_cents = None;
        self.curr_update_book_cents = None;
        self.last_update_ts_ms = None;
        self.last_db_cents = 0.0;
    }

    pub fn ema_basis_cents(&self) -> Option<f64> {
        self.ema_basis_cents
    }

    pub fn last_db_cents(&self) -> f64 {
        self.last_db_cents
    }

    /// Feeds one raw tick causally into the tracker. Idempotent for repeated/out-of-order
    /// timestamps (`tick.ts_ms <= last_ts_ms`).
    pub fn on_tick(&mut self, tick: PriceTick) {
        if !tick.price.is_finite() || tick.price <= 0.0 {
            return;
        }
        let prev_ts_ms = self.last_ts_ms;
        if prev_ts_ms.is_some_and(|prev| tick.ts_ms <= prev) {
            return;
        }

        let dt_ms = prev_ts_ms.map(|prev| tick.ts_ms.saturating_sub(prev));
        if dt_ms.is_some_and(|dt| dt > MARK_TRACKER_RESET_GAP_MS) {
            self.reset();
        }

        let mark_c = (tick.price * 100.0).round() as i64;
        let prev_o = self.last_oracle_cents;
        let prev_m = self.last_mark_cents;

        let o_c = tick
            .oracle_px
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| v * 100.0)
            .or(self.last_oracle_cents)
            .unwrap_or(tick.price * 100.0);
        let b_c = tick
            .mid_px
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| v * 100.0)
            .or(self.last_book_cents)
            .unwrap_or(tick.price * 100.0);
        let d_c = b_c - o_c;

        // Update 150 s continuous-time EMA of D = B - O at every record (init E = D).
        match (self.ema_basis_cents, dt_ms) {
            (Some(prev_e), Some(dt)) if dt <= MARK_TRACKER_RESET_GAP_MS => {
                let dt_secs = dt as f64 / 1_000.0;
                let alpha = 1.0 - (-dt_secs / MARK_EMA_TAU_SECS).exp();
                self.ema_basis_cents = Some(prev_e + alpha * (d_c - prev_e));
            }
            _ => {
                self.ema_basis_cents = Some(d_c);
            }
        }

        // Track B at the last two oracle updates to compute 3-second dB.
        let is_update = match self.last_update_ts_ms {
            None => true,
            Some(last_upd) => {
                let elapsed = tick.ts_ms.saturating_sub(last_upd);
                let oracle_or_mark_moved = prev_o.is_some_and(|po| (o_c - po).abs() > 1e-9)
                    || prev_m.is_some_and(|pm| mark_c != pm);
                elapsed >= UPDATE_MAX_INTERVAL_MS
                    || (elapsed >= UPDATE_MIN_INTERVAL_MS && oracle_or_mark_moved)
            }
        };
        if is_update {
            self.prev_update_book_cents = self.curr_update_book_cents;
            self.curr_update_book_cents = Some(b_c);
            self.last_update_ts_ms = Some(tick.ts_ms);
            if let Some(prev_b) = self.prev_update_book_cents {
                self.last_db_cents = b_c - prev_b;
            } else {
                self.last_db_cents = 0.0;
            }
        }

        self.last_ts_ms = Some(tick.ts_ms);
        self.last_oracle_cents = Some(o_c);
        self.last_book_cents = Some(b_c);
        self.last_mark_cents = Some(mark_c);
    }

    /// Returns the current state snapshot, falling back to `fallback_spot` if uninitialized.
    pub fn state(&self, fallback_spot: f64) -> MarkFormulaState {
        let spot_c = fallback_spot * 100.0;
        let oracle_cents = self.last_oracle_cents.unwrap_or(spot_c);
        let book_cents = self.last_book_cents.unwrap_or(spot_c);
        let ema_basis_cents = self
            .ema_basis_cents
            .unwrap_or(book_cents - oracle_cents);
        let mark_cents = self
            .last_mark_cents
            .unwrap_or_else(|| spot_c.round() as i64);
        MarkFormulaState {
            oracle_cents,
            book_cents,
            ema_basis_cents,
            last_db_cents: self.last_db_cents,
            mark_cents,
        }
    }
}

/// Fast deterministic Xoshiro256++ PRNG with paired Box-Muller standard normals.
struct FastRng {
    s: [u64; 4],
    spare_normal: Option<f64>,
}

impl FastRng {
    fn from_seed(mut seed: u64) -> Self {
        let mut next_splitmix = || -> u64 {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            s: [
                next_splitmix(),
                next_splitmix(),
                next_splitmix(),
                next_splitmix(),
            ],
            spare_normal: None,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform float in `(0, 1)`.
    #[inline]
    fn next_open01(&mut self) -> f64 {
        // Use top 53 bits shifted by 0.5 ULP so result is strictly in (0, 1).
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn next_normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        let u1 = self.next_open01();
        let u2 = self.next_open01();
        let r = (-2.0 * u1.ln()).sqrt();
        let (sin_t, cos_t) = (TAU * u2).sin_cos();
        self.spare_normal = Some(r * sin_t);
        r * cos_t
    }
}

/// Advances `(o, b, e, db, x)` by one 3-second oracle update according to the M4 mark formula.
#[inline]
fn step_mark_formula(
    o: &mut f64,
    b: &mut f64,
    e: &mut f64,
    db: &mut f64,
    x: &mut i64,
    params: &MarkFormulaParams,
    alpha_3s: f64,
    inv_ln_1mp: f64,
    rng: &mut FastRng,
) {
    // 1. Oracle jump J = +/- K with prob pj, K = 1 + Geom_0(p) on {1, 2, ...}.
    let j = if params.pj > 0.0 && rng.next_open01() < params.pj {
        let k = if params.geom_p >= 1.0 {
            1.0
        } else {
            1.0 + (rng.next_open01().ln() * inv_ln_1mp).floor()
        };
        if (rng.next_u64() & 1) == 0 { k } else { -k }
    } else {
        0.0
    };

    // 2. Oracle update using previous dB[n] and previous B[n] - O[n] - E[n].
    let d_o = params.a * *db + params.lam * (*b - *o - *e) + j;
    *o += d_o;

    // 3. Book AR(1) update: dB[n+1] = phi * dB[n] + eta, eta ~ Normal(0, s_eta).
    let eta = params.s_eta * rng.next_normal();
    let next_db = params.phi * *db + eta;
    *db = next_db;
    *b += next_db;

    // 4. Basis D[n+1] = B[n+1] - O[n+1] and 150 s EMA update.
    let d_new = *b - *o;
    *e += alpha_3s * (d_new - *e);

    // 5. Mark price X[n+1] = round_to_cent(O[n+1] + median(0, E[n+1], D[n+1])).
    *x = mark_from_basis(*o, *e, d_new);
}

/// Checks whether integer-cent bands form a contiguous uniform ladder `[base + r*B, base + (r+1)*B]`.
fn uniform_band_ladder(row_bounds: &[(i64, i64)]) -> Option<(i64, i64)> {
    let &(first_lo, first_hi) = row_bounds.first()?;
    let band_c = first_hi - first_lo;
    if band_c <= 0 {
        return None;
    }
    for (idx, &(lo, hi)) in row_bounds.iter().enumerate() {
        let expected_lo = first_lo + (idx as i64) * band_c;
        if lo != expected_lo || hi != expected_lo + band_c {
            return None;
        }
    }
    Some((first_lo, band_c))
}

/// Simulates exact Method-B touch probabilities `[row][window]` under the M4 trade.xyz
/// mark-price formula via Monte Carlo from the live causal state `(O, B, E, dB, X)`.
///
/// Uses the current regime's `MarkFormulaParams` for every quoted window (near and far).
/// When `inferred_phase` is `Some(phi)`, every path uses `phi`; when `None`, paths are
/// stratified evenly across `phi in {0, 1, 2}`.
pub fn price_grid_m4(
    bands: &[PriceBand],
    windows: &[TimeWindow],
    state: MarkFormulaState,
    params: &MarkFormulaParams,
    inferred_phase: Option<usize>,
    mc_n: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let rows = bands.len();
    let w_count = windows.len();
    let mut probs = vec![vec![0.0_f64; w_count]; rows];
    if rows == 0 || w_count == 0 || mc_n == 0 {
        return probs;
    }

    let row_bounds: Vec<(i64, i64)> = bands
        .iter()
        .map(|band| {
            let lc = (band.lower * 100.0).round() as i64;
            let uc = (band.upper * 100.0).round() as i64;
            (lc, uc)
        })
        .collect();
    let ladder = uniform_band_ladder(&row_bounds);

    let alpha_3s = 1.0 - (-(HL_ORACLE_CLOCK_SECS as f64) / MARK_EMA_TAU_SECS).exp();
    let inv_ln_1mp = if params.geom_p < 1.0 && params.geom_p > 0.0 {
        1.0 / (1.0 - params.geom_p).ln()
    } else {
        0.0
    };

    let monotone = windows
        .windows(2)
        .all(|pair| pair[0].end_secs <= pair[1].start_secs)
        && windows.iter().all(|w| w.start_secs <= w.end_secs);

    let mut rng = FastRng::from_seed(seed);

    if monotone {
        // Precompute oracle-tick counts before and inside each window for each phase in 0..3.
        let mut schedule: [Vec<(usize, usize)>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(w_count));
        for ph in 0..3usize {
            let mut cursor: usize = 1;
            for w in windows {
                let ws = w.start_secs as usize;
                let we = w.end_secs as usize;
                let pre = if ws > cursor {
                    let c = (cursor..ws)
                        .filter(|&s| (s + HL_ORACLE_CLOCK_SECS as usize - ph) % HL_ORACLE_CLOCK_SECS as usize == 0)
                        .count();
                    cursor = ws;
                    c
                } else {
                    0
                };
                let inside = if we > cursor {
                    let c = (cursor..we)
                        .filter(|&s| (s + HL_ORACLE_CLOCK_SECS as usize - ph) % HL_ORACLE_CLOCK_SECS as usize == 0)
                        .count();
                    cursor = we;
                    c
                } else {
                    0
                };
                schedule[ph].push((pre, inside));
            }
        }

        if let Some((base_lc, band_c)) = ladder {
            let mut hits_diff = vec![vec![0_i64; rows + 1]; w_count];
            let rows_i64 = rows as i64;
            for path_idx in 0..mc_n {
                let ph = inferred_phase.map(|p| p % 3).unwrap_or(path_idx % 3);
                let sched = &schedule[ph];
                let mut o = state.oracle_cents;
                let mut b = state.book_cents;
                let mut e = state.ema_basis_cents;
                let mut db = state.last_db_cents;
                let mut x = state.mark_cents;

                for (w_idx, &(pre, inside)) in sched.iter().enumerate() {
                    for _ in 0..pre {
                        step_mark_formula(
                            &mut o,
                            &mut b,
                            &mut e,
                            &mut db,
                            &mut x,
                            params,
                            alpha_3s,
                            inv_ln_1mp,
                            &mut rng,
                        );
                    }
                    let mut win_min = x;
                    let mut win_max = x;
                    for _ in 0..inside {
                        step_mark_formula(
                            &mut o,
                            &mut b,
                            &mut e,
                            &mut db,
                            &mut x,
                            params,
                            alpha_3s,
                            inv_ln_1mp,
                            &mut rng,
                        );
                        if x < win_min {
                            win_min = x;
                        }
                        if x > win_max {
                            win_max = x;
                        }
                    }
                    // Row r = [base + r*B, base + (r+1)*B] hits iff base + (r+1)*B >= win_min
                    // and base + r*B <= win_max.
                    let r_lo = (win_min - base_lc - 1).div_euclid(band_c).clamp(0, rows_i64) as usize;
                    let r_hi = (win_max - base_lc).div_euclid(band_c).min(rows_i64 - 1);
                    if (r_lo as i64) <= r_hi {
                        let diff = &mut hits_diff[w_idx];
                        diff[r_lo] += 1;
                        diff[(r_hi as usize) + 1] -= 1;
                    }
                }
            }
            let inv_n = 1.0 / mc_n as f64;
            for w_idx in 0..w_count {
                let mut running = 0_i64;
                for row in 0..rows {
                    running += hits_diff[w_idx][row];
                    probs[row][w_idx] = (running as f64 * inv_n).clamp(0.0, 1.0);
                }
            }
            return probs;
        }

        // General non-uniform bands with monotone windows.
        let mut hits = vec![vec![0usize; w_count]; rows];
        for path_idx in 0..mc_n {
            let ph = inferred_phase.map(|p| p % 3).unwrap_or(path_idx % 3);
            let sched = &schedule[ph];
            let mut o = state.oracle_cents;
            let mut b = state.book_cents;
            let mut e = state.ema_basis_cents;
            let mut db = state.last_db_cents;
            let mut x = state.mark_cents;

            for (w_idx, &(pre, inside)) in sched.iter().enumerate() {
                for _ in 0..pre {
                    step_mark_formula(
                        &mut o,
                        &mut b,
                        &mut e,
                        &mut db,
                        &mut x,
                        params,
                        alpha_3s,
                        inv_ln_1mp,
                        &mut rng,
                    );
                }
                let mut win_min = x;
                let mut win_max = x;
                for _ in 0..inside {
                    step_mark_formula(
                        &mut o,
                        &mut b,
                        &mut e,
                        &mut db,
                        &mut x,
                        params,
                        alpha_3s,
                        inv_ln_1mp,
                        &mut rng,
                    );
                    if x < win_min {
                        win_min = x;
                    }
                    if x > win_max {
                        win_max = x;
                    }
                }
                for (row, &(lc, uc)) in row_bounds.iter().enumerate() {
                    if win_min <= uc && win_max >= lc {
                        hits[row][w_idx] += 1;
                    }
                }
            }
        }
        let inv_n = 1.0 / mc_n as f64;
        for row in 0..rows {
            for w_idx in 0..w_count {
                probs[row][w_idx] = (hits[row][w_idx] as f64 * inv_n).clamp(0.0, 1.0);
            }
        }
        return probs;
    }

    // Fallback for overlapping or unsorted windows.
    let max_end = windows.iter().map(|w| w.end_secs as usize).max().unwrap_or(0);
    let mut x_path = vec![0_i64; max_end + 1];
    let mut hits = vec![vec![0usize; w_count]; rows];
    for path_idx in 0..mc_n {
        let ph = inferred_phase.map(|p| p % 3).unwrap_or(path_idx % 3);
        let mut o = state.oracle_cents;
        let mut b = state.book_cents;
        let mut e = state.ema_basis_cents;
        let mut db = state.last_db_cents;
        let mut x = state.mark_cents;
        x_path[0] = x;
        for s in 1..max_end {
            if (s + HL_ORACLE_CLOCK_SECS as usize - ph) % HL_ORACLE_CLOCK_SECS as usize == 0 {
                step_mark_formula(
                    &mut o,
                    &mut b,
                    &mut e,
                    &mut db,
                    &mut x,
                    params,
                    alpha_3s,
                    inv_ln_1mp,
                    &mut rng,
                );
            }
            x_path[s] = x;
        }
        for (w_idx, w) in windows.iter().enumerate() {
            let ws = w.start_secs as usize;
            let we = w.end_secs as usize;
            if ws >= we {
                continue;
            }
            let first = ws.saturating_sub(1);
            let mut win_min = x_path[first];
            let mut win_max = x_path[first];
            for &val in &x_path[ws..we] {
                if val < win_min {
                    win_min = val;
                }
                if val > win_max {
                    win_max = val;
                }
            }
            for (row, &(lc, uc)) in row_bounds.iter().enumerate() {
                if win_min <= uc && win_max >= lc {
                    hits[row][w_idx] += 1;
                }
            }
        }
    }
    let inv_n = 1.0 / mc_n as f64;
    for row in 0..rows {
        for w_idx in 0..w_count {
            probs[row][w_idx] = (hits[row][w_idx] as f64 * inv_n).clamp(0.0, 1.0);
        }
    }
    probs
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, StandardNormal};

    use super::*;
    use crate::v2::Regime;

    #[test]
    fn median_identity_holds_across_all_orderings() {
        let cases = [
            (14_588.0_f64, 5.0_f64, 14_600.5_f64),   // 0 < E < D
            (14_588.0, 20.0, 14_600.5),              // 0 < D < E
            (14_588.0, -4.0, 14_600.5),              // E < 0 < D
            (14_588.0, 4.0, 14_575.0),               // D < 0 < E
            (14_588.0, -2.0, 14_575.0),              // D < E < 0
            (14_588.0, -20.0, 14_575.0),             // E < D < 0
            (14_588.0, 0.0, 14_588.0),               // all equal
            (37_927.0, 7.5, 37_934.5),               // E == D
            (37_927.0, 0.0, 37_937.5),               // E == 0
        ];
        for (o, e, b) in cases {
            let d = b - o;
            let lhs = median3(o, o + e, b);
            let rhs = o + median3(0.0, e, d);
            assert!(
                (lhs - rhs).abs() < 1e-12,
                "O={o} E={e} B={b}: median(O,O+E,B)={lhs} != O+median(0,E,D)={rhs}"
            );
            assert_eq!(lhs.round() as i64, mark_from_basis(o, e, d));
        }
    }

    #[test]
    fn continuous_time_ema_matches_closed_form_and_resets_after_30s_gap() {
        let mut tracker = MarkFormulaTracker::new();

        // Init at t = 0 with D0 = B - O = +10c => E(0) = +10c.
        tracker.on_tick(PriceTick {
            ts_ms: 0,
            price: 140.00,
            mid_px: Some(140.10),
            oracle_px: Some(140.00),
        });
        assert!((tracker.ema_basis_cents().unwrap() - 10.0).abs() < 1e-12);
        assert_eq!(tracker.last_db_cents(), 0.0);

        // Step basis to constant D1 = +22c and feed ticks at irregular intervals summing to T = 45 s.
        // Continuous-time EMA must satisfy E(T) = D1 + (D0 - D1) * exp(-T / 150) identically.
        let dts_secs = [1u64, 2, 3, 1, 5, 3, 7, 2, 6, 4, 11];
        let mut t_secs = 0u64;
        for dt in dts_secs {
            t_secs += dt;
            tracker.on_tick(PriceTick {
                ts_ms: t_secs * 1_000,
                price: 140.10,
                mid_px: Some(140.22),
                oracle_px: Some(140.00),
            });
        }
        assert_eq!(t_secs, 45);
        let expected_e = 22.0 + (10.0 - 22.0) * (-45.0_f64 / 150.0).exp();
        assert!(
            (tracker.ema_basis_cents().unwrap() - expected_e).abs() < 1e-11,
            "got {:.12}, expected {:.12}",
            tracker.ema_basis_cents().unwrap(),
            expected_e
        );

        // Track B across two 3s updates: move mid by +4c at t = 48s.
        tracker.on_tick(PriceTick {
            ts_ms: 48_000,
            price: 140.14,
            mid_px: Some(140.26),
            oracle_px: Some(140.04),
        });
        assert!((tracker.last_db_cents() - 4.0).abs() < 1e-9);

        // 20s gap (<= 30s) must NOT reset E.
        let e_before_20s = tracker.ema_basis_cents().unwrap();
        tracker.on_tick(PriceTick {
            ts_ms: 68_000,
            price: 140.14,
            mid_px: Some(140.26),
            oracle_px: Some(140.04),
        });
        let expected_after_20s =
            22.0 + (e_before_20s - 22.0) * (-20.0_f64 / 150.0).exp();
        assert!((tracker.ema_basis_cents().unwrap() - expected_after_20s).abs() < 1e-11);

        // 31s gap (> 30s) MUST reset E = D_new and dB = 0.
        tracker.on_tick(PriceTick {
            ts_ms: 99_000,
            price: 141.00,
            mid_px: Some(140.95),
            oracle_px: Some(141.00),
        });
        // (140.95 - 141.00) * 100 is -4.999999999998 in f64 (dollar-to-cent rounding of
        // two ~141.0 values), so compare at 1e-9 cents: far below one cent, far above f64 noise.
        assert!((tracker.ema_basis_cents().unwrap() + 5.0).abs() < 1e-9);
        assert_eq!(tracker.last_db_cents(), 0.0);
    }

    #[test]
    fn degenerate_zero_params_keeps_mark_fixed_on_current_row() {
        let params = MarkFormulaParams {
            phi: 0.0,
            s_eta: 1e-9,
            a: 0.0,
            lam: 0.0,
            pj: 0.0,
            geom_p: 1.0,
            m_base: 0.025,
        };
        let state = MarkFormulaState {
            oracle_cents: 14_000.0,
            book_cents: 14_000.0,
            ema_basis_cents: 0.0,
            last_db_cents: 0.0,
            mark_cents: 14_000,
        };
        // Contiguous 3c ladder where row 2 = [139.99, 140.02] strictly contains 140.00.
        let bands: Vec<PriceBand> = (0..5)
            .map(|r| PriceBand {
                row_index: r,
                lower: 139.93 + 0.03 * r as f64,
                upper: 139.96 + 0.03 * r as f64,
            })
            .collect();
        let windows: Vec<TimeWindow> = (0..24)
            .map(|w| TimeWindow {
                start_secs: ((w + 1) * 5) as u32,
                end_secs: ((w + 2) * 5) as u32,
            })
            .collect();

        let probs = price_grid_m4(&bands, &windows, state, &params, None, 2_000, 42);
        for w in 0..24 {
            for r in 0..5 {
                let expected = if r == 2 { 1.0 } else { 0.0 };
                assert!(
                    (probs[r][w] - expected).abs() < 1e-12,
                    "row={r} w={w}: got {}, expected {expected}",
                    probs[r][w]
                );
            }
        }
    }

    /// Independent plain second-by-second reference simulator using `StdRng` + `StandardNormal`.
    #[allow(clippy::too_many_arguments)]
    fn reference_mc_m4(
        bands: &[PriceBand],
        windows: &[TimeWindow],
        state: MarkFormulaState,
        params: &MarkFormulaParams,
        inferred_phase: Option<usize>,
        n_paths: usize,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        let rows = bands.len();
        let w_count = windows.len();
        let max_end = windows.iter().map(|w| w.end_secs as usize).max().unwrap_or(0);
        let bounds: Vec<(i64, i64)> = bands
            .iter()
            .map(|b| ((b.lower * 100.0).round() as i64, (b.upper * 100.0).round() as i64))
            .collect();
        let alpha_3s = 1.0 - (-3.0_f64 / 150.0).exp();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut hits = vec![vec![0usize; w_count]; rows];

        for path_idx in 0..n_paths {
            let ph = inferred_phase.map(|p| p % 3).unwrap_or(path_idx % 3);
            let mut o = state.oracle_cents;
            let mut b = state.book_cents;
            let mut e = state.ema_basis_cents;
            let mut db = state.last_db_cents;
            let mut x = state.mark_cents;
            let mut hit_cell = vec![vec![false; w_count]; rows];

            for s in 1..max_end {
                let prev_x = x;
                if (s + 3 - ph) % 3 == 0 {
                    let j = if params.pj > 0.0 && rng.random::<f64>() < params.pj {
                        let k = if params.geom_p >= 1.0 {
                            1.0
                        } else {
                            let mut u: f64 = rng.random();
                            while u <= 0.0 {
                                u = rng.random();
                            }
                            1.0 + (u.ln() / (1.0 - params.geom_p).ln()).floor()
                        };
                        if rng.random::<bool>() { k } else { -k }
                    } else {
                        0.0
                    };
                    o += params.a * db + params.lam * (b - o - e) + j;
                    let z: f64 = StandardNormal.sample(&mut rng);
                    db = params.phi * db + params.s_eta * z;
                    b += db;
                    let d = b - o;
                    e += alpha_3s * (d - e);
                    x = median3(o, o + e, b).round() as i64;
                }
                let lo = prev_x.min(x);
                let hi = prev_x.max(x);
                for (w_idx, w) in windows.iter().enumerate() {
                    if s >= w.start_secs as usize && s < w.end_secs as usize {
                        for (r, &(lc, uc)) in bounds.iter().enumerate() {
                            if lo <= uc && hi >= lc {
                                hit_cell[r][w_idx] = true;
                            }
                        }
                    }
                }
            }
            for r in 0..rows {
                for w_idx in 0..w_count {
                    if hit_cell[r][w_idx] {
                        hits[r][w_idx] += 1;
                    }
                }
            }
        }
        hits.into_iter()
            .map(|row| row.into_iter().map(|c| c as f64 / n_paths as f64).collect())
            .collect()
    }

    #[test]
    fn m4_mc_matches_independent_plain_reference_simulator() {
        let nvda_bands: Vec<PriceBand> = (-2..=2)
            .map(|k| PriceBand {
                row_index: (k + 2) as usize,
                lower: 145.97 + 0.03 * k as f64,
                upper: 146.00 + 0.03 * k as f64,
            })
            .collect();
        let tsla_bands: Vec<PriceBand> = (-2..=2)
            .map(|k| PriceBand {
                row_index: (k + 2) as usize,
                lower: 379.26 + 0.08 * k as f64,
                upper: 379.34 + 0.08 * k as f64,
            })
            .collect();
        let windows_12: Vec<TimeWindow> = (0..12)
            .map(|w| TimeWindow {
                start_secs: ((w + 1) * 5) as u32,
                end_secs: ((w + 2) * 5) as u32,
            })
            .collect();

        let n_paths = 60_000usize;
        let cases = [
            (
                "NVDA_normal_mid_up",
                &nvda_bands,
                &windows_12[..6],
                MarkFormulaState {
                    oracle_cents: 14_594.0,
                    book_cents: 14_603.0,
                    ema_basis_cents: 5.0,
                    last_db_cents: 3.0,
                    mark_cents: 14_598,
                },
                &NVDA_M4_PARAMS.presets[Regime::Normal.index()],
                Some(1),
                501u64,
                601u64,
            ),
            (
                "TSLA_volatile_mid_down_far",
                &tsla_bands,
                &windows_12[..12],
                MarkFormulaState {
                    oracle_cents: 37_935.0,
                    book_cents: 37_922.0,
                    ema_basis_cents: -4.0,
                    last_db_cents: -5.0,
                    mark_cents: 37_930,
                },
                &TSLA_M4_PARAMS.presets[Regime::Volatile.index()],
                None,
                502u64,
                602u64,
            ),
            (
                "NVDA_news_heavy_tail",
                &nvda_bands,
                &windows_12[..6],
                MarkFormulaState {
                    oracle_cents: 14_590.0,
                    book_cents: 14_608.0,
                    ema_basis_cents: 6.0,
                    last_db_cents: 8.0,
                    mark_cents: 14_598,
                },
                &NVDA_M4_PARAMS.presets[Regime::News.index()],
                Some(2),
                503u64,
                603u64,
            ),
        ];

        for (label, bands, wins, state, params, ph, seed_fast, seed_ref) in cases {
            let fast = price_grid_m4(bands, wins, state, params, ph, n_paths, seed_fast);
            let reference = reference_mc_m4(bands, wins, state, params, ph, n_paths, seed_ref);
            for r in 0..bands.len() {
                for w in 0..wins.len() {
                    let p_hat = 0.5 * (fast[r][w] + reference[r][w]);
                    // Exact 2-sample binomial SE for two independent N=60,000 runs:
                    // SE = sqrt(2 * p * (1 - p) / N) <= 0.00289; 4.5*SE <= 0.013 (two-sided p < 6.8e-6).
                    let se = (2.0 * p_hat * (1.0 - p_hat) / n_paths as f64)
                        .sqrt()
                        .max(1e-4);
                    assert!(
                        (fast[r][w] - reference[r][w]).abs() <= 4.5 * se,
                        "{label} r={r} w={w}: fast={:.5} ref={:.5} diff={:+.5} (4.5*se={:.5})",
                        fast[r][w],
                        reference[r][w],
                        fast[r][w] - reference[r][w],
                        4.5 * se
                    );
                }
            }
            // On NVDA_normal_mid_up (book +5c above mark), d=+1 (row 3) must exceed d=-1 (row 1) at w0.
            if label == "NVDA_normal_mid_up" {
                assert!(
                    fast[3][0] > fast[1][0],
                    "expected mid-side prob {:.4} > other-side prob {:.4}",
                    fast[3][0],
                    fast[1][0]
                );
            }
        }
    }

    #[test]
    fn m4_params_exist_for_nvda_and_tsla_only() {
        assert!(m4_ticker_params("NVDA").is_some());
        assert!(m4_ticker_params("xyz:TSLA").is_some());
        assert!(m4_ticker_params("AAPL").is_none());
        assert!(m4_ticker_params("SP500").is_none());
    }
}
