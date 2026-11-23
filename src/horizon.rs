use std::collections::BTreeMap;

use crate::grid::{CalibrationKey, GridConfig};

pub const NEAR_WINDOWS: usize = 6;
pub const FAR_REFRESH_MS: u64 = 5_000;
pub const DEFAULT_WINDOWS: usize = 6;
pub const TWO_MINUTE_WINDOWS: usize = 24;
pub const CAP25_EPS: f64 = 1e-6;
pub const MIN_CALIBRATION_COUNT: usize = 200;

const SIGMA_BUCKETS: usize = 3;

pub fn far_reprice_due(
    now_ms: u64,
    price: f64,
    last_ms: u64,
    last_price: f64,
    sigma: f64,
    interval_ms: u64,
) -> bool {
    let elapsed = now_ms.saturating_sub(last_ms);
    if elapsed >= interval_ms {
        return true;
    }
    if !(price > 0.0 && last_price > 0.0 && sigma > 0.0) {
        return true;
    }
    let dt_seconds = elapsed as f64 / 1_000.0;
    if dt_seconds <= 0.0 {
        return false;
    }
    (price / last_price).ln().abs() > sigma * dt_seconds.sqrt()
}

pub fn move_trigger_due(
    now_ms: u64,
    price: f64,
    last_ms: u64,
    last_price: f64,
    sigma: f64,
) -> bool {
    far_reprice_due(now_ms, price, last_ms, last_price, sigma, u64::MAX)
}

pub fn factor_from_counts(
    hits: usize,
    count: usize,
    predicted_sum: f64,
    min_count: usize,
) -> Option<f64> {
    if count < min_count || predicted_sum <= 0.0 {
        return None;
    }
    let mean = predicted_sum / count as f64;
    if mean <= 0.0 {
        return None;
    }
    Some((hits as f64 / count as f64 / mean).clamp(0.25, 2.0))
}

/// Computes the exact `CalibrationKey.dist_bucket` matching `QuoteEngine::build_cent_bands`
/// and `compute_quote_probability`.
pub fn row_dist_bucket(spot: f64, band_cents: u32, lower_cents: i64, dist_max: usize) -> usize {
    if band_cents == 0 || !spot.is_finite() || spot <= 0.0 {
        return 0;
    }
    let band_c = band_cents as i64;
    let spot_cents = (spot * 100.0 + 1e-9).floor() as i64;
    let anchor_cents = spot_cents.div_euclid(band_c) * band_c;
    let center_lower_cents = anchor_cents - (band_c / 2);
    let offset = (lower_cents - center_lower_cents).div_euclid(band_c);
    (offset.unsigned_abs() as usize).min(dist_max)
}

pub fn parse_factor_table(text: &str) -> Result<Vec<(usize, usize, f64)>, String> {
    let mut entries = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|part| part.trim()).collect();
        if parts.len() != 3 {
            return Err(format!(
                "invalid factor row at line {}: expected window,dist,factor",
                line_idx + 1
            ));
        }
        let window_index: usize = parts[0]
            .parse()
            .map_err(|_| format!("invalid window_index at line {}", line_idx + 1))?;
        let dist_bucket: usize = parts[1]
            .parse()
            .map_err(|_| format!("invalid dist_bucket at line {}", line_idx + 1))?;
        let factor: f64 = parts[2]
            .parse()
            .map_err(|_| format!("invalid factor at line {}", line_idx + 1))?;
        if !factor.is_finite() || factor <= 0.0 {
            return Err(format!(
                "factor must be finite and positive at line {}",
                line_idx + 1
            ));
        }
        entries.push((window_index, dist_bucket, factor));
    }
    Ok(entries)
}

pub fn apply_factor_table(cfg: &mut GridConfig, entries: &[(usize, usize, f64)]) {
    for &(window_index, dist_bucket, factor) in entries {
        for sigma_bucket in 0..SIGMA_BUCKETS {
            cfg.calibration.factors.insert(
                CalibrationKey {
                    window_index,
                    dist_bucket,
                    sigma_bucket,
                },
                factor,
            );
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepriceStats {
    pub full_reprices: u64,
    pub near_reprices: u64,
    pub trigger_reprices: u64,
    pub cadence_reprices: u64,
    pub sample_count: u64,
    pub sample_ns: u128,
    pub full_ns: u128,
    pub near_ns: u128,
    pub trigger_timestamps_ms: Vec<u64>,
}

impl RepriceStats {
    pub fn mean_us(&self) -> u64 {
        if self.sample_count == 0 {
            0
        } else {
            (self.sample_ns / u128::from(self.sample_count) / 1_000) as u64
        }
    }

    pub fn full_mean_us(&self) -> u64 {
        if self.full_reprices == 0 {
            0
        } else {
            (self.full_ns / u128::from(self.full_reprices) / 1_000) as u64
        }
    }

    pub fn near_mean_us(&self) -> u64 {
        if self.near_reprices == 0 {
            0
        } else {
            (self.near_ns / u128::from(self.near_reprices) / 1_000) as u64
        }
    }

    pub fn merge(&mut self, other: &RepriceStats) {
        self.full_reprices += other.full_reprices;
        self.near_reprices += other.near_reprices;
        self.trigger_reprices += other.trigger_reprices;
        self.cadence_reprices += other.cadence_reprices;
        self.sample_count += other.sample_count;
        self.sample_ns += other.sample_ns;
        self.full_ns += other.full_ns;
        self.near_ns += other.near_ns;
        self.trigger_timestamps_ms
            .extend_from_slice(&other.trigger_timestamps_ms);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExposureInterval {
    pub open_ms: u64,
    pub close_ms: u64,
    pub stake: f64,
    pub liability: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OutstandingExposureSummary {
    pub active_minutes: usize,
    pub total_minutes: usize,
    pub mean_active_stake: f64,
    pub p95_active_peak_stake: f64,
    pub peak_stake: f64,
    pub peak_stake_ms: u64,
    pub mean_active_liability: f64,
    pub p95_active_peak_liability: f64,
    pub peak_liability: f64,
    pub peak_liability_ms: u64,
    pub mean_active_open_taps: f64,
    pub peak_open_taps: usize,
}

#[derive(Default)]
struct MinuteExposureAcc {
    stake_ms: f64,
    liability_ms: f64,
    taps_ms: f64,
    peak_stake: f64,
    peak_liability: f64,
}

pub fn compute_outstanding_exposure(intervals: &[ExposureInterval]) -> OutstandingExposureSummary {
    let mut events: Vec<(u64, bool, f64, f64)> = Vec::with_capacity(intervals.len() * 2);
    let mut min_open = u64::MAX;
    let mut max_close = 0u64;
    for interval in intervals {
        if interval.close_ms <= interval.open_ms || interval.stake <= 0.0 {
            continue;
        }
        min_open = min_open.min(interval.open_ms);
        max_close = max_close.max(interval.close_ms);
        events.push((interval.open_ms, true, interval.stake, interval.liability));
        events.push((interval.close_ms, false, interval.stake, interval.liability));
    }
    if events.is_empty() {
        return OutstandingExposureSummary::default();
    }
    // Process closes (false) before opens (true) at identical timestamps (half-open [open, close)).
    events.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    let mut cur_stake = 0.0_f64;
    let mut cur_liability = 0.0_f64;
    let mut open_taps = 0usize;
    let mut peak_stake = 0.0_f64;
    let mut peak_stake_ms = min_open;
    let mut peak_liability = 0.0_f64;
    let mut peak_liability_ms = min_open;
    let mut peak_open_taps = 0usize;

    let mut minutes: BTreeMap<u64, MinuteExposureAcc> = BTreeMap::new();
    let mut prev_ts = events[0].0;
    let mut idx = 0usize;

    while idx < events.len() {
        let ts = events[idx].0;
        if ts > prev_ts && open_taps > 0 {
            let mut m = prev_ts / 60_000 * 60_000;
            while m < ts {
                let seg_start = prev_ts.max(m);
                let seg_end = ts.min(m.saturating_add(60_000));
                if seg_end > seg_start {
                    let dur = (seg_end - seg_start) as f64;
                    let acc = minutes.entry(m).or_default();
                    acc.stake_ms += cur_stake * dur;
                    acc.liability_ms += cur_liability * dur;
                    acc.taps_ms += (open_taps as f64) * dur;
                    if cur_stake > acc.peak_stake {
                        acc.peak_stake = cur_stake;
                    }
                    if cur_liability > acc.peak_liability {
                        acc.peak_liability = cur_liability;
                    }
                }
                m = m.saturating_add(60_000);
            }
        }
        while idx < events.len() && events[idx].0 == ts {
            let (_, is_open, stake, liability) = events[idx];
            if is_open {
                cur_stake += stake;
                cur_liability += liability;
                open_taps += 1;
            } else {
                cur_stake = (cur_stake - stake).max(0.0);
                cur_liability = (cur_liability - liability).max(0.0);
                open_taps = open_taps.saturating_sub(1);
            }
            idx += 1;
        }
        if cur_stake > peak_stake {
            peak_stake = cur_stake;
            peak_stake_ms = ts;
        }
        if cur_liability > peak_liability {
            peak_liability = cur_liability;
            peak_liability_ms = ts;
        }
        if open_taps > peak_open_taps {
            peak_open_taps = open_taps;
        }
        prev_ts = ts;
    }

    let first_minute = min_open / 60_000 * 60_000;
    let last_minute = max_close.saturating_sub(1) / 60_000 * 60_000;
    let total_minutes = ((last_minute.saturating_sub(first_minute)) / 60_000 + 1) as usize;
    let active_minutes = minutes.len();
    if active_minutes == 0 {
        return OutstandingExposureSummary {
            total_minutes,
            peak_stake,
            peak_stake_ms,
            peak_liability,
            peak_liability_ms,
            peak_open_taps,
            ..OutstandingExposureSummary::default()
        };
    }

    let mean_active_stake = minutes
        .values()
        .map(|acc| acc.stake_ms / 60_000.0)
        .sum::<f64>()
        / active_minutes as f64;
    let mean_active_liability = minutes
        .values()
        .map(|acc| acc.liability_ms / 60_000.0)
        .sum::<f64>()
        / active_minutes as f64;
    let mean_active_open_taps = minutes
        .values()
        .map(|acc| acc.taps_ms / 60_000.0)
        .sum::<f64>()
        / active_minutes as f64;

    let mut peak_stakes: Vec<f64> = minutes.values().map(|acc| acc.peak_stake).collect();
    let mut peak_liabilities: Vec<f64> = minutes.values().map(|acc| acc.peak_liability).collect();
    peak_stakes.sort_by(|a, b| a.total_cmp(b));
    peak_liabilities.sort_by(|a, b| a.total_cmp(b));
    let p95_idx = ((active_minutes as f64 - 1.0) * 0.95).floor() as usize;

    OutstandingExposureSummary {
        active_minutes,
        total_minutes,
        mean_active_stake,
        p95_active_peak_stake: peak_stakes[p95_idx.min(active_minutes - 1)],
        peak_stake,
        peak_stake_ms,
        mean_active_liability,
        p95_active_peak_liability: peak_liabilities[p95_idx.min(active_minutes - 1)],
        peak_liability,
        peak_liability_ms,
        mean_active_open_taps,
        peak_open_taps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::stock_grid_config_for_ticker;

    #[test]
    fn quiet_tick_waits_for_the_five_second_cadence() {
        assert!(!far_reprice_due(1_000, 100.0, 0, 100.0, 0.001, FAR_REFRESH_MS));
        assert!(far_reprice_due(5_000, 100.0, 0, 100.0, 0.001, FAR_REFRESH_MS));
    }

    #[test]
    fn news_sized_move_reprices_before_the_cadence() {
        let sigma = 0.001_f64;
        let jumped = 100.0 * 0.01_f64.exp();
        assert!(far_reprice_due(1_000, jumped, 0, 100.0, sigma, FAR_REFRESH_MS));
        assert!(move_trigger_due(1_000, jumped, 0, 100.0, sigma));
        let tiny = 100.0 * (0.5 * sigma).exp();
        assert!(!far_reprice_due(1_000, tiny, 0, 100.0, sigma, FAR_REFRESH_MS));
        assert!(!move_trigger_due(1_000, tiny, 0, 100.0, sigma));
        // Same-millisecond duplicate tick does not trigger.
        assert!(!move_trigger_due(0, jumped, 0, 100.0, sigma));
    }

    #[test]
    fn factor_is_hit_rate_over_predicted_and_clamped() {
        assert_eq!(factor_from_counts(10, 40, 20.0, 50), None);
        let factor = factor_from_counts(40, 100, 80.0, 50).unwrap();
        assert!((factor - 0.5).abs() < 1e-12);
        assert_eq!(factor_from_counts(100, 100, 10.0, 50).unwrap(), 2.0);
        assert_eq!(factor_from_counts(1, 100, 90.0, 50).unwrap(), 0.25);
    }

    #[test]
    fn row_dist_bucket_matches_band_row_distance_from_center() {
        for &(spot, band_cents) in &[(139.08_f64, 3u32), (157.53, 3), (250.04, 8), (5823.40, 150)] {
            let band_c = band_cents as i64;
            let spot_cents = (spot * 100.0 + 1e-9).floor() as i64;
            let anchor_cents = spot_cents.div_euclid(band_c) * band_c;
            let center_lower = anchor_cents - (band_c / 2);
            let center_row = 9usize;
            for row_index in 0..20usize {
                let lower_cents = center_lower + (row_index as i64 - center_row as i64) * band_c;
                let expected = row_index.abs_diff(center_row).min(6);
                let got = row_dist_bucket(spot, band_cents, lower_cents, 6);
                assert_eq!(got, expected, "spot={spot} band={band_cents} row={row_index}");
            }
        }
    }

    #[test]
    fn factor_table_parses_and_applies_without_clobbering_near_windows() {
        let text = "# comment\n\n6,0,0.92\n6,1,0.75\n23,6,1.10\n";
        let entries = parse_factor_table(text).unwrap();
        assert_eq!(entries, vec![(6, 0, 0.92), (6, 1, 0.75), (23, 6, 1.10)]);

        let mut cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        apply_factor_table(&mut cfg, &entries);
        // Near windows stay untouched.
        assert_eq!(
            cfg.calibration.factors.get(&CalibrationKey {
                window_index: 0,
                dist_bucket: 1,
                sigma_bucket: 1,
            }),
            Some(&0.72)
        );
        // Far windows populated for all 3 sigma buckets.
        for sigma_bucket in 0..3 {
            assert_eq!(
                cfg.calibration.factors.get(&CalibrationKey {
                    window_index: 6,
                    dist_bucket: 0,
                    sigma_bucket,
                }),
                Some(&0.92)
            );
            assert_eq!(
                cfg.calibration.factors.get(&CalibrationKey {
                    window_index: 23,
                    dist_bucket: 6,
                    sigma_bucket,
                }),
                Some(&1.10)
            );
        }

        assert!(parse_factor_table("6,1,0.0").is_err());
        assert!(parse_factor_table("6,1,-0.5").is_err());
        assert!(parse_factor_table("6,1,NaN").is_err());
        assert!(parse_factor_table("6,1").is_err());
    }

    #[test]
    fn outstanding_exposure_reflects_horizon_length_and_half_open_boundaries() {
        // Two back-to-back non-overlapping 30s bets: peak should be $10, not $20.
        let non_overlap = vec![
            ExposureInterval {
                open_ms: 0,
                close_ms: 30_000,
                stake: 10.0,
                liability: 25.0,
            },
            ExposureInterval {
                open_ms: 30_000,
                close_ms: 60_000,
                stake: 10.0,
                liability: 40.0,
            },
        ];
        let s1 = compute_outstanding_exposure(&non_overlap);
        assert_eq!(s1.active_minutes, 1);
        assert_eq!(s1.peak_stake, 10.0);
        assert_eq!(s1.peak_liability, 40.0);
        assert!((s1.mean_active_stake - 10.0).abs() < 1e-9);
        assert_eq!(s1.peak_open_taps, 1);

        // Two 120s bets placed 30s apart overlap for 90s -> peak is $20 across 2 active taps.
        let far_overlap = vec![
            ExposureInterval {
                open_ms: 0,
                close_ms: 120_000,
                stake: 10.0,
                liability: 30.0,
            },
            ExposureInterval {
                open_ms: 30_000,
                close_ms: 150_000,
                stake: 10.0,
                liability: 50.0,
            },
        ];
        let s2 = compute_outstanding_exposure(&far_overlap);
        assert_eq!(s2.active_minutes, 3);
        assert_eq!(s2.peak_stake, 20.0);
        assert_eq!(s2.peak_stake_ms, 30_000);
        assert_eq!(s2.peak_liability, 80.0);
        assert_eq!(s2.peak_open_taps, 2);
        // Minute 0: 30s@$10 + 30s@$20 = $15; Minute 1: 60s@$20 = $20; Minute 2: 30s@$10 = $5 -> mean $40/3.
        assert!((s2.mean_active_stake - 40.0 / 3.0).abs() < 1e-9);
    }
}
