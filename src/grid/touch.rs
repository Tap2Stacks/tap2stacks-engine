use super::types::{PriceBand, TimeWindow};

/// Returns `1.0` if any step in the half-open window `[start, end)` has
/// `round(price * 100)` within `[round(lower * 100), round(upper * 100)]` inclusive, else `0.0`.
pub fn window_hit_discrete(
    path: &[f64],
    band: &PriceBand,
    window: &TimeWindow,
    lock_offset: u32,
) -> f64 {
    let len = path.len();
    let start = (window.start_secs.saturating_add(lock_offset) as usize).min(len);
    let end = (window.end_secs.saturating_add(lock_offset) as usize).min(len);
    window_hit_discrete_steps(path, band.lower, band.upper, start, end)
}

pub(crate) fn window_hit_discrete_steps(
    path: &[f64],
    lower: f64,
    upper: f64,
    start: usize,
    end: usize,
) -> f64 {
    if path.is_empty() || start >= end || !lower.is_finite() || !upper.is_finite() {
        return 0.0;
    }
    // Half-open [start, end): matches settlement, which counts samples with
    // windowStartMs <= tsMs < windowEndMs.
    let end = end.min(path.len());
    if start >= end {
        return 0.0;
    }
    let lower_cents = (lower * 100.0).round() as i64;
    let upper_cents = (upper * 100.0).round() as i64;
    for &price in &path[start..end] {
        if price.is_finite() {
            let cents = (price * 100.0).round() as i64;
            if cents >= lower_cents && cents <= upper_cents {
                return 1.0;
            }
        }
    }
    0.0
}

pub fn window_hit_probability(
    path: &[f64],
    band: PriceBand,
    window: TimeWindow,
    sigma: f64,
    lock_offset: u32,
    epsilon: f64,
) -> f64 {
    let sigma_sq = sigma.powi(2);
    let n_steps = path.len().saturating_sub(1);
    let start = (window.start_secs.saturating_add(lock_offset) as usize).min(n_steps);
    let end = (window.end_secs.saturating_add(lock_offset) as usize).min(n_steps);
    window_hit_probability_steps(path, band.lower, band.upper, start, end, sigma_sq, epsilon)
}

pub(crate) fn window_hit_probability_steps(
    path: &[f64],
    lower: f64,
    upper: f64,
    start: usize,
    end: usize,
    sigma_sq: f64,
    epsilon: f64,
) -> f64 {
    let mut p_not_touch = 1.0;
    let band = PriceBand {
        row_index: 0,
        lower,
        upper,
    };

    for second in start..end {
        let p_hit = interval_hit_probability(path[second], path[second + 1], band, sigma_sq, epsilon);
        p_not_touch *= 1.0 - p_hit;
    }

    1.0 - p_not_touch
}

/// Below `1.0_f64`, ULP spacing is `2^-53 ≈ 1.11e-16`, so half-ULP is `2^-54 ≈ 5.55e-17`.
/// `exp(-38.0) ≈ 3.14e-17 < 2^-54`, so `1.0 - exp(exponent) == 1.0` bit-for-bit in IEEE-754 f64.
pub(crate) const BRIDGE_PRUNE_EXPONENT: f64 = -38.0;

/// Fast log-space equivalent of `window_hit_probability_steps`.
/// `log_path[t]` is `ln(price[t])`; `log_lower`/`log_upper` are `ln(band.lower)`/`ln(band.upper)`;
/// `neg_two_inv_denom` is `-2.0 / (sigma_sq + epsilon)`.
pub(crate) fn window_hit_probability_log_steps(
    log_path: &[f64],
    log_lower: f64,
    log_upper: f64,
    start: usize,
    end: usize,
    neg_two_inv_denom: f64,
) -> f64 {
    let mut p_not_touch = 1.0_f64;
    for second in start..end {
        let s = log_path[second];
        let e = log_path[second + 1];
        if s < log_lower && e < log_lower {
            let exponent = neg_two_inv_denom * (log_lower - s) * (log_lower - e);
            if exponent > BRIDGE_PRUNE_EXPONENT {
                let p_hit = exponent.exp().clamp(0.0, 1.0);
                p_not_touch *= 1.0 - p_hit;
            }
        } else if s > log_upper && e > log_upper {
            let exponent = neg_two_inv_denom * (log_upper - s) * (log_upper - e);
            if exponent > BRIDGE_PRUNE_EXPONENT {
                let p_hit = exponent.exp().clamp(0.0, 1.0);
                p_not_touch *= 1.0 - p_hit;
            }
        } else {
            return 1.0;
        }
    }
    1.0 - p_not_touch
}

pub fn interval_hit_probabilities(
    path: &[f64],
    band: PriceBand,
    window: TimeWindow,
    sigma: f64,
    lock_offset: u32,
    epsilon: f64,
) -> Vec<f64> {
    let sigma_sq = sigma.powi(2);
    let n_steps = path.len().saturating_sub(1);
    let start = (window.start_secs.saturating_add(lock_offset) as usize).min(n_steps);
    let end = (window.end_secs.saturating_add(lock_offset) as usize).min(n_steps);
    let mut probabilities = Vec::with_capacity(end.saturating_sub(start));

    for second in start..end {
        let segment_start = path[second];
        let segment_end = path[second + 1];
        probabilities.push(interval_hit_probability(
            segment_start,
            segment_end,
            band,
            sigma_sq,
            epsilon,
        ));
    }

    probabilities
}

pub fn interval_hit_probability(
    start: f64,
    end: f64,
    band: PriceBand,
    sigma_sq: f64,
    epsilon: f64,
) -> f64 {
    if point_in_band(start, band) || point_in_band(end, band) {
        return 1.0;
    }

    if start < band.lower && end > band.upper {
        return 1.0;
    }
    if start > band.upper && end < band.lower {
        return 1.0;
    }

    if start < band.lower && end < band.lower {
        return lower_barrier_hit_probability(start, end, band.lower, sigma_sq, epsilon);
    }

    if start > band.upper && end > band.upper {
        return upper_barrier_hit_probability(start, end, band.upper, sigma_sq, epsilon);
    }

    1.0
}

fn point_in_band(price: f64, band: PriceBand) -> bool {
    price >= band.lower && price <= band.upper
}

fn lower_barrier_hit_probability(
    start: f64,
    end: f64,
    lower: f64,
    sigma_sq: f64,
    epsilon: f64,
) -> f64 {
    (-2.0 * (lower / start).ln() * (lower / end).ln() / (sigma_sq + epsilon))
        .exp()
        .clamp(0.0, 1.0)
}

fn upper_barrier_hit_probability(
    start: f64,
    end: f64,
    upper: f64,
    sigma_sq: f64,
    epsilon: f64,
) -> f64 {
    (-2.0 * (upper / start).ln() * (upper / end).ln() / (sigma_sq + epsilon))
        .exp()
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_band() -> PriceBand {
        PriceBand {
            row_index: 0,
            lower: 50_000.0,
            upper: 50_010.0,
        }
    }

    #[test]
    fn interval_below_band_uses_barrier_probability_not_certain_hit() {
        let probability = interval_hit_probability(49_900.0, 49_910.0, sample_band(), 0.25, 1e-6);

        assert!(probability > 0.0);
        assert!(probability < 1.0);
    }

    #[test]
    fn interval_above_band_uses_barrier_probability_not_certain_hit() {
        let probability = interval_hit_probability(50_100.0, 50_090.0, sample_band(), 0.25, 1e-6);

        assert!(probability > 0.0);
        assert!(probability < 1.0);
    }

    #[test]
    fn discrete_sample_at_window_end_does_not_count() {
        // Window [1, 3): indexes 1 and 2 count, index 3 does not.
        let path = [157.00, 157.00, 157.00, 157.55];
        let band = PriceBand {
            row_index: 0,
            lower: 157.52,
            upper: 157.55,
        };
        let window = TimeWindow {
            start_secs: 1,
            end_secs: 3,
        };
        assert_eq!(window_hit_discrete(&path, &band, &window, 0), 0.0);
    }

    #[test]
    fn discrete_hit_when_a_step_lands_in_band() {
        let path = [157.50, 157.52, 157.55];
        let band = PriceBand {
            row_index: 0,
            lower: 157.52,
            upper: 157.55,
        };
        let window = TimeWindow {
            start_secs: 1,
            end_secs: 3,
        };

        assert_eq!(window_hit_discrete(&path, &band, &window, 0), 1.0);
    }

    #[test]
    fn discrete_miss_when_path_jumps_over_band() {
        let path = [157.40, 157.70];
        let band = PriceBand {
            row_index: 0,
            lower: 157.52,
            upper: 157.55,
        };
        let window = TimeWindow {
            start_secs: 0,
            end_secs: 1,
        };

        assert_eq!(window_hit_discrete(&path, &band, &window, 0), 0.0);
        assert!(window_hit_probability(&path, band, window, 0.01, 0, 1e-6) > 0.0);
    }

    #[test]
    fn band_edge_counts_for_both_adjacent_cells() {
        let path = [157.40, 157.53];
        let lower_band = PriceBand {
            row_index: 0,
            lower: 157.50,
            upper: 157.53,
        };
        let upper_band = PriceBand {
            row_index: 1,
            lower: 157.53,
            upper: 157.56,
        };
        // Index 1 (157.53, the shared edge) is inside the half-open window [1, 2).
        let window = TimeWindow {
            start_secs: 1,
            end_secs: 2,
        };

        assert_eq!(window_hit_discrete(&path, &lower_band, &window, 0), 1.0);
        assert_eq!(window_hit_discrete(&path, &upper_band, &window, 0), 1.0);
    }

    #[test]
    fn log_space_bridge_matches_reference_touch_probability() {
        let path = [140.00_f64, 140.04, 140.01, 140.07, 140.03, 140.09];
        let log_path: Vec<f64> = path.iter().map(|p| p.ln()).collect();
        let sigma = 0.0004_f64;
        let eps = 1e-12_f64;
        let neg_two_inv_denom = -2.0 / (sigma * sigma + eps);
        let window = TimeWindow {
            start_secs: 0,
            end_secs: 5,
        };
        for row in -10..=10_i32 {
            let lower = 140.00 + f64::from(row) * 0.03;
            let upper = lower + 0.03;
            let band = PriceBand {
                row_index: 0,
                lower,
                upper,
            };
            let ref_p = window_hit_probability(&path, band, window, sigma, 0, eps);
            let fast_p = window_hit_probability_log_steps(
                &log_path,
                lower.ln(),
                upper.ln(),
                0,
                5,
                neg_two_inv_denom,
            );
            assert!(
                (ref_p - fast_p).abs() < 1e-12,
                "row={row} ref={ref_p} fast={fast_p}"
            );
        }
    }
}
