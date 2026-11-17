use super::types::{PriceBand, TimeWindow};

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
}
