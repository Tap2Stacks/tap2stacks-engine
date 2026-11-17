use super::types::{CalibrationKey, GridConfig, TimeWindow, VolatilitySnapshot};

pub fn compute_m_eff(volatility: VolatilitySnapshot, config: &GridConfig) -> f64 {
    let epsilon = config.volatility.variance_floor;
    let sigma_term = (volatility.sigma / config.volatility.sigma_ref.max(epsilon) - 1.0).max(0.0);
    let lambda_term =
        (volatility.jump_intensity / config.pricing.lambda_ref.max(epsilon) - 1.0).max(0.0);
    let m_risk = (config.pricing.a_sigma * volatility.sigma * sigma_term
        + config.pricing.a_lambda * lambda_term)
        .clamp(0.0, config.pricing.m_risk_max);

    config.pricing.m_base + m_risk
}

pub fn compute_quote_probability(
    raw_hit_probability: f64,
    row_index: usize,
    window_index: usize,
    window: TimeWindow,
    volatility: VolatilitySnapshot,
    current_window0_row_index: usize,
    config: &GridConfig,
) -> f64 {
    let epsilon = config.volatility.variance_floor;
    let mut p_quote = raw_hit_probability.clamp(epsilon, 1.0);
    let center_row = config.spec.center_row;
    let max_dist = center_row.max(
        config
            .spec
            .rows
            .saturating_sub(1)
            .saturating_sub(center_row),
    );

    if config.pricing.p_raw_gamma > 0.0 && max_dist > 0 {
        let dist = row_index.abs_diff(center_row);
        let ratio = dist as f64 / max_dist as f64;
        let gamma = 1.0 + config.pricing.p_raw_gamma * ratio;
        p_quote = p_quote.powf(gamma).max(epsilon);
    }

    if !config.calibration.factors.is_empty() && max_dist > 0 {
        let dist_bucket = row_index
            .abs_diff(center_row)
            .min(config.pricing.calibration_dist_max);
        let sigma_ratio = volatility.sigma / config.volatility.sigma_ref.max(epsilon);
        let (edge_lo, edge_hi) = config.pricing.calibration_sigma_edges;
        let sigma_bucket = if sigma_ratio < edge_lo {
            0
        } else if sigma_ratio < edge_hi {
            1
        } else {
            2
        };
        let key = CalibrationKey {
            window_index,
            dist_bucket,
            sigma_bucket,
        };
        if let Some(factor) = config.calibration.factors.get(&key) {
            p_quote = (p_quote * *factor).clamp(epsilon, 1.0 - epsilon);
        }
    }

    if config.pricing.p_near_boost > 0.0 && max_dist > 0 {
        let dist = row_index.abs_diff(center_row);
        let ratio = dist as f64 / max_dist as f64;
        let boost = config.pricing.p_near_boost * (1.0 - ratio);
        p_quote = (p_quote * (1.0 + boost)).clamp(epsilon, 1.0 - epsilon);
    }

    if config.pricing.p_time_boost != 0.0 {
        let max_end = config
            .spec
            .windows
            .iter()
            .map(|current_window| current_window.end_secs)
            .max()
            .unwrap_or(window.end_secs)
            .max(1);
        let time_ratio = if window_index == 0 {
            0.0
        } else {
            window.end_secs as f64 / max_end as f64
        };
        let boost = config.pricing.p_time_boost * time_ratio;
        p_quote = (p_quote * (1.0 + boost)).clamp(epsilon, 1.0 - epsilon);
    }

    if window_index == 0 {
        let sigma_term =
            (volatility.sigma / config.volatility.sigma_ref.max(epsilon) - 1.0).max(0.0);
        let mut zone_shift = 0.0;
        if row_index == current_window0_row_index {
            zone_shift = config.pricing.current_row_logit_shift;
        } else if row_index.abs_diff(current_window0_row_index) == 1 {
            zone_shift = config.pricing.adjacent_row_logit_shift;
        }
        zone_shift += config.pricing.zone_vol_shift_scale * sigma_term;
        p_quote = apply_quote_logit_shift(p_quote, zone_shift, epsilon);
    }

    p_quote
}

pub fn compute_k_skew(liability: f64, config: &GridConfig) -> f64 {
    let epsilon = config.volatility.variance_floor;
    1.0 / (1.0 + config.pricing.beta_skew * (liability / config.pricing.pool_cap.max(epsilon)))
}

pub fn smooth_used_liability(
    previous_liability: f64,
    instant_liability: f64,
    config: &GridConfig,
) -> f64 {
    if instant_liability >= previous_liability {
        instant_liability
    } else {
        config.pricing.liability_rho * previous_liability
            + (1.0 - config.pricing.liability_rho) * instant_liability
    }
}

pub fn compute_k_safety(scenario_payouts: &[f64], config: &GridConfig) -> f64 {
    if scenario_payouts.is_empty() {
        return 1.0;
    }
    let epsilon = config.volatility.variance_floor;
    let mut payouts = scenario_payouts.to_vec();
    payouts.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let quantile_index = ((payouts.len() as f64 - 1.0) * 0.99).floor() as usize;
    let q = payouts[quantile_index.min(payouts.len() - 1)];
    let mut tail_sum = 0.0;
    let mut tail_count = 0usize;
    for payout in payouts {
        if payout >= q {
            tail_sum += payout;
            tail_count += 1;
        }
    }
    let cvar = if tail_count == 0 {
        q
    } else {
        tail_sum / tail_count as f64
    };
    if cvar <= 0.0 {
        return 1.0;
    }

    (config.pricing.risk_budget / cvar.max(epsilon)).clamp(0.0, 1.0)
}

pub fn apply_quote_logit_shift(probability: f64, shift: f64, epsilon: f64) -> f64 {
    if shift <= 0.0 {
        return probability;
    }
    let p = probability.clamp(epsilon, 1.0 - epsilon);
    let logit = (p / (1.0 - p)).ln();
    let shifted = 1.0 / (1.0 + (-(logit - shift)).exp());
    shifted.clamp(epsilon, 1.0 - epsilon)
}

pub fn price_multiplier(
    hit_probability: f64,
    volatility: VolatilitySnapshot,
    liability: f64,
    k_safety: f64,
    config: &GridConfig,
) -> f64 {
    let bounded_probability = hit_probability.clamp(config.volatility.variance_floor, 1.0);
    let base = (1.0 - compute_m_eff(volatility, config)) / bounded_probability;
    let k_skew = compute_k_skew(liability, config);
    (base * k_skew * k_safety * config.pricing.skew_factor * config.pricing.safety_factor)
        .clamp(config.pricing.min_multiplier, config.pricing.max_multiplier)
}
