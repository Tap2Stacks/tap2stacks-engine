use super::types::{GridConfig, GridError, OhlcBar, RuntimeState, VolatilitySnapshot};

/// Prior sample weight `k` when blending initial EWMA variance from restart history.
const SEED_PRIOR_WEIGHT: f64 = 12.0;

pub fn estimate_volatility(
    spot: f64,
    history: &[OhlcBar],
    config: &GridConfig,
) -> Result<VolatilitySnapshot, GridError> {
    Ok(estimate_runtime_state(spot, history, config)?.volatility)
}

pub fn estimate_runtime_state(
    spot: f64,
    history: &[OhlcBar],
    config: &GridConfig,
) -> Result<RuntimeState, GridError> {
    estimate_runtime_state_with_seed_prior(spot, history, config, None)
}

/// Like `estimate_runtime_state`, but substitutes a measured variance for the
/// cold-start `sigma_ref` prior.
pub fn estimate_runtime_state_with_seed_prior(
    spot: f64,
    history: &[OhlcBar],
    config: &GridConfig,
    seed_variance_prior: Option<f64>,
) -> Result<RuntimeState, GridError> {
    if spot <= 0.0 {
        return Err(GridError::InvalidSpot);
    }
    if history.is_empty() {
        return Err(GridError::InvalidHistory(
            "at least one OHLC bar is required",
        ));
    }

    for bar in history {
        validate_bar(*bar)?;
    }

    let mut state = if let Some(seed_variance) =
        seeded_initial_variance(history, config, seed_variance_prior)
    {
        let mut s = RuntimeState {
            last_price: Some(history[0].close),
            variance: Some(seed_variance),
            ..RuntimeState::default()
        };
        for bar in &history[1..] {
            s = update_runtime_state_with_seed_prior(
                &s,
                bar.close,
                Some(bar),
                config,
                seed_variance_prior,
            )?;
        }
        s
    } else {
        update_runtime_state_with_seed_prior(
            &RuntimeState::default(),
            history[0].close,
            Some(&history[0]),
            config,
            seed_variance_prior,
        )?
    };

    state.volatility = reprice_sigma_for_spot(state.volatility, spot, config);
    Ok(state)
}

fn prior_variance(config: &GridConfig, seed_variance_prior: Option<f64>) -> f64 {
    seed_variance_prior
        .filter(|variance| (*variance).is_finite() && *variance > 0.0)
        .unwrap_or_else(|| config.volatility.sigma_ref.powi(2))
}

fn seeded_initial_variance(
    history: &[OhlcBar],
    config: &GridConfig,
    seed_variance_prior: Option<f64>,
) -> Option<f64> {
    if history.len() < 2 {
        return None;
    }
    let variance_floor = config.volatility.variance_floor;
    let mut sum_sq = 0.0;
    for pair in history.windows(2) {
        let log_return = (pair[1].close / pair[0].close.max(variance_floor)).ln();
        sum_sq += log_return * log_return;
    }
    let n = (history.len() - 1) as f64;
    let realized = sum_sq / n;
    let prior = prior_variance(config, seed_variance_prior);
    let blended = (n * realized + SEED_PRIOR_WEIGHT * prior) / (n + SEED_PRIOR_WEIGHT);
    Some(blended.max(variance_floor))
}

pub fn update_runtime_state(
    previous_state: &RuntimeState,
    price: f64,
    bar: Option<&OhlcBar>,
    config: &GridConfig,
) -> Result<RuntimeState, GridError> {
    update_runtime_state_with_seed_prior(previous_state, price, bar, config, None)
}

fn update_runtime_state_with_seed_prior(
    previous_state: &RuntimeState,
    price: f64,
    bar: Option<&OhlcBar>,
    config: &GridConfig,
    seed_variance_prior: Option<f64>,
) -> Result<RuntimeState, GridError> {
    if price <= 0.0 {
        return Err(GridError::InvalidSpot);
    }
    if let Some(bar) = bar {
        validate_bar(*bar)?;
    }

    let decay = (-std::f64::consts::LN_2 / config.volatility.ewma_half_life_seconds).exp();
    let variance_floor = config.volatility.variance_floor;
    let previous_price = previous_state.last_price.unwrap_or(price);
    let log_return = (price / previous_price.max(variance_floor)).ln();
    let park_var = bar
        .map(|current_bar| {
            ((current_bar.high / current_bar.low).ln()).powi(2) / (4.0 * std::f64::consts::LN_2)
        })
        .unwrap_or(0.0);
    let observed_variance = (1.0 - config.volatility.parkinson_weight) * log_return.powi(2)
        + config.volatility.parkinson_weight * park_var;

    let previous_variance = previous_state
        .variance
        .unwrap_or_else(|| prior_variance(config, seed_variance_prior));
    let variance =
        ((1.0 - decay) * observed_variance + decay * previous_variance).max(variance_floor);
    let sigma_raw = variance.sqrt().max(variance_floor.sqrt());
    let z_score = log_return.abs() / (sigma_raw + variance_floor);
    let sigma_ratio =
        (sigma_raw / config.volatility.sigma_ref.max(variance_floor)).max(variance_floor);
    let jump_threshold = clamp(
        config.volatility.jump.kappa0 * sigma_ratio.powf(config.volatility.jump.kappa_q),
        config.volatility.jump.kappa_min,
        config.volatility.jump.kappa_max,
    );
    let jump_flag = config.volatility.jump.enabled && z_score > jump_threshold;
    let mut jump_sampler = previous_state.jump_sampler.clone();
    if jump_flag {
        jump_sampler.record(log_return);
    }

    let lambda_prev = if previous_state.volatility.jump_intensity > 0.0 {
        previous_state.volatility.jump_intensity
    } else {
        config.volatility.jump.hawkes_mu0
    };
    let jump_intensity = (config.volatility.jump.hawkes_mu0
        + (-config.volatility.jump.beta).exp() * (lambda_prev - config.volatility.jump.hawkes_mu0)
        + config.volatility.jump.alpha * if jump_flag { 1.0 } else { 0.0 })
    .max(0.0);
    let price_adj = price_adjustment(price, config);
    let sigma = (sigma_raw * config.volatility.sigma_scale * price_adj).max(variance_floor.sqrt());

    Ok(RuntimeState {
        variance: Some(variance),
        last_price: Some(price),
        volatility: VolatilitySnapshot {
            variance,
            sigma_raw,
            sigma,
            jump_intensity,
            jump_flag,
        },
        jump_sampler,
    })
}

fn validate_bar(bar: OhlcBar) -> Result<(), GridError> {
    if bar.open <= 0.0 || bar.high <= 0.0 || bar.low <= 0.0 || bar.close <= 0.0 {
        return Err(GridError::InvalidHistory("prices must be positive"));
    }
    if bar.low > bar.high {
        return Err(GridError::InvalidHistory("low must be <= high"));
    }
    Ok(())
}

fn price_adjustment(price: f64, config: &GridConfig) -> f64 {
    let base = (price.max(config.volatility.price_adj_min_price)
        / config.volatility.price_adj_reference)
        .sqrt();
    clamp(
        base,
        config.volatility.price_adj_min,
        config.volatility.price_adj_max,
    )
}

fn reprice_sigma_for_spot(
    mut snapshot: VolatilitySnapshot,
    spot: f64,
    config: &GridConfig,
) -> VolatilitySnapshot {
    snapshot.sigma =
        (snapshot.sigma_raw * config.volatility.sigma_scale * price_adjustment(spot, config))
            .max(config.volatility.variance_floor.sqrt());
    snapshot
}

fn clamp(value: f64, lower: f64, upper: f64) -> f64 {
    value.max(lower).min(upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::stock::stock_grid_config_for_ticker;

    fn flat_bars(prices: &[f64]) -> Vec<OhlcBar> {
        prices
            .iter()
            .map(|&p| OhlcBar {
                open: p,
                high: p,
                low: p,
                close: p,
            })
            .collect()
    }

    #[test]
    fn single_bar_falls_back_to_sigma_ref_prior() {
        let cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        let decay = (-std::f64::consts::LN_2 / cfg.volatility.ewma_half_life_seconds).exp();
        let vol = estimate_volatility(157.5, &flat_bars(&[157.5]), &cfg).unwrap();
        let expected_raw = (decay * cfg.volatility.sigma_ref.powi(2))
            .max(cfg.volatility.variance_floor)
            .sqrt();
        assert!((vol.sigma_raw - expected_raw).abs() < 1e-12);
    }

    #[test]
    fn seed_prior_override_replaces_sigma_ref_for_single_bar() {
        let cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        let prior = 8e-8;
        let decay = (-std::f64::consts::LN_2 / cfg.volatility.ewma_half_life_seconds).exp();
        let state =
            estimate_runtime_state_with_seed_prior(157.5, &flat_bars(&[157.5]), &cfg, Some(prior))
                .unwrap();
        let expected_raw = (decay * prior).max(cfg.volatility.variance_floor).sqrt();
        assert!((state.volatility.sigma_raw - expected_raw).abs() < 1e-12);
    }

    #[test]
    fn seed_prior_override_replaces_static_prior_in_blend() {
        let cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        let prior = 7.2e-8;
        let bars = flat_bars(&[157.5; 25]);
        let n = (bars.len() - 1) as f64;
        let seed_var = ((SEED_PRIOR_WEIGHT * prior) / (n + SEED_PRIOR_WEIGHT))
            .max(cfg.volatility.variance_floor);

        let mut expected = RuntimeState {
            last_price: Some(bars[0].close),
            variance: Some(seed_var),
            ..RuntimeState::default()
        };
        for bar in &bars[1..] {
            expected = update_runtime_state_with_seed_prior(
                &expected,
                bar.close,
                Some(bar),
                &cfg,
                Some(prior),
            )
            .unwrap();
        }
        let actual =
            estimate_runtime_state_with_seed_prior(157.5, &bars, &cfg, Some(prior)).unwrap();
        assert!((actual.variance.unwrap() - expected.variance.unwrap()).abs() < 1e-15);
        assert!((actual.volatility.sigma_raw - expected.volatility.sigma_raw).abs() < 1e-12);
    }

    #[test]
    fn quiet_restart_history_seeds_below_sigma_ref() {
        let cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        let bars = flat_bars(&vec![157.5; 25]);
        let vol = estimate_volatility(157.5, &bars, &cfg).unwrap();
        assert!(vol.sigma_raw < cfg.volatility.sigma_ref * 0.7);
    }

    #[test]
    fn zero_variance_history_is_floored_at_variance_floor() {
        let mut cfg = stock_grid_config_for_ticker("NVDA", 3, Vec::new());
        cfg.volatility.variance_floor = 1e-8;
        let bars = flat_bars(&vec![157.5; 250]);
        let state = estimate_runtime_state(157.5, &bars, &cfg).unwrap();
        assert!((state.variance.unwrap() - cfg.volatility.variance_floor).abs() < 1e-15);
        assert!(
            (state.volatility.sigma_raw - cfg.volatility.variance_floor.sqrt()).abs() < 1e-12
        );
    }

    #[test]
    fn seed_blends_realized_with_prior_by_sample_count() {
        let cfg = stock_grid_config_for_ticker("TSLA", 8, Vec::new());
        let prices = [320.0, 320.1, 320.0, 320.15];
        let bars = flat_bars(&prices);
        let n = (bars.len() - 1) as f64;
        let sum_sq: f64 = prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln().powi(2))
            .sum();
        let realized = sum_sq / n;
        let seed_var = ((n * realized + SEED_PRIOR_WEIGHT * cfg.volatility.sigma_ref.powi(2))
            / (n + SEED_PRIOR_WEIGHT))
        .max(cfg.volatility.variance_floor);

        let mut expected = RuntimeState {
            last_price: Some(bars[0].close),
            variance: Some(seed_var),
            ..RuntimeState::default()
        };
        for bar in &bars[1..] {
            expected = update_runtime_state(&expected, bar.close, Some(bar), &cfg).unwrap();
        }
        let actual = estimate_runtime_state(320.15, &bars, &cfg).unwrap();
        assert!((actual.variance.unwrap() - expected.variance.unwrap()).abs() < 1e-15);
        assert!((actual.volatility.sigma_raw - expected.volatility.sigma_raw).abs() < 1e-12);
    }
}
