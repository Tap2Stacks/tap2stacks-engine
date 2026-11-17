use super::types::{GridConfig, GridError, OhlcBar, RuntimeState, VolatilitySnapshot};

pub fn estimate_volatility(
    spot: f64,
    history: &[OhlcBar],
    config: &GridConfig,
) -> Result<VolatilitySnapshot, GridError> {
    if spot <= 0.0 {
        return Err(GridError::InvalidSpot);
    }
    if history.is_empty() {
        return Err(GridError::InvalidHistory(
            "at least one OHLC bar is required",
        ));
    }

    let mut state = RuntimeState::default();

    for bar in history {
        validate_bar(*bar)?;
        state = update_runtime_state(&state, bar.close, Some(bar), config)?;
    }

    Ok(reprice_sigma_for_spot(state.volatility, spot, config))
}

pub fn estimate_runtime_state(
    spot: f64,
    history: &[OhlcBar],
    config: &GridConfig,
) -> Result<RuntimeState, GridError> {
    if spot <= 0.0 {
        return Err(GridError::InvalidSpot);
    }
    if history.is_empty() {
        return Err(GridError::InvalidHistory(
            "at least one OHLC bar is required",
        ));
    }

    let mut state = RuntimeState::default();
    for bar in history {
        validate_bar(*bar)?;
        state = update_runtime_state(&state, bar.close, Some(bar), config)?;
    }

    state.volatility = reprice_sigma_for_spot(state.volatility, spot, config);
    Ok(state)
}

pub fn update_runtime_state(
    previous_state: &RuntimeState,
    price: f64,
    bar: Option<&OhlcBar>,
    config: &GridConfig,
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
        .unwrap_or(config.volatility.sigma_ref.powi(2));
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
