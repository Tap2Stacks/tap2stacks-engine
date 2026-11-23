use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandleWarmup {
    pub candle_count: usize,
    pub last_close: f64,
    pub variance_per_second: f64,
    pub sigma_per_second: f64,
}

#[derive(Debug, Deserialize)]
pub struct HlCandle {
    #[serde(rename = "t")]
    pub open_time_ms: u64,
    #[serde(rename = "c")]
    pub close: String,
}

pub fn parse_candle_snapshot(body: &str) -> Result<Vec<HlCandle>, serde_json::Error> {
    serde_json::from_str(body)
}

/// Estimates per-second variance from 1m close-to-close log returns using population standard deviation.
pub fn estimate_candle_warmup(candles: &[HlCandle]) -> Option<CandleWarmup> {
    let mut closes: Vec<(u64, f64)> = candles
        .iter()
        .filter_map(|candle| {
            let close = candle.close.parse::<f64>().ok()?;
            (close.is_finite() && close > 0.0).then_some((candle.open_time_ms, close))
        })
        .collect();
    closes.sort_unstable_by_key(|entry| entry.0);
    closes.dedup_by_key(|entry| entry.0);

    if closes.len() < 2 {
        return None;
    }

    let returns: Vec<f64> = closes
        .windows(2)
        .map(|pair| (pair[1].1 / pair[0].1).ln())
        .collect();
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let minute_variance = returns
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let variance_per_second = minute_variance / 60.0;
    if !variance_per_second.is_finite() || variance_per_second <= 0.0 {
        return None;
    }

    Some(CandleWarmup {
        candle_count: closes.len(),
        last_close: closes[closes.len() - 1].1,
        variance_per_second,
        sigma_per_second: variance_per_second.sqrt(),
    })
}

#[derive(Debug)]
pub enum WarmupError {
    Http(reqwest::Error),
    Decode(serde_json::Error),
    NoUsableData,
}

impl Display for WarmupError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(err) => write!(f, "hyperliquid candle request failed: {err}"),
            Self::Decode(err) => write!(f, "hyperliquid candle decode failed: {err}"),
            Self::NoUsableData => write!(
                f,
                "hyperliquid candle response had no usable positive-variance data"
            ),
        }
    }
}

impl Error for WarmupError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Http(err) => Some(err),
            Self::Decode(err) => Some(err),
            Self::NoUsableData => None,
        }
    }
}

pub async fn fetch_candle_warmup(
    client: &reqwest::Client,
    info_url: &str,
    coin: &str,
    minutes: u64,
    end_time_ms: u64,
) -> Result<CandleWarmup, WarmupError> {
    if minutes == 0 {
        return Err(WarmupError::NoUsableData);
    }
    let start_time_ms = end_time_ms.saturating_sub(minutes.saturating_mul(60_000));
    let request = json!({
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": "1m",
            "startTime": start_time_ms,
            "endTime": end_time_ms,
        }
    });

    let body = client
        .post(info_url)
        .json(&request)
        .send()
        .await
        .map_err(WarmupError::Http)?
        .error_for_status()
        .map_err(WarmupError::Http)?
        .text()
        .await
        .map_err(WarmupError::Http)?;
    let candles = parse_candle_snapshot(&body).map_err(WarmupError::Decode)?;
    estimate_candle_warmup(&candles).ok_or(WarmupError::NoUsableData)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_AAPL_CANDLES: &str =
        include_str!("../../tests/fixtures/hl_candle_snapshot_aapl.json");

    fn candle(open_time_ms: u64, close: &str) -> HlCandle {
        HlCandle {
            open_time_ms,
            close: close.to_string(),
        }
    }

    #[test]
    fn parses_real_hyperliquid_candle_rows() {
        let candles = parse_candle_snapshot(REAL_AAPL_CANDLES).unwrap();
        assert_eq!(candles.len(), 13);
        assert_eq!(candles[0].open_time_ms, 1_795_412_280_000);
        assert_eq!(candles[0].close, "347.44");
    }

    #[test]
    fn estimates_per_second_variance_from_real_rows() {
        let candles = parse_candle_snapshot(REAL_AAPL_CANDLES).unwrap();
        let estimate = estimate_candle_warmup(&candles).unwrap();
        assert_eq!(estimate.candle_count, 13);
        assert!((estimate.last_close - 347.89).abs() < 1e-12);
        assert!((estimate.variance_per_second - 2.83795315302e-9).abs() < 1e-15);
        assert!((estimate.sigma_per_second - 5.32724427168e-5).abs() < 1e-12);
    }

    #[test]
    fn rejects_empty_flat_or_unusable_closes() {
        assert!(estimate_candle_warmup(&[]).is_none());
        assert!(estimate_candle_warmup(&[candle(1, "100.0"), candle(2, "100.0")]).is_none());
        assert!(estimate_candle_warmup(&[candle(1, "100.0"), candle(2, "bad")]).is_none());
    }
}
