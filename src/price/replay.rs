use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;

use super::deribit::{
    DeribitSampleEvent, DeribitSampler, deribit_index_name, parse_deribit_index_value,
    parse_deribit_ticker_value,
};
use super::hyperliquid::parse_active_asset_ctx_full;
use super::types::{HlTick, PriceSource, PriceTick};

#[derive(Debug, Deserialize)]
struct RecordedLine {
    rx: u64,
    ch: String,
    data: serde_json::Value,
}

/// Parses recorded Hyperliquid JSONL `activeAssetCtx` ticks for `coin` from any `BufRead`,
/// preserving optional `midPx` and `oraclePx` alongside `markPx` (`price`).
pub fn read_hl_jsonl_ctx_reader<R: BufRead>(reader: R, coin: &str) -> io::Result<Vec<HlTick>> {
    let mut ticks = Vec::new();

    for line_res in reader.lines() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let record: RecordedLine = serde_json::from_str(trimmed)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

        if record.ch != "activeAssetCtx" {
            continue;
        }

        if let Some(ctx) = parse_active_asset_ctx_full(record.data) {
            if ctx.coin == coin {
                ticks.push(HlTick {
                    ts_ms: record.rx,
                    price: ctx.mark_px,
                    mid_px: ctx.mid_px,
                    oracle_px: ctx.oracle_px,
                });
            }
        }
    }

    Ok(ticks)
}

/// Reads recorded Hyperliquid JSONL ticks for `coin` carrying optional `midPx` and `oraclePx`,
/// using `rx` as `ts_ms` and skipping non-`activeAssetCtx` lines or other coins.
pub fn read_hl_jsonl_ctx(path: &Path, coin: &str) -> io::Result<Vec<HlTick>> {
    let file = File::open(path)?;
    read_hl_jsonl_ctx_reader(BufReader::new(file), coin)
}

/// Reads recorded Hyperliquid JSONL ticks for `coin`, using `rx` as `ts_ms`, populating
/// optional `mid_px` / `oracle_px`, and skipping non-`activeAssetCtx` lines or other coins.
pub fn read_hl_jsonl(path: &Path, coin: &str) -> io::Result<Vec<PriceTick>> {
    read_hl_jsonl_ctx(path, coin)
}

/// Parses recorded Deribit JSONL ticks for `coin` from any `BufRead` using the shared
/// `DeribitSampler` rule (matching the live WebSocket feed):
/// - Index changes from `deribit_price_index.<sym>_usdc` or `ticker.<SYM>_USDC-PERPETUAL.*`
///   emit immediately.
/// - Any recorded message of any channel proves socket liveness at its timestamp and emits a
///   carried unchanged sample when `>= 1 s` has elapsed since the coin's last emitted sample.
/// - Attaches the latest perp book midpoint (`mid_px`) and `mark_price` (`oracle_px`) from
///   the coin's ticker messages (`None` until first seen).
pub fn read_drb_jsonl_reader<R: BufRead>(reader: R, coin: &str) -> io::Result<Vec<PriceTick>> {
    let target_index = deribit_index_name(coin);
    let mut events = Vec::new();

    for line_res in reader.lines() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let record: RecordedLine = serde_json::from_str(trimmed)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

        let event = if record.ch.starts_with("deribit_price_index.") {
            match parse_deribit_index_value(record.data) {
                Some(idx) => {
                    let ts_ms = if idx.timestamp > 0 {
                        idx.timestamp
                    } else {
                        record.rx
                    };
                    DeribitSampleEvent::IndexPrint {
                        index_name: idx.index_name,
                        price: idx.price,
                        ts_ms,
                    }
                }
                None => DeribitSampleEvent::Liveness { ts_ms: record.rx },
            }
        } else if record.ch.starts_with("ticker.") {
            match parse_deribit_ticker_value(&record.ch, record.data) {
                Some(ticker) => {
                    let ts_ms = if ticker.timestamp > 0 {
                        ticker.timestamp
                    } else {
                        record.rx
                    };
                    match ticker.index_name() {
                        Some(index_name) => DeribitSampleEvent::Ticker {
                            index_name,
                            index_price: ticker.index_price,
                            mid_px: ticker.mid_price(),
                            mark_price: ticker.mark_price,
                            ts_ms,
                        },
                        None => DeribitSampleEvent::Liveness { ts_ms },
                    }
                }
                None => DeribitSampleEvent::Liveness { ts_ms: record.rx },
            }
        } else if let Some(ts) = record
            .data
            .as_array()
            .and_then(|arr| arr.last())
            .and_then(|item| item.get("timestamp"))
            .and_then(|val| val.as_u64())
            .filter(|&ts| ts > 0)
        {
            DeribitSampleEvent::Liveness { ts_ms: ts }
        } else {
            DeribitSampleEvent::Liveness { ts_ms: record.rx }
        };

        events.push(event);
    }

    events.sort_by_key(DeribitSampleEvent::ts_ms);

    let mut sampler = DeribitSampler::new([target_index]);
    let mut ticks = Vec::new();
    for event in &events {
        for (_, tick) in sampler.on_event(event) {
            ticks.push(tick);
        }
    }

    Ok(ticks)
}

/// Reads recorded Deribit JSONL index ticks (`deribit_price_index.<sym>_usdc`) for `coin`.
pub fn read_drb_jsonl(path: &Path, coin: &str) -> io::Result<Vec<PriceTick>> {
    let file = File::open(path)?;
    read_drb_jsonl_reader(BufReader::new(file), coin)
}

/// Dispatches JSONL tick reading by `PriceSource` (`Hyperliquid` or `Deribit`).
pub fn read_ticks_for_source(
    path: &Path,
    coin: &str,
    source: PriceSource,
) -> io::Result<Vec<PriceTick>> {
    match source {
        PriceSource::Hyperliquid => read_hl_jsonl(path, coin),
        PriceSource::Deribit => read_drb_jsonl(path, coin),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::path::Path;

    use super::*;

    #[test]
    fn replay_reads_only_mark_lines_in_order() {
        let ticks = read_hl_jsonl(Path::new("tests/fixtures/hl_sample.jsonl"), "xyz:NVDA").unwrap();
        assert!(!ticks.is_empty());
        assert!(ticks.windows(2).all(|w| w[0].ts_ms <= w[1].ts_ms));
    }

    #[test]
    fn replay_ctx_reads_optional_mid_and_oracle_prices() {
        let sample =
            read_hl_jsonl_ctx(Path::new("tests/fixtures/hl_sample.jsonl"), "xyz:NVDA").unwrap();
        assert!(!sample.is_empty());
        assert_eq!(sample[0].price, 157.53);
        assert_eq!(sample[0].mid_px, Some(157.57));
        assert_eq!(sample[0].oracle_px, Some(157.44));

        let jsonl = concat!(
            r#"{"rx":1000,"ch":"activeAssetCtx","data":{"coin":"xyz:NVDA","ctx":{"oraclePx":"145.88","markPx":"145.98","midPx":"146.005"}}}"#, "\n",
            r#"{"rx":2000,"ch":"activeAssetCtx","data":{"coin":"xyz:NVDA","ctx":{"markPx":"146.01"}}}"#, "\n",
            r#"{"rx":3000,"ch":"activeAssetCtx","data":{"coin":"xyz:NVDA","ctx":{"markPx":"146.02","midPx":null,"oraclePx":"145.90"}}}"#, "\n",
            r#"{"rx":4000,"ch":"bbo","data":{"coin":"xyz:NVDA"}}"#, "\n",
            r#"{"rx":5000,"ch":"activeAssetCtx","data":{"coin":"xyz:TSLA","ctx":{"markPx":"380.0","midPx":"380.1"}}}"#, "\n",
        );
        let parsed = read_hl_jsonl_ctx_reader(Cursor::new(jsonl), "xyz:NVDA").unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(
            parsed[0],
            HlTick {
                ts_ms: 1000,
                price: 145.98,
                mid_px: Some(146.005),
                oracle_px: Some(145.88),
            }
        );
        assert_eq!(
            parsed[1],
            HlTick {
                ts_ms: 2000,
                price: 146.01,
                mid_px: None,
                oracle_px: None,
            }
        );
        assert_eq!(
            parsed[2],
            HlTick {
                ts_ms: 3000,
                price: 146.02,
                mid_px: None,
                oracle_px: Some(145.90),
            }
        );
    }

    #[test]
    fn replay_drb_reads_index_and_attaches_mid_and_mark_from_ticker() {
        let jsonl = concat!(
            r#"{"rx":1795450000200,"ch":"deribit_price_index.nvda_usdc","data":{"index_name":"nvda_usdc","price":146.12,"timestamp":1795450000123}}"#, "\n",
            r#"{"rx":1795450000250,"ch":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","mark_price":146.15,"index_price":146.12,"best_bid_price":146.10,"best_ask_price":146.20,"last_price":146.14,"timestamp":1795450000150}}"#, "\n",
            r#"{"rx":1795450000300,"ch":"trades.NVDA_USDC-PERPETUAL.100ms","data":[{"price":146.14,"amount":5.0,"timestamp":1795450000280}]}"#, "\n",
            r#"{"rx":1795450000400,"ch":"deribit_price_index.tsla_usdc","data":{"index_name":"tsla_usdc","price":379.45,"timestamp":1795450000390}}"#, "\n",
            r#"{"rx":1795450001200,"ch":"deribit_price_index.nvda_usdc","data":{"index_name":"nvda_usdc","price":146.15,"timestamp":1795450001120}}"#, "\n",
        );
        let nvda = read_drb_jsonl_reader(Cursor::new(jsonl), "NVDA").unwrap();
        assert_eq!(nvda.len(), 2);
        assert_eq!(nvda[0], PriceTick::new(1_795_450_000_123, 146.12));
        assert_eq!(nvda[1].ts_ms, 1_795_450_001_120);
        assert_eq!(nvda[1].price, 146.15);
        assert!((nvda[1].mid_px.unwrap() - 146.15).abs() < 1e-12);
        assert_eq!(nvda[1].oracle_px, Some(146.15));
        // Also accepts lowercase index name or xyz: prefix.
        assert_eq!(
            read_drb_jsonl_reader(Cursor::new(jsonl), "nvda_usdc").unwrap(),
            nvda
        );
        assert_eq!(
            read_drb_jsonl_reader(Cursor::new(jsonl), "xyz:TSLA").unwrap(),
            vec![PriceTick::new(1_795_450_000_390, 379.45)]
        );
    }

    #[test]
    fn replay_drb_carries_quiet_seconds_preserves_5s_hole_and_attaches_mid_mark() {
        let jsonl = concat!(
            // t=0.1s: initial NVDA index print (no ticker seen yet -> mid/oracle None)
            r#"{"rx":1795450000120,"ch":"deribit_price_index.nvda_usdc","data":{"index_name":"nvda_usdc","price":146.12,"timestamp":1795450000100}}"#, "\n",
            // t=0.3s: NVDA ticker arrives with same index_price -> updates mid/mark, no flood
            r#"{"rx":1795450000320,"ch":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","mark_price":146.16,"index_price":146.12,"best_bid_price":146.10,"best_ask_price":146.20,"timestamp":1795450000300}}"#, "\n",
            // t=1.15s: NVDA index quiet, TSLA print proves socket alive -> NVDA gets 1 Hz carried sample
            r#"{"rx":1795450001160,"ch":"deribit_price_index.tsla_usdc","data":{"index_name":"tsla_usdc","price":379.45,"timestamp":1795450001150}}"#, "\n",
            // t=2.20s: trades message on another channel proves socket alive -> NVDA gets 1 Hz carried sample
            r#"{"rx":1795450002210,"ch":"trades.TSLA_USDC-PERPETUAL.100ms","data":[{"price":379.50,"amount":2.0,"timestamp":1795450002200}]}"#, "\n",
            // 5-second recording hole (t=2.20s -> t=7.30s): NO lines recorded, so NO carried samples
            // t=7.30s: NVDA ticker resumes with changed index_price -> immediate sample after 5.1s gap
            r#"{"rx":1795450007310,"ch":"ticker.NVDA_USDC-PERPETUAL.100ms","data":{"instrument_name":"NVDA_USDC-PERPETUAL","mark_price":146.22,"index_price":146.19,"best_bid_price":146.18,"best_ask_price":146.24,"timestamp":1795450007300}}"#, "\n",
        );

        let nvda = read_drb_jsonl_reader(Cursor::new(jsonl), "NVDA").unwrap();
        assert_eq!(nvda.len(), 4);
        assert_eq!(nvda[0], PriceTick::new(1_795_450_000_100, 146.12));

        assert_eq!(nvda[1].ts_ms, 1_795_450_001_150);
        assert_eq!(nvda[1].price, 146.12);
        assert!((nvda[1].mid_px.unwrap() - 146.15).abs() < 1e-12);
        assert_eq!(nvda[1].oracle_px, Some(146.16));

        assert_eq!(nvda[2].ts_ms, 1_795_450_002_200);
        assert_eq!(nvda[2].price, 146.12);
        assert!((nvda[2].mid_px.unwrap() - 146.15).abs() < 1e-12);
        assert_eq!(nvda[2].oracle_px, Some(146.16));

        assert_eq!(nvda[3].ts_ms, 1_795_450_007_300);
        assert_eq!(nvda[3].price, 146.19);
        assert!((nvda[3].mid_px.unwrap() - 146.21).abs() < 1e-12);
        assert_eq!(nvda[3].oracle_px, Some(146.22));

        // Verify the 5 s recording hole produces a > 3 s gap while quiet seconds stay <= 3 s.
        assert!(nvda[1].ts_ms - nvda[0].ts_ms <= 3_000);
        assert!(nvda[2].ts_ms - nvda[1].ts_ms <= 3_000);
        assert_eq!(nvda[3].ts_ms - nvda[2].ts_ms, 5_100);
    }

    #[test]
    fn default_source_is_hyperliquid_with_identical_output() {
        assert_eq!(PriceSource::default(), PriceSource::Hyperliquid);
        let path = Path::new("tests/fixtures/hl_sample.jsonl");
        let direct = read_hl_jsonl(path, "xyz:NVDA").unwrap();
        let via_default = read_ticks_for_source(path, "xyz:NVDA", PriceSource::default()).unwrap();
        assert_eq!(via_default, direct);
    }
}
