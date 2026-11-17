use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;

use super::hyperliquid::parse_active_asset_ctx_value;
use super::types::PriceTick;

#[derive(Debug, Deserialize)]
struct RecordedLine {
    rx: u64,
    ch: String,
    data: serde_json::Value,
}

/// Reads recorded Hyperliquid JSONL ticks for `coin`, using `rx` as `ts_ms` and
/// skipping non-`activeAssetCtx` lines or other coins.
pub fn read_hl_jsonl(path: &Path, coin: &str) -> io::Result<Vec<PriceTick>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
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

        if let Some((parsed_coin, price)) = parse_active_asset_ctx_value(record.data) {
            if parsed_coin == coin {
                ticks.push(PriceTick {
                    ts_ms: record.rx,
                    price,
                });
            }
        }
    }

    Ok(ticks)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn replay_reads_only_mark_lines_in_order() {
        let ticks = read_hl_jsonl(Path::new("tests/fixtures/hl_sample.jsonl"), "xyz:NVDA").unwrap();
        assert!(!ticks.is_empty());
        assert!(ticks.windows(2).all(|w| w[0].ts_ms <= w[1].ts_ms));
    }
}
