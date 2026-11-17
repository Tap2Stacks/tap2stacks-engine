use serde::{Deserialize, Serialize};

use crate::quote::StockQuote;

/// Outbound WebSocket frames sent from the engine to the backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum EngineOut {
    Quote(StockQuote),
    Sample {
        ticker: String,
        ts_ms: u64,
        price: f64,
    },
    Health {
        ticker: String,
        server_ts_ms: u64,
        last_sample_age_ms: Option<u64>,
        stale: bool,
        prev_day_px: Option<f64>,
    },
}

/// Inbound WebSocket frames sent from the backend to the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum EngineIn {
    Exposure {
        ticker: String,
        cells: Vec<ExposureCell>,
    },
}

/// Single cell exposure liability entry in an `EngineIn::Exposure` replacement frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExposureCell {
    pub window_start_ms: u64,
    pub lower_cents: i64,
    pub liability: f64,
}
