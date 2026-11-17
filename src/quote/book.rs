use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Unique identifier for a quoted price-band and time-window cell.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CellKey {
    pub window_start_ms: u64,
    pub lower_cents: i64,
}

/// Live exposure map keyed by `CellKey`, replaced atomically per update.
#[derive(Debug, Clone, Default)]
pub struct ExposureBook {
    by_cell: HashMap<CellKey, f64>,
}

impl ExposureBook {
    /// Replaces the entire exposure map with the provided non-negative finite entries.
    pub fn replace(&mut self, cells: Vec<(CellKey, f64)>) {
        self.by_cell.clear();
        for (key, value) in cells {
            if value.is_finite() && value > 0.0 {
                self.by_cell.insert(key, value);
            }
        }
    }

    /// Returns the live exposure for `key`, or `0.0` if absent.
    pub fn get(&self, key: &CellKey) -> f64 {
        self.by_cell.get(key).copied().unwrap_or(0.0)
    }
}
