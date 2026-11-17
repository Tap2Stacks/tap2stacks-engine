use crate::grid::GridSnapshot;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct AppMarketView {
    pub latest_price: Option<f64>,
    pub latest_price_ts_ms: Option<u64>,
    pub latest_price_server_ts_ms: Option<u64>,
    pub completed_bar_count: usize,
    pub grid: Option<GridSnapshot>,
}
