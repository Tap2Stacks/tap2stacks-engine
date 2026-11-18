use crate::price::PriceTick;

pub const VOID_GAP_MS: u64 = 3_000;
pub const RESTART_GAP_MS: u64 = 30_000;
pub const WARMUP_MS: u64 = 300_000;

pub fn collect_gaps_over_3s(ticks: &[PriceTick]) -> Vec<(u64, u64)> {
    ticks
        .windows(2)
        .filter_map(|pair| {
            let gap = pair[1].ts_ms.saturating_sub(pair[0].ts_ms);
            if gap > VOID_GAP_MS {
                Some((pair[0].ts_ms, pair[1].ts_ms))
            } else {
                None
            }
        })
        .collect()
}

pub fn window_is_void(
    ticks: &[PriceTick],
    start_hint: usize,
    window_start_ms: u64,
    window_end_ms: u64,
) -> bool {
    let mut idx = start_hint;
    while idx < ticks.len() && ticks[idx].ts_ms < window_start_ms {
        idx += 1;
    }
    if idx >= ticks.len() || ticks[idx].ts_ms >= window_end_ms {
        return true;
    }
    while idx < ticks.len() {
        let curr_ts = ticks[idx].ts_ms;
        if idx > 0 {
            let prev_ts = ticks[idx - 1].ts_ms;
            if prev_ts < window_end_ms
                && curr_ts >= window_start_ms
                && curr_ts.saturating_sub(prev_ts) > VOID_GAP_MS
            {
                return true;
            }
        }
        if curr_ts >= window_end_ms {
            break;
        }
        idx += 1;
    }
    false
}

pub fn window_hit_bands(
    ticks: &[PriceTick],
    start_hint: usize,
    window_start_ms: u64,
    window_end_ms: u64,
    bands_cents: &[(i64, i64)],
) -> Vec<bool> {
    let mut hits = vec![false; bands_cents.len()];
    let mut idx = start_hint;
    while idx < ticks.len() && ticks[idx].ts_ms < window_start_ms {
        idx += 1;
    }
    while idx < ticks.len() && ticks[idx].ts_ms < window_end_ms {
        let curr = ticks[idx];
        let curr_cents = (curr.price * 100.0).round() as i64;
        let prev_cents =
            if idx > 0 && curr.ts_ms.saturating_sub(ticks[idx - 1].ts_ms) <= VOID_GAP_MS {
                (ticks[idx - 1].price * 100.0).round() as i64
            } else {
                curr_cents
            };
        let lo = prev_cents.min(curr_cents);
        let hi = prev_cents.max(curr_cents);
        for (cell_idx, &(lower_c, upper_c)) in bands_cents.iter().enumerate() {
            if !hits[cell_idx] && lo <= upper_c && hi >= lower_c {
                hits[cell_idx] = true;
            }
        }
        idx += 1;
    }
    hits
}
