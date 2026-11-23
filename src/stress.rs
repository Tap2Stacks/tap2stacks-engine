use rand::Rng;

use crate::calibrate::eval::{collect_gaps_over_3s, window_is_void};
use crate::price::PriceTick;

pub const POLICIES: [&str; 8] = [
    "closest",
    "three_rows",
    "row_sweep",
    "long_bias",
    "short_bias",
    "random_near",
    "momentum",
    "chase_far",
];

pub const QUOTE_LEAD_MS: u64 = 5_000;
pub const WINDOW_MS: u64 = 5_000;
pub const SHOCK_HORIZONS_MS: [u64; 3] = [10_000, 60_000, 300_000];
pub const SIGMA_LAG_OFFSETS_MS: [u64; 6] = [0, 10_000, 30_000, 60_000, 120_000, 300_000];

const FAR_MULT_MIN: f64 = 10.0;
const FAR_MULT_MAX: f64 = 25.0;

#[derive(Debug, Clone, Copy)]
pub struct CellTap {
    pub quote_ts_ms: u64,
    pub start_ms: u64,
    pub window_index: usize,
    pub signed_dist: i64,
    pub dist: i64,
    pub multiplier: f64,
    pub hit: bool,
    pub us: bool,
    pub move_sign: i8,
    pub sigma: f64,
    pub sigma_raw: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct WindowQuote {
    pub start_ms: u64,
    pub sigma: f64,
    pub sigma_raw: f64,
    pub max_mult: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct StakeStats {
    pub edge_pct: f64,
    pub payout_ratio: f64,
    pub loss_per_1000: f64,
    pub house_per_1000: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Period {
    pub name: String,
    pub from_ms: u64,
    pub to_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GapAudit {
    pub start_ms: u64,
    pub end_ms: u64,
    pub gap_ms: u64,
    pub overlapping_windows: usize,
    pub voided_windows: usize,
}

pub fn parse_utc(text: &str) -> Result<u64, String> {
    if !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()) {
        return text.parse::<u64>().map_err(|_| format!("bad timestamp {text}"));
    }
    let bytes = text.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return Err(format!("bad timestamp {text}"));
    }
    let year: i32 = parse_num(&text[0..4])?;
    let month: u32 = parse_num(&text[5..7])?;
    let day: u32 = parse_num(&text[8..10])?;
    let hour: u64 = parse_num(&text[11..13])?;
    let minute: u64 = parse_num(&text[14..16])?;
    let second: u64 = parse_num(&text[17..19])?;
    if month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || minute > 59 || second > 59 {
        return Err(format!("bad timestamp {text}"));
    }
    let days = days_from_civil(year, month, day);
    if days < 0 {
        return Err(format!("bad timestamp {text}"));
    }
    Ok((days as u64) * 86_400_000 + (hour * 3600 + minute * 60 + second) * 1000)
}

pub fn parse_period(text: &str) -> Result<Period, String> {
    let mut parts = text.split(',');
    let name = parts.next().unwrap_or("").trim();
    let from = parse_utc(parts.next().unwrap_or("").trim())?;
    let to = parse_utc(parts.next().unwrap_or("").trim())?;
    if name.is_empty() || from >= to || parts.next().is_some() {
        return Err(format!("bad period {text}"));
    }
    Ok(Period {
        name: name.to_string(),
        from_ms: from,
        to_ms: to,
    })
}

pub fn format_utc(ms: u64) -> String {
    let secs = ms / 1000;
    let days = (secs / 86_400) as i64;
    let sod = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = sod / 3600;
    let minute = (sod % 3600) / 60;
    let second = sod % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

pub fn price_at_or_before(ticks: &[PriceTick], ts_ms: u64) -> Option<f64> {
    let idx = ticks.partition_point(|tick| tick.ts_ms <= ts_ms);
    if idx == 0 {
        None
    } else {
        Some(ticks[idx - 1].price)
    }
}

pub fn price_at_or_after(ticks: &[PriceTick], ts_ms: u64) -> Option<(u64, f64)> {
    let idx = ticks.partition_point(|tick| tick.ts_ms < ts_ms);
    ticks.get(idx).map(|tick| (tick.ts_ms, tick.price))
}

pub fn move_sign(ticks: &[PriceTick], quote_ms: u64, lookback_ms: u64) -> i8 {
    let Some(now) = price_at_or_before(ticks, quote_ms) else {
        return 0;
    };
    let Some(prev) = price_at_or_before(ticks, quote_ms.saturating_sub(lookback_ms)) else {
        return 0;
    };
    if now > prev {
        1
    } else if now < prev {
        -1
    } else {
        0
    }
}

pub fn choose_taps(policy: &str, taps: &[CellTap], count: usize, rng: &mut impl Rng) -> Vec<CellTap> {
    let mut pool = taps.to_vec();
    match policy {
        "closest" => pool.sort_by_key(|tap| (tap.dist, tap.start_ms)),
        "three_rows" => {
            let nearest = pool.iter().map(|tap| tap.start_ms).min().unwrap_or(0);
            pool.retain(|tap| tap.start_ms == nearest);
            pool.sort_by_key(|tap| tap.dist);
        }
        "row_sweep" => {
            let best = pool.iter().map(|tap| tap.dist).min().unwrap_or(0);
            pool.retain(|tap| tap.dist == best);
            pool.sort_by_key(|tap| tap.start_ms);
        }
        "long_bias" => {
            pool.retain(|tap| tap.signed_dist > 0);
            pool.sort_by_key(|tap| (tap.dist, tap.start_ms));
        }
        "short_bias" => {
            pool.retain(|tap| tap.signed_dist < 0);
            pool.sort_by_key(|tap| (tap.dist, tap.start_ms));
        }
        "random_near" => {
            let mut picked = Vec::new();
            while picked.len() < count && !pool.is_empty() {
                let weights: Vec<f64> = pool.iter().map(|tap| 1.0 / (1.0 + tap.dist as f64)).collect();
                let total: f64 = weights.iter().sum();
                let mut draw = rng.random::<f64>() * total;
                let mut idx = pool.len() - 1;
                for (i, weight) in weights.iter().enumerate() {
                    draw -= *weight;
                    if draw <= 0.0 {
                        idx = i;
                        break;
                    }
                }
                picked.push(pool.remove(idx));
            }
            return picked;
        }
        "momentum" => {
            pool.retain(|tap| tap.move_sign != 0 && tap.signed_dist.signum() as i8 == tap.move_sign);
            pool.sort_by_key(|tap| (tap.dist, tap.start_ms));
        }
        "chase_far" => {
            pool.retain(|tap| {
                tap.move_sign != 0
                    && tap.signed_dist.signum() as i8 == tap.move_sign
                    && in_far_band(tap.multiplier)
            });
            pool.sort_by(|left, right| {
                right
                    .dist
                    .cmp(&left.dist)
                    .then(right.multiplier.total_cmp(&left.multiplier))
                    .then(left.start_ms.cmp(&right.start_ms))
            });
        }
        _ => {}
    }
    pool.truncate(count);
    pool
}

pub fn normal_session_starts(
    first_tap_ms: u64,
    last_tap_ms: u64,
    from_ms: u64,
    to_ms: u64,
    watch_ms: u64,
    burst_ms: u64,
    return_min_ms: u64,
    return_max_ms: u64,
    cap: usize,
    rng: &mut impl Rng,
) -> Vec<u64> {
    if return_min_ms >= return_max_ms || to_ms <= from_ms {
        return Vec::new();
    }
    let mut start = first_tap_ms.saturating_add(watch_ms);
    let mut included = 0usize;
    let mut out = Vec::new();
    while start.saturating_add(burst_ms) < last_tap_ms {
        if start >= to_ms {
            break;
        }
        if start >= from_ms {
            if cap > 0 && included >= cap {
                break;
            }
            out.push(start);
            included += 1;
            if cap > 0 && included >= cap {
                break;
            }
        }
        start = start.saturating_add(rng.random_range(return_min_ms..return_max_ms));
    }
    out
}

pub fn dense_session_starts(
    from_ms: u64,
    to_ms: u64,
    step_ms: u64,
    burst_ms: u64,
    last_tap_ms: u64,
) -> Vec<u64> {
    if step_ms == 0 || from_ms >= to_ms {
        return Vec::new();
    }
    let mut start = from_ms;
    let mut out = Vec::new();
    while start < to_ms && start.saturating_add(burst_ms) < last_tap_ms {
        out.push(start);
        start = start.saturating_add(step_ms);
    }
    out
}

pub fn merge_starts(mut starts: Vec<u64>, extra: Vec<u64>) -> Vec<u64> {
    starts.extend(extra);
    starts.sort_unstable();
    starts.dedup();
    starts
}

pub fn stake_stats(volume: f64, payout: f64) -> StakeStats {
    if volume <= 0.0 {
        return StakeStats {
            edge_pct: 0.0,
            payout_ratio: 0.0,
            loss_per_1000: 0.0,
            house_per_1000: 0.0,
        };
    }
    let house = volume - payout;
    StakeStats {
        edge_pct: house / volume * 100.0,
        payout_ratio: payout / volume,
        loss_per_1000: if house < 0.0 { -house / volume * 1000.0 } else { 0.0 },
        house_per_1000: house / volume * 1000.0,
    }
}

pub fn worst_minute(minutes: &[(u64, f64, f64)]) -> Option<(u64, f64)> {
    minutes
        .iter()
        .filter(|(_, volume, _)| *volume > 0.0)
        .map(|(minute, volume, payout)| (*minute, volume - payout))
        .min_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)))
}

pub fn worst_edge_minute(minutes: &[(u64, f64, f64)], min_volume: f64) -> Option<(u64, f64)> {
    minutes
        .iter()
        .filter(|(_, volume, _)| *volume >= min_volume)
        .map(|(minute, volume, payout)| (*minute, (volume - payout) / volume * 100.0))
        .min_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)))
}

pub fn bucket_quote(quotes: &[WindowQuote], from_ms: u64, to_ms: u64) -> (f64, f64, f64, usize) {
    let lo = quotes.partition_point(|quote| quote.start_ms < from_ms);
    let hi = quotes.partition_point(|quote| quote.start_ms < to_ms);
    let slice = &quotes[lo..hi];
    if slice.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }
    let count = slice.len() as f64;
    let sigma = slice.iter().map(|quote| quote.sigma).sum::<f64>() / count;
    let raw = slice.iter().map(|quote| quote.sigma_raw).sum::<f64>() / count;
    let max_mult = slice.iter().map(|quote| quote.max_mult).fold(0.0_f64, f64::max);
    (sigma, raw, max_mult, slice.len())
}

pub fn quoted_sigma_at(quotes: &[WindowQuote], ts_ms: u64, lead_ms: u64) -> Option<(f64, f64)> {
    let limit = ts_ms.saturating_add(lead_ms);
    let idx = quotes.partition_point(|quote| quote.start_ms <= limit);
    if idx == 0 {
        None
    } else {
        Some((quotes[idx - 1].sigma, quotes[idx - 1].sigma_raw))
    }
}

pub fn realized_step_sigma(ticks: &[PriceTick], from_ms: u64, to_ms: u64) -> Option<(f64, usize)> {
    if to_ms <= from_ms {
        return None;
    }
    let mut returns = Vec::new();
    let mut previous: Option<(u64, f64)> = None;
    let mut sec = from_ms / 1000;
    let end_sec = to_ms / 1000;
    while sec < end_sec {
        if let Some(price) = price_in_second(ticks, sec * 1000) {
            if let Some((prev_sec, prev_price)) = previous {
                if sec == prev_sec + 1 && prev_price > 0.0 && price > 0.0 {
                    returns.push((price / prev_price).ln());
                }
            }
            previous = Some((sec, price));
        }
        sec += 1;
    }
    if returns.len() < 2 {
        return None;
    }
    let count = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / count;
    let variance = returns.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (count - 1.0);
    Some((variance.max(0.0).sqrt(), returns.len()))
}

pub fn max_gap_overlapping(ticks: &[PriceTick], from_ms: u64, to_ms: u64) -> u64 {
    ticks.windows(2).fold(0u64, |best, pair| {
        let gap_start = pair[0].ts_ms;
        let gap_end = pair[1].ts_ms;
        if gap_end > from_ms && gap_start < to_ms {
            best.max(gap_end.saturating_sub(gap_start))
        } else {
            best
        }
    })
}

pub fn log_move(ticks: &[PriceTick], from_ms: u64, to_ms: u64) -> Option<f64> {
    let earlier = price_at_or_before(ticks, from_ms)?;
    let later = price_at_or_before(ticks, to_ms)?;
    if earlier <= 0.0 || later <= 0.0 {
        return None;
    }
    Some((later / earlier).ln())
}

pub fn tripwire_k(log_move_abs: f64, quoted_sigma: f64, horizon_secs: f64) -> Option<f64> {
    if quoted_sigma <= 0.0 || horizon_secs <= 0.0 {
        return None;
    }
    Some(log_move_abs / (quoted_sigma * horizon_secs.sqrt()))
}

pub fn audit_feed_gaps(ticks: &[PriceTick], window_width_ms: u64) -> Vec<GapAudit> {
    if window_width_ms == 0 {
        return Vec::new();
    }
    collect_gaps_over_3s(ticks)
        .into_iter()
        .map(|(start, end)| {
            let mut overlapping = 0usize;
            let mut voided = 0usize;
            let mut window_start = start / window_width_ms * window_width_ms;
            if window_start + window_width_ms <= start {
                window_start = window_start.saturating_add(window_width_ms);
            }
            while window_start < end.saturating_add(window_width_ms) && window_start <= end {
                let window_end = window_start.saturating_add(window_width_ms);
                if window_end > start && window_start <= end {
                    overlapping += 1;
                    if window_is_void(ticks, 0, window_start, window_end) {
                        voided += 1;
                    }
                }
                window_start = window_start.saturating_add(window_width_ms);
            }
            GapAudit {
                start_ms: start,
                end_ms: end,
                gap_ms: end.saturating_sub(start),
                overlapping_windows: overlapping,
                voided_windows: voided,
            }
        })
        .collect()
}

pub fn hour_starts(from_ms: u64, to_ms: u64) -> Vec<u64> {
    if to_ms <= from_ms {
        return Vec::new();
    }
    let mut hour = from_ms / 3_600_000 * 3_600_000;
    let mut out = Vec::new();
    while hour < to_ms {
        out.push(hour);
        hour = hour.saturating_add(3_600_000);
    }
    out
}

pub fn minute_starts(from_ms: u64, to_ms: u64) -> Vec<u64> {
    if to_ms <= from_ms {
        return Vec::new();
    }
    let mut minute = from_ms / 60_000 * 60_000;
    if minute < from_ms {
        minute = minute.saturating_add(60_000);
    }
    let mut out = Vec::new();
    while minute < to_ms {
        out.push(minute);
        minute = minute.saturating_add(60_000);
    }
    out
}

fn in_far_band(multiplier: f64) -> bool {
    multiplier + 1e-9 >= FAR_MULT_MIN && multiplier <= FAR_MULT_MAX + 1e-6
}

fn price_in_second(ticks: &[PriceTick], sec_ms: u64) -> Option<f64> {
    let end = sec_ms.saturating_add(1000);
    let idx = ticks.partition_point(|tick| tick.ts_ms < end);
    if idx == 0 {
        return None;
    }
    let tick = ticks[idx - 1];
    if tick.ts_ms >= sec_ms {
        Some(tick.price)
    } else {
        None
    }
}

fn parse_num<T: std::str::FromStr>(text: &str) -> Result<T, String> {
    text.parse::<T>().map_err(|_| format!("bad timestamp {text}"))
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = (year - era * 400) as u32;
    let shifted_month = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era as i64 * 146_097 + day_of_era as i64 - 719_468
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = (z - era * 146_097) as u64;
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_part = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_part + 2) / 5 + 1;
    let month = if month_part < 10 { month_part + 3 } else { month_part - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    (year as i32, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    fn tap(start_ms: u64, signed_dist: i64, multiplier: f64, move_sign: i8) -> CellTap {
        CellTap {
            quote_ts_ms: start_ms.saturating_sub(QUOTE_LEAD_MS),
            start_ms,
            window_index: 0,
            signed_dist,
            dist: signed_dist.abs(),
            multiplier,
            hit: false,
            us: true,
            move_sign,
            sigma: 0.001,
            sigma_raw: 0.001,
        }
    }

    #[test]
    fn utc_roundtrip_covers_epoch_leap_day_and_news_second() {
        for (ms, text) in [
            (0, "1970-01-01T00:00:00Z"),
            (1_709_164_800_000, "2024-02-29T00:00:00Z"),
            (1_709_251_200_000, "2024-03-01T00:00:00Z"),
            (1_795_392_000_000, "2026-11-23T00:00:00Z"),
            (1_795_442_189_000, "2026-11-23T13:56:29Z"),
            (1_795_444_200_000, "2026-11-23T14:30:00Z"),
        ] {
            assert_eq!(parse_utc(text).unwrap(), ms, "{text}");
            assert_eq!(format_utc(ms), text);
        }
    }

    #[test]
    fn normal_starts_keep_the_watch_and_honor_a_zero_cap() {
        let mut rng = StdRng::seed_from_u64(7);
        let capped = normal_session_starts(
            1_000_000,
            5_000_000,
            0,
            9_000_000,
            120_000,
            30_000,
            60_000,
            60_001,
            3,
            &mut rng,
        );
        assert_eq!(capped, vec![1_120_000, 1_180_000, 1_240_000]);

        let mut rng = StdRng::seed_from_u64(7);
        let all = normal_session_starts(
            1_000_000,
            1_400_000,
            0,
            9_000_000,
            120_000,
            30_000,
            60_000,
            60_001,
            0,
            &mut rng,
        );
        assert_eq!(all, vec![1_120_000, 1_180_000, 1_240_000, 1_300_000, 1_360_000]);
    }

    #[test]
    fn dense_starts_are_every_step_and_merge_dedupes() {
        let dense = dense_session_starts(1_000, 1_050, 10, 30, 10_000);
        assert_eq!(dense, vec![1_000, 1_010, 1_020, 1_030, 1_040]);
        assert_eq!(
            merge_starts(vec![1_000, 1_030], dense),
            vec![1_000, 1_010, 1_020, 1_030, 1_040]
        );
    }

    #[test]
    fn momentum_takes_nearest_cells_in_the_move_direction() {
        let taps = vec![
            tap(10, -1, 2.0, 1),
            tap(10, 4, 8.0, 1),
            tap(10, 1, 1.2, 1),
            tap(10, 2, 3.0, 1),
            tap(15, -2, 4.0, -1),
        ];
        let mut rng = StdRng::seed_from_u64(1);
        let picked = choose_taps("momentum", &taps, 2, &mut rng);
        assert_eq!(picked.len(), 2);
        assert!(picked.iter().all(|item| item.signed_dist > 0));
        assert_eq!(picked[0].signed_dist, 1);
        assert_eq!(picked[1].signed_dist, 2);
    }

    #[test]
    fn chase_far_keeps_only_x10_to_x25_in_the_move_direction() {
        let taps = vec![
            tap(10, 12, 9.99, 1),
            tap(10, 8, 10.0, 1),
            tap(10, 20, 25.0, 1),
            tap(10, 21, 25.01, 1),
            tap(10, -15, 25.0, 1),
            tap(10, 3, 12.0, -1),
        ];
        let mut rng = StdRng::seed_from_u64(1);
        let picked = choose_taps("chase_far", &taps, 18, &mut rng);
        assert_eq!(picked.len(), 2);
        assert_eq!(picked[0].signed_dist, 20);
        assert_eq!(picked[1].signed_dist, 8);
    }

    #[test]
    fn flat_move_leaves_shock_policies_idle_and_closest_unchanged() {
        let taps = vec![tap(10, 1, 25.0, 0), tap(10, -2, 25.0, 0)];
        let mut rng = StdRng::seed_from_u64(1);
        assert!(choose_taps("momentum", &taps, 18, &mut rng).is_empty());
        assert!(choose_taps("chase_far", &taps, 18, &mut rng).is_empty());
        let closest = choose_taps("closest", &taps, 1, &mut rng);
        assert_eq!(closest[0].signed_dist, 1);
    }

    #[test]
    fn stake_stats_and_worst_minute_match_the_house_books() {
        let stats = stake_stats(1_000.0, 1_250.0);
        assert!((stats.edge_pct + 25.0).abs() < 1e-9);
        assert!((stats.payout_ratio - 1.25).abs() < 1e-9);
        assert!((stats.loss_per_1000 - 250.0).abs() < 1e-9);
        let calm = stake_stats(1_000.0, 900.0);
        assert_eq!(calm.loss_per_1000, 0.0);
        let worst = worst_minute(&[(60_000, 100.0, 50.0), (120_000, 100.0, 400.0), (180_000, 0.0, 0.0)]);
        assert_eq!(worst, Some((120_000, -300.0)));
    }

    #[test]
    fn sixteen_second_gap_is_voided_on_every_overlapping_window() {
        let ticks = vec![
            PriceTick { ts_ms: 0, price: 140.0 },
            PriceTick { ts_ms: 16_000, price: 141.0 },
        ];
        let audits = audit_feed_gaps(&ticks, WINDOW_MS);
        assert_eq!(audits.len(), 1);
        assert_eq!(audits[0].gap_ms, 16_000);
        assert!(audits[0].overlapping_windows >= 4);
        assert_eq!(audits[0].voided_windows, audits[0].overlapping_windows);
    }

    #[test]
    fn realized_sigma_uses_only_adjacent_one_second_returns() {
        let ticks: Vec<PriceTick> = (0..5u64)
            .map(|sec| PriceTick {
                ts_ms: sec * 1000,
                price: if sec % 2 == 0 { 100.0 } else { 100.0 * 0.01_f64.exp() },
            })
            .collect();
        let (sigma, returns) = realized_step_sigma(&ticks, 0, 5_000).unwrap();
        assert_eq!(returns, 4);
        let expected = (4.0 * 0.01 * 0.01 / 3.0_f64).sqrt();
        assert!((sigma - expected).abs() < 1e-9, "{sigma} {expected}");
        let quotes = [
            WindowQuote { start_ms: 5_000, sigma: 0.002, sigma_raw: 0.001, max_mult: 4.0 },
            WindowQuote { start_ms: 10_000, sigma: 0.004, sigma_raw: 0.003, max_mult: 25.0 },
        ];
        assert_eq!(quoted_sigma_at(&quotes, 1_000, QUOTE_LEAD_MS), Some((0.002, 0.001)));
        let (avg_sigma, raw, max_mult, count) = bucket_quote(&quotes, 0, 20_000);
        assert_eq!(count, 2);
        assert!((avg_sigma - 0.003).abs() < 1e-12);
        assert!((raw - 0.002).abs() < 1e-12);
        assert_eq!(max_mult, 25.0);
        let k = tripwire_k(0.008, 0.001, 60.0).unwrap();
        assert!((k - 0.008 / (0.001 * 60.0_f64.sqrt())).abs() < 1e-9);
    }

    #[test]
    fn policies_tap_into_far_windows_on_a_24_window_board() {
        let quote_ts = 1_000_000_u64;
        let mut board = Vec::new();
        for window_index in 0..24usize {
            let start_ms = quote_ts + ((window_index + 1) as u64) * WINDOW_MS;
            for signed_dist in -3..=3i64 {
                board.push(CellTap {
                    quote_ts_ms: quote_ts,
                    start_ms,
                    window_index,
                    signed_dist,
                    dist: signed_dist.abs(),
                    multiplier: if signed_dist.abs() == 3 { 15.0 } else { 2.0 },
                    hit: false,
                    us: true,
                    move_sign: 1,
                    sigma: 0.001,
                    sigma_raw: 0.001,
                });
            }
        }
        let mut rng = StdRng::seed_from_u64(42);
        for policy in ["closest", "row_sweep", "long_bias", "short_bias", "random_near", "momentum", "chase_far"] {
            let picked = choose_taps(policy, &board, 18, &mut rng);
            assert_eq!(picked.len(), 18, "policy={policy}");
            assert!(
                picked.iter().any(|t| t.window_index >= 6),
                "expected policy {policy} to tap far windows (>=6)"
            );
        }
    }
}
