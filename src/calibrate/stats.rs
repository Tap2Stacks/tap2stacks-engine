#[derive(Debug, Clone, PartialEq)]
pub struct Bucket {
    pub lo: f64,
    pub hi: f64,
    pub count: usize,
    pub mean_predicted: f64,
    pub hit_rate: f64,
}

pub fn reliability_table(predicted_and_outcome: &[(f64, bool)], buckets: usize) -> Vec<Bucket> {
    if buckets == 0 {
        return Vec::new();
    }
    let mut counts = vec![0usize; buckets];
    let mut pred_sums = vec![0.0_f64; buckets];
    let mut hit_sums = vec![0usize; buckets];

    for &(raw_p, hit) in predicted_and_outcome {
        let p = if raw_p.is_finite() {
            raw_p.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let idx = ((p * buckets as f64).floor() as usize).min(buckets - 1);
        counts[idx] += 1;
        pred_sums[idx] += p;
        if hit {
            hit_sums[idx] += 1;
        }
    }

    (0..buckets)
        .map(|i| {
            let count = counts[i];
            Bucket {
                lo: i as f64 / buckets as f64,
                hi: (i + 1) as f64 / buckets as f64,
                count,
                mean_predicted: if count > 0 {
                    pred_sums[i] / count as f64
                } else {
                    0.0
                },
                hit_rate: if count > 0 {
                    hit_sums[i] as f64 / count as f64
                } else {
                    0.0
                },
            }
        })
        .collect()
}

pub fn brier_score(predicted_and_outcome: &[(f64, bool)]) -> f64 {
    if predicted_and_outcome.is_empty() {
        return 0.0;
    }
    let sum: f64 = predicted_and_outcome
        .iter()
        .map(|&(p, hit)| {
            let y = if hit { 1.0 } else { 0.0 };
            (p - y).powi(2)
        })
        .sum();
    sum / predicted_and_outcome.len() as f64
}

pub fn implied_house_edge(samples: &[(f64, bool, f64)]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let payout_sum: f64 = samples
        .iter()
        .map(|&(_p, hit, mult)| if hit { mult } else { 0.0 })
        .sum();
    1.0 - payout_sum / samples.len() as f64
}

/// US regular session: weekdays (Mon..=Fri), 13:30..20:00 UTC.
pub fn is_us_regular_session_utc(ts_ms: u64) -> bool {
    let days_since_epoch = ts_ms / 86_400_000;
    // 1970-01-01 was Thursday (weekday 4 in Mon=1..Sun=7).
    let weekday = ((days_since_epoch + 3) % 7) + 1;
    if weekday > 5 {
        return false;
    }
    let secs_in_day = (ts_ms / 1_000) % 86_400;
    secs_in_day >= 48_600 && secs_in_day < 72_000
}
