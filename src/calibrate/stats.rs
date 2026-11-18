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

/// US regular session (`09:30-16:00 America/New_York`, weekdays, non-holiday).
pub fn is_us_regular_session_utc(ts_ms: u64) -> bool {
    crate::session::is_us_regular_session(ts_ms)
}

#[derive(Debug, Clone, PartialEq)]
pub struct TailBucket {
    pub lo: f64,
    pub hi: f64,
    pub count: usize,
    pub mean_predicted: f64,
    pub hit_rate: f64,
    pub hit_to_pred_ratio: f64,
    pub implied_edge: f64,
}

const TAIL_BOUNDS: [(f64, f64); 5] = [
    (0.0, 0.005),
    (0.005, 0.01),
    (0.01, 0.02),
    (0.02, 0.05),
    (0.05, 0.1),
];

pub fn tail_reliability_table(samples: &[(f64, bool, f64)]) -> Vec<TailBucket> {
    let mut counts = [0usize; 5];
    let mut pred_sums = [0.0_f64; 5];
    let mut hit_sums = [0usize; 5];
    let mut payout_sums = [0.0_f64; 5];

    for &(raw_p, hit, mult) in samples {
        if !raw_p.is_finite() || raw_p >= 0.1 {
            continue;
        }
        let p = raw_p.max(0.0);
        let Some(idx) = TAIL_BOUNDS.iter().position(|&(lo, hi)| p >= lo && p < hi) else {
            continue;
        };
        counts[idx] += 1;
        pred_sums[idx] += p;
        if hit {
            hit_sums[idx] += 1;
            payout_sums[idx] += mult;
        }
    }

    TAIL_BOUNDS
        .iter()
        .enumerate()
        .map(|(i, &(lo, hi))| {
            let count = counts[i];
            let mean_predicted = if count > 0 {
                pred_sums[i] / count as f64
            } else {
                0.0
            };
            let hit_rate = if count > 0 {
                hit_sums[i] as f64 / count as f64
            } else {
                0.0
            };
            let hit_to_pred_ratio = if mean_predicted > 0.0 {
                hit_rate / mean_predicted
            } else {
                0.0
            };
            let implied_edge = if count > 0 {
                1.0 - payout_sums[i] / count as f64
            } else {
                0.0
            };
            TailBucket {
                lo,
                hi,
                count,
                mean_predicted,
                hit_rate,
                hit_to_pred_ratio,
                implied_edge,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_table_splits_sub_buckets_and_computes_ratio_and_edge() {
        let samples = vec![
            (0.002, false, 25.0),
            (0.004, true, 25.0),
            (0.006, false, 25.0),
            (0.015, true, 25.0),
            (0.030, false, 25.0),
            (0.040, true, 22.0),
            (0.060, true, 15.0),
            (0.080, false, 12.0),
            (0.100, true, 9.5),
            (0.500, true, 1.9),
        ];

        let tail = tail_reliability_table(&samples);
        assert_eq!(tail.len(), 5);
        assert_eq!((tail[0].lo, tail[0].hi, tail[0].count), (0.0, 0.005, 2));
        assert!((tail[0].mean_predicted - 0.003).abs() < 1e-12);
        assert!((tail[0].hit_rate - 0.5).abs() < 1e-12);
        assert!((tail[0].hit_to_pred_ratio - (0.5 / 0.003)).abs() < 1e-9);
        assert!((tail[0].implied_edge - (1.0 - 12.5)).abs() < 1e-12);

        assert_eq!((tail[1].lo, tail[1].hi, tail[1].count), (0.005, 0.01, 1));
        assert!((tail[1].implied_edge - 1.0).abs() < 1e-12);

        assert_eq!((tail[2].lo, tail[2].hi, tail[2].count), (0.01, 0.02, 1));
        assert!((tail[2].hit_rate - 1.0).abs() < 1e-12);

        assert_eq!((tail[3].lo, tail[3].hi, tail[3].count), (0.02, 0.05, 2));
        assert!((tail[3].hit_rate - 0.5).abs() < 1e-12);
        assert!((tail[3].implied_edge - (1.0 - 11.0)).abs() < 1e-12);

        assert_eq!((tail[4].lo, tail[4].hi, tail[4].count), (0.05, 0.1, 2));
        assert!((tail[4].mean_predicted - 0.070).abs() < 1e-12);
        assert!((tail[4].hit_rate - 0.5).abs() < 1e-12);
        assert!((tail[4].implied_edge - (1.0 - 7.5)).abs() < 1e-12);
    }
}
