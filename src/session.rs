const SECS_PER_DAY: i64 = 86_400;
const SECS_PER_HOUR: i64 = 3_600;
const EST_OFFSET_SECS: i64 = -5 * SECS_PER_HOUR;
const EDT_OFFSET_SECS: i64 = -4 * SECS_PER_HOUR;
const OPEN_SECS_LOCAL: i64 = 9 * SECS_PER_HOUR + 30 * 60;
const CLOSE_SECS_LOCAL: i64 = 16 * SECS_PER_HOUR;

/// NYSE holidays matching `tap2stacks-backend/src/modules/session/nyse-calendar.ts`.
const NYSE_HOLIDAYS: &[(i64, u32, u32)] = &[(2026, 11, 26), (2026, 12, 25), (2027, 1, 1)];

/// Returns true when `ts_ms` falls within NYSE regular trading hours (`09:30..16:00 America/New_York`),
/// Monday through Friday, excluding NYSE holidays, accounting for US DST rules.
pub fn is_us_regular_session(ts_ms: u64) -> bool {
    let utc_secs = (ts_ms / 1_000) as i64;
    let offset = eastern_utc_offset_secs(utc_secs);
    let ny_secs = utc_secs + offset;
    let ny_days = ny_secs.div_euclid(SECS_PER_DAY);
    let sec_in_day = ny_secs.rem_euclid(SECS_PER_DAY);

    if !(OPEN_SECS_LOCAL..CLOSE_SECS_LOCAL).contains(&sec_in_day) {
        return false;
    }

    let weekday = weekday_mon_1(ny_days);
    if weekday > 5 {
        return false;
    }

    let date = civil_from_days(ny_days);
    !NYSE_HOLIDAYS.contains(&date)
}

fn eastern_utc_offset_secs(utc_secs: i64) -> i64 {
    let utc_days = utc_secs.div_euclid(SECS_PER_DAY);
    let (year, _, _) = civil_from_days(utc_days);
    if is_us_eastern_dst(year, utc_secs) {
        EDT_OFFSET_SECS
    } else {
        EST_OFFSET_SECS
    }
}

/// US DST: 2nd Sunday of March 02:00 EST (07:00 UTC) to 1st Sunday of November 02:00 EDT (06:00 UTC).
fn is_us_eastern_dst(year: i64, utc_secs: i64) -> bool {
    let mar1_days = days_from_civil(year, 3, 1);
    let mar_second_sunday = 8 + (7 - weekday_mon_1(mar1_days)) % 7;
    let dst_start_utc = days_from_civil(year, 3, mar_second_sunday) * SECS_PER_DAY + 7 * SECS_PER_HOUR;

    let nov1_days = days_from_civil(year, 11, 1);
    let nov_first_sunday = 1 + (7 - weekday_mon_1(nov1_days)) % 7;
    let dst_end_utc = days_from_civil(year, 11, nov_first_sunday) * SECS_PER_DAY + 6 * SECS_PER_HOUR;

    utc_secs >= dst_start_utc && utc_secs < dst_end_utc
}

/// Monday = 1 ..= Sunday = 7 (1970-01-01 was Thursday = 4).
fn weekday_mon_1(days_since_epoch: i64) -> u32 {
    ((days_since_epoch + 3).rem_euclid(7) + 1) as u32
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = year - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = month as i64;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    (y + i64::from(m <= 2), m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc_ms(year: i64, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> u64 {
        let secs = days_from_civil(year, month, day) * SECS_PER_DAY
            + i64::from(hour) * SECS_PER_HOUR
            + i64::from(min) * 60
            + i64::from(sec);
        (secs as u64) * 1_000
    }

    #[test]
    fn nyse_winter_est_boundaries() {
        assert!(is_us_regular_session(utc_ms(2026, 11, 17, 14, 30, 0)));
        assert!(is_us_regular_session(utc_ms(2026, 11, 17, 20, 59, 59)));
        assert!(!is_us_regular_session(utc_ms(2026, 11, 17, 14, 29, 59)));
        assert!(!is_us_regular_session(utc_ms(2026, 11, 17, 21, 0, 0)));
    }

    #[test]
    fn nyse_summer_edt_holidays_and_weekends() {
        // Summer EDT: 09:30 NY == 13:30 UTC
        assert!(is_us_regular_session(utc_ms(2026, 7, 15, 13, 30, 0)));
        assert!(!is_us_regular_session(utc_ms(2026, 7, 15, 13, 29, 59)));
        // Thanksgiving 2026-11-26
        assert!(!is_us_regular_session(utc_ms(2026, 11, 26, 15, 0, 0)));
        // Saturday 2026-11-21
        assert!(!is_us_regular_session(utc_ms(2026, 11, 21, 15, 0, 0)));
    }

    #[test]
    fn dst_transitions_match_us_rules() {
        // 2026: DST starts Mar 8 07:00 UTC, ends Nov 1 06:00 UTC
        assert_eq!(eastern_utc_offset_secs((utc_ms(2026, 3, 8, 6, 59, 59) / 1_000) as i64), EST_OFFSET_SECS);
        assert_eq!(eastern_utc_offset_secs((utc_ms(2026, 3, 8, 7, 0, 0) / 1_000) as i64), EDT_OFFSET_SECS);
        assert_eq!(eastern_utc_offset_secs((utc_ms(2026, 11, 1, 5, 59, 59) / 1_000) as i64), EDT_OFFSET_SECS);
        assert_eq!(eastern_utc_offset_secs((utc_ms(2026, 11, 1, 6, 0, 0) / 1_000) as i64), EST_OFFSET_SECS);
    }
}
