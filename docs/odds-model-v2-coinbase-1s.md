# Odds model v2 on the Coinbase 1-second index (draft for the owner's review)

Status: DRAFT, 2026-11-24 19:30 UTC. Early numbers come from 3.8 h of US-hours data
(14:35-18:20 UTC, 2026-11-24). The full-day fit runs after 24 h of recording
(2026-11-25 14:45 UTC); the 7-day backtest decides the cap.

This is the main line (QTAP pricing formula + v2 maths). It changes the price source
and the clock. It is not a new model version.

## 0. What stays exactly the same

- Pricing formula: `M = clamp((1 - m_eff) / P × K_skew × K_safety, 1.05, cap)`,
  `m_eff = m_base(preset) + m_risk`, `m_risk = clamp(0.005·(R_60/R_ref - 1)+, 0, 2.5%)`.
- Win rule (QTAP): the straight segment between two consecutive prints counts; a row
  is hit if any segment inside the window crosses it.
- Four presets chosen by `R_60` (minute RMS of 1 s cent moves), fast upgrade, 15 s
  downgrade hysteresis, single-print shock trigger.
- M3 margin rule: `m_base = max(2.5%, Q0.75(m_solv) + 2%)`, with `m_solv` taken from the
  model's own error on recorded data.
- Cap rule: test x25 / x50 / x100 and keep the highest cap that passes ("raise only").
- Pass rule: every QTAP player style has pooled edge > 0, and the average is 2-5%.

## 1. The price source

Coinbase-owned Deribit perp index `deribit_price_index.nvda_usdc` / `tsla_usdc`
(public WebSocket, no key). Measured:

| | NVDA | TSLA |
|---|---:|---:|
| Index prints per second | 0.81 | 0.70 |
| Print spacing (ms) | 1006 / 2014 / 3021, on a 1 s clock | same |
| Prints with no price change | 1.2% | 2.3% |
| Price step | whole cents | whole cents |

The index prints on a 1 s clock **only when the price changes**. A silent second
means "unchanged", as long as the connection is alive.

## 2. Data rule (settlement and quoting use the same samples)

One sample per second per ticker:

1. A new index print with a changed value is emitted immediately. The perp ticker's
   `index_price` is never used as a price. In 3.8 h, 94 (NVDA) and 75 (TSLA) ticker
   messages that were newer than the latest print still showed the previous index value,
   and 14 and 13 showed values the index never printed. Using them would create about
   40 fake price reversals per hour. Ticker messages only prove the connection is alive
   and supply the book mid and mark.
2. Otherwise, if at least 1 s has passed and the connection is proven alive in that
   second (1 s `public/test` ping answered, or any message received), re-emit the
   last value. This sample means the index was confirmed unchanged.
3. If liveness is not proven, nothing is emitted. A gap over 3 s voids the windows
   it overlaps, as today.

On the recording, this rule leaves 1 real gap over 3 s per ticker in 3.8 h. Using
index prints alone would leave 380 (NVDA) and 900 (TSLA), and void many fair windows.

## 3. The process: 1-second clock

The 3 s clock, its phase φ and `p_move = min(1, 3q)` all go. Each second:

`C_t = C_{t-1} + I_t · S_t · K_t`, with `I_t ~ Bernoulli(q)`, `S_t = ±1` equally likely,
and `K_t ~ Geom(p)` (two-geometric mix only where the fit says it is needed).

The touch probability is the same exact 1-D DP as before, stepping every second.
It is simpler, because there is no phase to average over.

Early fit (US hours only; thresholds will move once night minutes are included):

| Preset | NVDA q | NVDA E[K] | TSLA q | TSLA E[K] |
|---|---:|---:|---:|---:|
| quiet | 0.749 | 2.17c | 0.615 | 3.47c |
| normal | 0.828 | 3.11c | 0.736 | 5.15c |
| volatile | 0.878 | 4.57c | 0.790 | 7.20c |
| news (3 min only) | 0.917 | 6.38c | 0.839 | 11.80c |

R_60 thresholds (US hours only): NVDA 2.87 / 4.48 / 7.05c, TSLA 4.86 / 7.51 / 13.34c.

## 4. The one new term: the order book leads the index by about 2 s

Measured with `C` = index, `B` = perp book mid and `K` = perp mark, all in cents.
The gap `G_t = B_t - K_t` is the fresh book move that the index has not shown yet;
the mark already contains the slow premium. Regressing `C_{t+h} - C_t` on `G_t`:

| h (s) | 1 | 2 | 3 | 5 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| NVDA slope β | 0.43 | 0.82 | 0.96 | 1.03 | 1.05 | 1.02 |
| NVDA R² | 0.48 | 0.76 | 0.65 | 0.41 | 0.25 | 0.19 |
| TSLA slope β | 0.34 | 0.69 | 0.86 | 0.94 | 0.96 | 0.97 |
| TSLA R² | 0.40 | 0.67 | 0.64 | 0.42 | 0.26 | 0.21 |

By 3 s, the index has caught up with the whole gap (β ≈ 1). For h ≥ 3, R² ≈ L/h with
L ≈ 2 s. This is what you would see if the index were the book price delayed by about
2 s. Taps must be placed at least 5 s before their window starts, so every window
starts after the catch-up has finished.

**Model.** Let `X` be the book price. Then `C_t = X_{t-L}`, and `X_t = C_t + G_t` is
known now. For a window `[a, b)` with `a - t ≥ 5 s`, the index path inside the window
is `X` on `[a - L, b - L)`. The touch probability is therefore the same DP:

- start at `c0 = C_t + round(G_t)`, not at `C_t`;
- run `a - t - L` seconds to the window start, not `a - t`;
- then run the 5 window seconds as before.

That is two numbers per ticker, both measured from the stream (`β∞ ≈ 1`, `L ≈ 2 s`),
with no fitting to profit and loss. With `G = 0` and `L = 0` it is exactly the plain
v2. Safety: use `G = 0` when the ticker is over 2 s old, and clamp `|G|` to the p99
of the recorded gap. Without this term, a player who watches the book (`mid_follow`
style, or any real-time NVDA quote) gets the same edge that broke M3 on Hyperliquid.

## 5. Row height (a game-feel choice, not maths)

The 1 s index moves more per window than the old 3 s mark: RMS per second is 3.5c
(NVDA) and 5.9c (TSLA). With today's rows (NVDA 3c, TSLA 8c):

- The rows next to the price are near coin flips (about x1.8).
- x25 sits about 9-11 rows out for NVDA and 5 rows out for TSLA, in the first window.

Wider rows (for example NVDA 5c, TSLA 10c) would spread the multipliers over fewer,
clearer rows. The maths is the same either way.

## 6. Validation before any switch

1. Fit presets and `L`, `β∞` on the first full 24 h with `logs/calibration/drb-tick-stats.py`
   and `drb_predict.py`. Check nights and weekends: who feeds the index when US markets
   are closed?
2. Run the 7-day policy backtest (`policy_backtest --source deribit`, replaying the exact
   live sample rule) with all 9 QTAP styles, including `mid_follow` reading the perp
   book, at x25 / x50 / x100.
3. Pass: every style's pooled edge > 0, average 2-5%. Keep the highest cap that passes.
4. the owner approves, then deploy. After that, delete v1, M4 and the Hyperliquid feed.

Scripts: `logs/calibration/drb-tick-stats.py` (preset fitter on the 1 s grid);
`logs/calibration/drb-analysis/drb_day1.py` (cadence and touch rates) and
`drb_predict.py` (book lead); copies run on analysis-vps in the e7 work folder.
