# Odds Model v2, M4 proposal: direction memory and the payout cap

Status: draft for the owner's review, 2026-11-24. Nothing here is built. Live odds are M3 (NVDA, TSLA, 6 windows, cap x25).

Data: Hyperliquid xyz `markPx`, 1 Hz, 2026-11-18 to 2026-11-24, NVDA and TSLA. Scripts and raw outputs are in `../logs/calibration/m4-trend/` (`trend-check.*`, `sign_acf.py` + `sign-acf.txt`, `trend_fit.py` + `trend-fit.txt`).

## 0. Update, 2026-11-24 evening: the "trend" is the settlement price catching up with the order-book mid

After the owner approved two trend speeds, the fits (`dirfit-two-speed.txt`, `dirfit-onoff.txt`) showed that no hidden-trend shape fits quiet markets. Two speeds overshoot the measured autocorrelation by 2–3x. An on/off trend undershoots the tail and needs a strength of 0.49 (near-certain staircases). That pointed to a mechanism, not a random trend.

**The mechanism:** each `activeAssetCtx` record carries both `markPx` (settlement) and `midPx` (order-book mid). Over the next 3 s (one oracle update), `markPx` moves toward `midPx` (`gap_check.py`, `mid-catchup.txt`):

| | moves preceded by a gap ≥ 0.5c | moves toward the mid | catch-up per 3 s ($\kappa$, regression) |
|---|---:|---:|---:|
| NVDA quiet | 94% | 91.0% | 0.05 |
| NVDA normal | 96% | 88.7% | 0.24 |
| NVDA volatile | 97% | 89.0% | 0.70 |
| NVDA news | 99% | 89.6% | 0.45 |
| TSLA quiet | 97% | 84.4% | 0.06 |
| TSLA normal | 98% | 80.2% | 0.35 |
| TSLA volatile | 99% | 83.3% | 0.61 |
| TSLA news | 99% | 87.0% | 0.55 |

The oracle price gap does not predict direction (38–47% toward). The mid does.

Example, NVDA normal, gap +3..+6c: over the next 3 s the mark goes up 72.8% of the time, down 9.4%, and stays 17.8%. The v2 coin flip says 50/50.

**Consequences:**
- **Explains everything in §1.** The sign persistence, the staircases in quiet (the mark crawls toward a distant mid at κ ≈ 0.05), and why `momentum` and `chase_far` beat the model.
- **Visible to anyone.** A player watching the Hyperliquid book can predict the next settlement move about 85% of the time. Live M3 ignores the mid, so it can be beaten this way. A `mid_follow` player style is being added to the backtest to measure by how much.
- **Better model.** Two prices: the mid $Y$ (a random walk) and the settlement $X$, which at each oracle update moves toward $Y$ by a measured fraction plus noise. The hidden trends in §2–3 below become unnecessary. The engine already receives `midPx` in the same Hyperliquid message.

## 0b. The settlement formula, and a model built on it (proposal for the owner)

**Source** (docs.trade.xyz, Mark Price page). The mark is the median of three components, recomputed at each relayer update (about every 3 s):
1. the oracle price $O$;
2. $O$ plus a 150-second continuous-time EMA of the basis (mid − oracle);
3. the median of best bid, best ask and last trade.

The oracle follows the external fair price during US hours. Outside them it advances by its own 30-minute EMA of the perp book's impact prices.

**Identity.** Let $B$ be the book price, $D = B - O$ the basis and $E = \text{EMA}_{150\,\text{s}}(D)$. Then

$$X = O + \operatorname{median}(0,\ E,\ D).$$

Check on the 1 Hz data, share of mark updates where this lands within 1.5¢ (`mark-inputs.txt`):

| | quiet | normal | volatile | news |
|---|---:|---:|---:|---:|
| NVDA | 95.5% | 86.8% | 75.9% | 46.7% |
| TSLA | 88.6% | 83.6% | 73.8% | 50.0% |

The misses come from 1-second snapshots instead of the exact update instant, and from using the mid in place of median(bid, ask, last). A per-market clamp did not improve the match (`mark_formula2.py`). Quiet markets are 99% outside US hours; volatile and news are 58–81% inside them.

**Input dynamics, measured per regime** (`mark-model-sim2.txt`):
- Book $B$: random walk with per-second noise $\sigma_B$.
- Oracle $O$: at each update $O \leftarrow O + \lambda\,(B - O - E) + J$, where $J = \pm K$ with probability $p_J$ and $K \sim 1 + \text{Geom}$. The oracle follows the book. Regressing the book on the oracle gives a pull of ≈ 0, so the book leads.
- $E$: deterministic, computed exactly as the relayer does.
- Mark $X = \operatorname{round}(O + \operatorname{median}(0, E, B - O))$ at each update.

| | $\lambda$ per 3 s | $p_J$ | Geom $p$ | $\sigma_B$ ¢/s |
|---|---:|---:|---:|---:|
| NVDA quiet | 0.023 | 0.27 | 0.50 | 1.27 |
| NVDA normal | 0.142 | 0.76 | 0.40 | 2.04 |
| NVDA volatile | 0.359 | 0.87 | 0.24 | 4.03 |
| NVDA news | 0.292 | 0.96 | 0.065 | 12.2 |
| TSLA quiet | 0.033 | 0.25 | 0.29 | 1.58 |
| TSLA normal | 0.231 | 0.87 | 0.23 | 3.14 |
| TSLA volatile | 0.456 | 0.93 | 0.15 | 5.69 |
| TSLA news | 0.402 | 0.95 | 0.068 | 13.4 |

**Validation: simulate the model, then compare with the real mark:**
- **Share of moves toward the mid:** model 0.86–0.92, real 0.78–0.89. Reproduced from the formula alone, without fitting to it.
- **Moves per 100 s:** model 22–33, real 13–37. Quiet is too high (22–24 vs 13–16).
- **Slow persistence in quiet:** partly reproduced. Model $\rho$ is 0.03–0.12 vs real 0.04–0.20.
- **Same-direction pair at consecutive updates:** not reproduced. Real $\rho(1)$ is 0.08–0.23, model ≈ 0.
  - It is not an artifact of how updates arrive. Pairs exactly 3 s apart show P(same) 0.556 (NVDA) and 0.565 (TSLA) (`pair-timing.txt`).
  - Next check: whether the book's own 3-second moves persist. If they do, it becomes a measured property of $B$, not a correction factor.

**Pricing:**
- **Method:** Monte Carlo from the live state ($X$, $O$, $B$, $E$, oracle phase). The engine already receives `oraclePx` and `midPx` with `markPx`. Simulate $B$ per second and $O$ per update, compute $X$ at each update, and settle each window as in v2 Method B.
- **Parameters:** every one is measured from the price stream; none is fitted to house P&L.
- **Margin and cap:** the M3 margin rule is recomputed from the new model's errors. Caps x25, x50 and x100 are tested.

Sections 1–3 below are kept as the record of how we got here.

## 1. What the data says

The v2 price process (§1.4 of `odds-model-v2.md`) draws the sign of every 3-second move as a fair coin. The data says it is not.

**1.1 Next move goes the same way** (0.500 would be a fair coin):

| Regime | NVDA | TSLA |
|---|---:|---:|
| quiet | 0.545 | 0.589 |
| normal | 0.549 | 0.541 |
| volatile | 0.576 | 0.583 |
| news | 0.624 | 0.595 |

**1.2 How long the direction is remembered.** Sign autocorrelation $\rho(k) = \mathbb{E}[S_n S_{n+k}]$ between a move and the $k$-th next move (`sign-acf.txt`):

| Regime | $k=1$ | $k=2$ | $k=5$ | $k=10$ | $k=30$ | seconds for $k=30$ |
|---|---:|---:|---:|---:|---:|---:|
| NVDA quiet | 0.090 | 0.166 | 0.088 | 0.075 | 0.046 | ~200 |
| NVDA normal | 0.098 | 0.038 | 0.009 | 0.008 | -0.004 | ~120 |
| NVDA volatile | 0.153 | 0.023 | 0.002 | -0.003 | 0.014 | ~100 |
| TSLA quiet | 0.175 | 0.214 | 0.153 | 0.129 | 0.074 | ~260 |
| TSLA normal | 0.082 | 0.033 | 0.015 | 0.007 | 0.004 | ~130 |
| TSLA volatile | 0.165 | 0.027 | 0.017 | -0.008 | 0.004 | ~105 |

Two different shapes:
- **Normal and volatile:** a spike at $k=1$, then almost nothing. A move remembers mainly the move before it.
- **Quiet:** a long flat tail. The direction persists for minutes: a slow drift.

**1.3 Why the M3 accuracy tables still looked fine.** These tables average the up side and the down side of the board. The trend makes cells on the side of the last move more likely and cells on the other side less likely, so the two errors cancel in the average. `momentum` and `chase_far` tap only the trend side, so they collect the error. That is why they are the styles that fail at 2 minutes (TSLA 24w momentum −7.6%) and at higher caps (x100 momentum −19.9% NVDA).

## 2. Model: moves that remember their direction

Everything in v2 stays the same: the 3-second oracle clock and phase, the move rate $p_{\text{move}}$, the jump size $K$, settlement, regimes. Only the sign rule changes.

Let $S_n \in \{-1,+1\}$ be the sign of the $n$-th move and $Z_n \in \{-1,+1\}$ a hidden trend.

$$\mathbb{P}(S_n = +1 \mid Z_n, S_{n-1}) = \tfrac12 + \beta Z_n + \gamma S_{n-1}, \qquad |\beta| + |\gamma| < \tfrac12$$

$$\mathbb{P}(Z_n \ne Z_{n-1}) = h \quad \text{(per move)}$$

- $\gamma$: memory of the last move (the $k=1$ spike).
- $\beta$: strength of the hidden trend.
- $h$: how often the trend flips. The trend lasts $1/h$ moves on average.
- With $\beta=\gamma=0$ this is exactly v2.

**Implied autocorrelation** (closed form, used to check the fit). With $a=2\beta$, $g=2\gamma$, $\lambda = 1-2h$:

$$\rho(k) = \frac{a^2}{1-g\lambda}\,\lambda^k + g\,\rho(k-1), \qquad \rho(0)=1.$$

Check, NVDA normal: $\beta=0.122$, $\gamma=0.028$, $h=0.164$ gives $\rho(1)=0.098$, $\rho(2)=0.034$. Measured: 0.097 and 0.034.

**Per-move size is unchanged**, so the 3-second variance that sets $R_{60}$ is unchanged. What changes is how moves add up over 15–120 s, and which side they go to.

### 2.1 Live inference (causal, exact Bayes)

The engine keeps $\pi_n = \mathbb{P}(Z_n = +1 \mid S_1..S_n)$. At each observed move:

$$\pi^- = \pi_{n-1}(1-h) + (1-\pi_{n-1})\,h$$
$$\pi_n = \frac{\pi^-\,\ell(+1)}{\pi^-\,\ell(+1) + (1-\pi^-)\,\ell(-1)}, \qquad \ell(z) = \tfrac12 + S_n(\beta z + \gamma S_{n-1})$$

These are the same equations used to fit the parameters, so the fit and the live engine cannot disagree.

### 2.2 Touch probability (exact DP, extended state)

The v2 DP state $x$ (price in cents) becomes $(x, S_{\text{last}}, Z)$: 4 combinations.
- At each oracle tick: with probability $1-p_{\text{move}}$ nothing changes.
- On a move, $Z$ flips with probability $h$, the sign is drawn by the rule above, and $K$ is drawn as in v2.
- Start state: $x = C_0$, $S_{\text{last}}$ = the sign of the last observed move, $Z \sim (\pi, 1-\pi)$.
- Window, crossing and void rules are unchanged.

Cost: 4x the v2 DP. v2 at 6 windows is about 3 ms per ticker per reprice, so this stays well inside budget.

## 3. Fit: maximum likelihood on the direction of every move

Fitted on the sign sequence only. House P&L is not used anywhere in the fit. Runs are split at regime changes and at gaps over 120 s (`trend-fit.txt`).

| | $\beta$ | $\gamma$ | $h$ | trend life | LL gain vs coin | LL gain vs last-move-only |
|---|---:|---:|---:|---:|---:|---:|
| NVDA quiet | 0.179 | −0.013 | 0.054 | ~89 s | 647 | 520 |
| NVDA normal | 0.122 | +0.028 | 0.164 | ~24 s | 225 | 20 |
| NVDA volatile | 0.024 | +0.075 | — | — | 239 | 1.1 |
| NVDA news | 0.145 | +0.082 | 0.058 | ~56 s | 50 | 5.8 |
| TSLA quiet | 0.192 | +0.017 | 0.038 | ~154 s | 1048 | 636 |
| TSLA normal | 0.072 | +0.030 | 0.059 | ~76 s | 137 | 17 |
| TSLA volatile | 0.049 | +0.081 | — | — | 283 | 1.0 |
| TSLA news | 0.075 | +0.078 | — | — | 25 | 1.1 |

**Structure test (likelihood ratio, 2 extra numbers, 5% level: gain must exceed 3.0):**
- **Volatile (both) and TSLA news:** the hidden trend is not supported (gain ~1). Only the last-move memory stays: $\gamma \approx 0.08$, $\beta = 0$.
- **Normal and NVDA news:** both parts are supported.
- **Quiet:** both parts are strongly supported, but the fit misses the slow tail:

| | $\rho(5)$ meas / model | $\rho(10)$ meas / model | $\rho(30)$ meas / model |
|---|---|---|---|
| NVDA quiet | 0.105 / 0.069 | 0.100 / 0.037 | 0.082 / 0.004 |
| TSLA quiet | 0.191 / 0.109 | 0.189 / 0.073 | 0.178 / 0.016 |

Normal, volatile and news match within ~0.01 at every lag (see `trend-fit.txt`).

**Proposed fix for quiet: two hidden trends,** a fast one $Z^f$ (flip rate $h_f$) and a slow one $Z^s$ (flip rate $h_s$):

$$\mathbb{P}(S_n=+1) = \tfrac12 + \beta_f Z^f_n + \beta_s Z^s_n + \gamma S_{n-1}$$

- Same structure for every regime, 5 numbers each.
- The likelihood-ratio test decides per regime which parts stay. Volatile will reduce to $\gamma$ only.
- DP state: 8 combinations. Still exact, still cheap.

## 4. The payout cap

Edge on one tap, true probability $P$, quoted $M = \min\!\big(M_{\max}, (1-m)/\hat P\big)$:

$$e = 1 - M P$$

- **Uncapped, model right ($\hat P = P$):** $e = m$.
- **Model wrong by a ratio $r = P/\hat P$:** $e = 1 - (1-m)\,r$. At $r = 2$ the house pays out 1.9x the stake on average, and this holds whatever the multiplier.
- **Capped ($\hat P < (1-m)/M_{\max}$):** $e = 1 - M_{\max} P$, which is more than $m$. I call the difference "cap rent".

Average edge over a mix of taps with weights $w_i$:

$$\bar e = m + \sum_{\text{capped}} w_i\,\big(1 - M_{\max} P_i - m\big)$$

- **At x25:** a third to two thirds of tapped volume lands on capped cells (`cap%` 35–68 in `m3-summary.txt` §2). Cap rent is why the 8-style average is 20–37% instead of 2–5%.
- **Raising the cap** removes cap rent but sells cells with small $\hat P$. Those are the far rows, where the trend error ratio $r$ is largest. That is the x50/x100 failure.
- **The trend model is what makes a higher cap safe:** it fixes $r$ on the trend side.

**Liability per $10 tap** (pool $5,000): x25 → $250, x50 → $500, x100 → $1,000.

**Options:**
- **A. Raise after the fix.** Test x50 and x100 with the trend model and keep the highest cap that passes your rule.
- **B. Do not sell above the cap.** Cells whose fair payout exceeds $M_{\max}$ show "—" and cannot be tapped. Then $\bar e = m$ exactly, with no cap rent at any cap, but fewer far cells to play.
- **C. A + B.** Raise the cap as far as the backtest allows, and do not sell beyond it.

## 5. Margin

Same M3 rule, no new knob: $m_{\text{base}} = \max(2.5\%,\ Q_{0.75}(m_{\text{solv}}) + 2\%)$, recomputed from the new model's errors.
- If the trend explains the M3 errors, normal and volatile should fall from 7–20% toward 2.5–5%.
- If they don't fall, the model is still missing something, and the backtest will show it.

## 6. Test plan (before anything goes live)

1. **Out of sample.** Fit on 11-18..11-21 and test on 11-22..11-24. Then report the full 7 days as well.
2. **Accuracy by side.** Split by trend side vs. other side, so errors can no longer cancel.
3. **Player-style backtest.** Your 8 styles, 6w and 24w, NVDA and TSLA, at x25 / x50 / x100. Pass: every style pooled > 0, average 2–5%.
4. **News replay** of 2026-11-23, including `chase_far`.
5. **CPU check** for 4 tickers.

## 7. Decisions for the owner

1. **Quiet structure:** two hidden trends (fast + slow), or one trend and keep quiet at x25?
2. **Cap:** A, B or C?
