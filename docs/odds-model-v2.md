# Odds Model v2 — Mathematical Specification

**Status:** Maths-only proposal (no Rust changes). Prepared for review with the owner before implementation.  
**References:** QTAP paper (`3_proposed.tex`, `4_experimental.tex`), current engine spec (`MATH_MODEL.md`), empirical dataset `logs/calibration/m1-tick-stats.txt` (7 recorded days, `2026-11-17` through `2026-11-23`, Hyperliquid `xyz:NVDA` and `xyz:TSLA`).

---

## Executive Diagnosis: Why v1 Failed

In the QTAP paper (`4_experimental.tex`, Table 2), the continuous Brownian Bridge + EWMA/Parkinson + Hawkes model achieved **1.8%–5.0%** house edge across all 6 order policies and all 3 regimes (mean **3.26%**) on 1-second Binance BTC/USDT OHLCV bars, with **zero** per-row or per-window correction factors.

On Hyperliquid `xyz` stock perps, v1 broke down — systematically over-predicting rows `dist >= 3` by `+15–22 pp` while under-predicting the live price row (`dist = 0`) — and accumulated layers of patches (`distance_factors`, `far_distance_factors`, `current_row_logit_shift`, `off_hours_m_base`, `sigma_scale`, `p_raw_gamma`). The 7-day tick data shows **three structural mismatches** between v1's assumptions and the actual `xyz` feed:

1. **A 3-second oracle update clock, not 1-second continuous diffusion.** Although `activeAssetCtx` broadcasts at 1 Hz (median `dt = 1,022 ms`), `markPx` updates on a ~3-second cycle: `P(move at t | moved at t-1) = 0.0002` (0.02%), lag-1 move-indicator autocorrelation is `-0.26` (`NVDA`) / `-0.22` (`TSLA`), lag-3 autocorrelation is **`+0.49` / `+0.47`**, and the 1-minute move rate hard-caps at **`1/3 = 0.333`**. A 5-second window contains only **1 or 2 price updates** (expected `5/3 = 1.67`), not 5 seconds of continuous motion.
2. **Whole-cent zero-inflated geometric jumps, not Gaussian log-returns.** `100.00%` of `markPx` prints are exact integer cents (`496,184/496,184` NVDA, `496,180/496,180` TSLA). Conditioned on a volatility regime (`quiet`, `normal`, `volatile`), the non-zero jump size `|ΔC|` is closely **Geometric** (total variation distance `0.017–0.028` for NVDA, `0.033–0.089` for TSLA). The apparent heavy tail in unconditional 7-day data is primarily a mixture across regimes, plus a 2-component tail during `news` (top 1% of minutes).
3. **Print-to-print segment crossing settlement, not continuous Brownian Bridge.** Production settlement (`src/calibrate/eval.rs`) pays iff a print-to-print segment `[min(C_{s-1}, C_s), max(C_{s-1}, C_s)]` intersects `[L, U]` inside `[T_start, T_end)`, with any gap `> 3 s` voiding the window. Continuous Brownian Bridge assigns substantial intra-second touch probability between stationary or adjacent-cent prints that settlement never pays, explaining the `+15–22 pp` over-prediction on outer rows.

v2 replaces the broken continuous bridge and deletes every calibration patch. It models the actual 3-second discrete cent-jump process, computes exact print-crossing touch probabilities under the production settlement rule, defines 4 data-derived scenario presets (`quiet`, `normal`, `volatile`, `news`), and prices every cell directly with `M = (1 - m_eff) / P`.

---

## Notation

Each symbol is defined once here and used consistently throughout.

| Symbol | Meaning |
|---|---|
| `t ∈ ℤ≥0` | Discrete 1-second clock index (`t = 0` is quote time). |
| `C_t ∈ ℤ` | Integer-cent mark price at second `t`: `C_t = round(100 · markPx_t)`. |
| `ΔC_t ∈ ℤ` | 1-second cent change: `ΔC_t = C_t - C_{t-1}`. |
| `I_t ∈ {0, 1}` | Move indicator: `I_t = 1{ΔC_t ≠ 0}`. |
| `T_clk = 3` | Oracle mark-price refresh period in seconds. |
| `φ ∈ {0, 1, 2}` | Phase offset of the 3-second oracle clock relative to quote time `t = 0`. |
| `q ∈ (0, 1/3]` | Unconditional 1-second move probability: `q = P(ΔC_t ≠ 0)`. |
| `p_move ∈ (0, 1]` | Per-clock-tick move probability: `p_move = min(1, T_clk · q) = min(1, 3q)`. |
| `K ∈ ℤ≥1` | Conditional jump magnitude in integer cents: `K = |ΔC_t| | (ΔC_t ≠ 0)`. |
| `p_g ∈ (0, 1]` | Single Geometric parameter on `{1, 2, ...}`: `P(K = k) = (1 - p_g)^{k-1} p_g`, mean `1/p_g`. |
| `(w, p_1, p_2)` | 2-component Geometric mixture for `news`: `P(K = k) = w(1-p_1)^{k-1}p_1 + (1-w)(1-p_2)^{k-1}p_2`. |
| `B ∈ ℤ≥1` | Price-band width in integer cents (`B = 3` for NVDA, `B = 8` for TSLA). |
| `[L, U]` | Target cell price band in integer cents (`U = L + B`), at distance `d` bands from spot. |
| `[T_start, T_end)` | Target 5-second settlement window in seconds after quote time (`T_end = T_start + 5`, `T_start ≥ 5`). |
| `P(c)` | Exact print-crossing touch probability of contract `c = ([L, U], [T_start, T_end))` under v2. |
| `R_60` | Trailing 60-second realized RMS cent move per second: `sqrt((1/n) Σ ΔC_s^2)` over valid `dt ≤ 3 s` steps. |
| `R_ref` | Reference RMS cents/s for the ticker (empirical `normal` regime median `R_60`). |
| `m_base, m_risk, m_eff` | Base margin, dynamic risk premium, and effective margin `m_eff = m_base + m_risk`. |
| `K_skew, K_safety` | Liability skew dampener and CVaR tail-safety factor from QTAP (`3_proposed.tex`). |
| `M(c)` | Final quoted multiplier: `clamp(((1 - m_eff)/max(P(c), ε)) · K_skew · K_safety, M_min, M_max)`. |

---

## 1. Price Process & Empirical Evidence

### 1.1 Dataset & Missing-Time Accounting

Data comes from 7 recorded days (`2026-11-17` through `2026-11-23`) of Hyperliquid `activeAssetCtx` `markPx` logs (`496,184` NVDA ticks, `496,180` TSLA ticks):

- **Price granularity:** `100.00%` exact integer cents on both tickers. Median prices: NVDA `$139.65` (`$124.92–$171.40`, `3c = 2.15 bps`), TSLA `$363.97` (`$344.50–$393.54`, `8c = 2.20 bps`).
- **Broadcast cadence:** `dt_ms` p10/p50/p90 = `944 / 1,022 / 1,120 ms` (identical for both tickers); mean `4.87` prints per 5-second window (p25/p50/p90 = `5 / 5 / 5`).
- **Gaps & valid minutes (`8,491` of `10,080` calendar minutes):** Across 7 calendar days (`10,080` min), `2026-11-17` starts at `05:38 UTC` (`-338` min), the `2026-11-23` snapshot ends at `16:30:46 UTC` (`-450` min), and recorder restarts/outages account for the remaining `~801` missing minutes (`25` gaps `> 3 s` total). Minutes with `< 30` valid 1-second steps are excluded from regime-quantile estimation; `> 3 s` gaps break the 1-second difference chain and void any overlapping 5-second settlement window (`src/calibrate/eval.rs:21-51`).

### 1.2 The 3-Second Oracle Clock (Autocorrelation & Clustering)

Measured across `509,394` (NVDA) and `509,383` (TSLA) contiguous 1-second steps:

| Statistic | NVDA (3c) | TSLA (8c) | Interpretation |
|---|---:|---:|---|
| `P(ΔC_t = 0)` (all / US / OFF) | `0.793 / 0.709 / 0.813` | `0.817 / 0.698 / 0.847` | 70–85% of 1s prints are unchanged |
| `P(I_t = 1 \| I_{t-1} = 1)` | **`0.0002`** | **`0.0002`** | Consecutive 1s moves almost never occur |
| `P(I_t = 1 \| I_{t-1} = 0)` | `0.2616` | `0.2233` | Moves occur only after ≥2 unchanged seconds |
| `Corr(I_t, I_{t-k})` at `k = 1, 2, 3, 30, 60 s` | `-0.26, -0.17, +0.49, +0.30, +0.20` | `-0.22, -0.14, +0.47, +0.30, +0.20` | Periodic 3-second oracle update clock |
| `Corr(ΔC_t, ΔC_{t-k})` at `k = 1, 2, 3, 5 s` | `+0.000, +0.029, +0.088, +0.023` | `-0.000, +0.023, +0.104, +0.006` | Mild positive continuation at the 3s clock |
| `P(same sign on consecutive non-zero moves)` | `0.5525` | `0.5639` | ~5–6 pp directional persistence between updates |
| `P(ΔC_t > 0 \| ΔC_t ≠ 0)` | `0.5161` | `0.5039` | Near-symmetric unconditional direction |
| `Corr(\|ΔC_t\|, \|ΔC_{t-k}\|)` at `k = 3, 30, 60 s` | `+0.322, +0.236, +0.189` | `+0.320, +0.223, +0.174` | Strong multi-minute volatility clustering |
| 1-minute `q` quantiles `p50 / p85 / p99 / max` | `0.233 / 0.300 / 0.333 / 0.350` | `0.200 / 0.300 / 0.333 / 0.350` | **Hard saturation at `1/3 = 0.333`** |

**Key consequence for regime detection:** Because the oracle refreshes every ~3 seconds, `move_rate` (`q`) saturates near `1/3` in `normal`, `volatile`, and `news` alike (in fact, during the `2026-11-23 13:56Z` NVDA news minute, `q = 0.150` while `RMS = 39.05c`!). Therefore, **jump magnitude (`R_60`), not move rate (`q`), separates regimes.**

### 1.3 Conditional Jump-Size Distribution: Empirical vs Fitted

Pooling all 7 days mixes four distinct volatility regimes, creating an artificial heavy tail where a single Geometric fits poorly (`TVD = 0.117` NVDA, `0.126` TSLA) while Discrete Lomax fits best (`TVD = 0.014` NVDA, `0.060` TSLA). **Once conditioned on a regime, the mixture disappears:**

| Ticker / Subset | `q` | `E[\|ΔC\| \| nz]` | Single Geom `p_g` (`TVD`) | 2-Geom Mix (`TVD`) | Disc. Lomax (`TVD`) | Empirical vs Geom `P(K ≥ 5c)` |
|---|---:|---:|---|---|---|---:|
| NVDA All 7d (uncond.) | `0.2074` | `3.08c` | `0.3249` (`0.1172`) | `0.0425` | `0.0142` | `0.174` vs `0.208` |
| NVDA `quiet` | `0.1426` | `1.53c` | **`0.6517` (`0.0277`)** | `0.0241` | `0.0313` | `0.016` vs `0.015` |
| NVDA `normal` | `0.2602` | `2.65c` | **`0.3767` (`0.0213`)** | `0.0213` *(collapses)* | `0.0594` | `0.151` vs `0.151` |
| NVDA `volatile` | `0.3017` | `5.40c` | **`0.1850` (`0.0169`)** | `0.0179` *(collapses)* | `0.0624` | `0.446` vs `0.441` |
| NVDA `news` (top 1%) | `0.2949` | `20.33c` | `0.0492` (`0.1077`) | **`w=0.815, p1=0.0643, p2=0.0242` (`0.0890`)** | `0.1358` | `0.787` vs `0.792` *(2-Geom)* |
| TSLA All 7d (uncond.) | `0.1826` | `4.93c` | `0.2027` (`0.1261`) | `0.0678` | `0.0596` | `0.348` vs `0.404` |
| TSLA `quiet` | `0.1108` | `2.15c` | **`0.4643` (`0.0888`)** | `0.0623` | `0.0497` | `0.101` vs `0.082` |
| TSLA `normal` | `0.2391` | `4.54c` | **`0.2202` (`0.0434`)** | `0.0433` *(collapses)* | `0.0441` | `0.375` vs `0.370` |
| TSLA `volatile` | `0.2900` | `8.45c` | **`0.1183` (`0.0334`)** | `0.0335` *(collapses)* | `0.0587` | `0.597` vs `0.604` |
| TSLA `news` (top 1%) | `0.3027` | `19.67c` | `0.0508` (`0.0969`) | **`w=0.771, p1=0.0645, p2=0.0297` (`0.0888`)** | `0.1220` | `0.788` vs `0.794` *(2-Geom)* |

For **99% of all minutes** (`quiet`, `normal`, `volatile`), the conditional jump size is a **1-parameter Geometric distribution** (the EM 2-component fit collapses onto the single Geometric for both `normal` and `volatile` on both tickers!). Only `news` (top 1%) requires a 2-component Geometric mixture to capture `≥50c` shock prints (`8.8–9.1%` of `news` moves).

### 1.4 Formal v2 Stochastic Process

Within a scenario preset $\theta = (p_{\text{move}}, w, p_1, p_2)$ (with $w = 1, p_1 = p_2 = p_g$ for `quiet`, `normal`, `volatile`), the integer-cent price path $\{C_t\}_{t=0}^{T}$ evolves from current spot $C_0$ as a **3-second-clock Zero-Inflated Discrete Laplace (Geometric-Jump) process**:

1. Draw initial oracle clock phase $\phi \sim \text{Uniform}\{0, 1, 2\}$.
2. For each second $t = 1, 2, \dots, T$:
   - If $(t - \phi) \not\equiv 0 \pmod{3}$: $C_t = C_{t-1}$ (no oracle update; print repeats).
   - If $(t - \phi) \equiv 0 \pmod{3}$:
     $$C_t = C_{t-1} + I_t \cdot S_t \cdot K_t$$
     where:
     - $I_t \sim \text{Bernoulli}(p_{\text{move}})$, with $p_{\text{move}} = \min(1, 3q)$,
     - $S_t \sim \text{Uniform}\{-1, +1\}$ (symmetric; see Open Question 3),
     - $K_t \sim w\,\text{Geom}(p_1) + (1 - w)\,\text{Geom}(p_2)$ on $\{1, 2, \dots\}$, independent across ticks.

The per-tick signed increment PMF is therefore:
$$\pi(0) = 1 - p_{\text{move}}, \qquad \pi(\pm k) = \frac{p_{\text{move}}}{2}\left[w(1-p_1)^{k-1}p_1 + (1-w)(1-p_2)^{k-1}p_2\right] \quad (k \ge 1).$$

---

## 2. Touch Probability Under Exact Settlement

### 2.1 Production Settlement Rule

From `st-engine/src/calibrate/eval.rs:21-84`, a 5-second window $[T_{\text{start}}, T_{\text{end}})$:
- is **void** (stake refunded) if zero prints arrive in $[T_{\text{start}}, T_{\text{end}})$ or if any consecutive pair $(t_{j-1}, t_j)$ overlapping the window has $t_j - t_{j-1} > 3{,}000\text{ ms}$;
- otherwise **hits** band $[L, U]$ (inclusive, in integer cents) iff there exists a print at $t_j \in [T_{\text{start}}, T_{\text{end}})$ whose print-to-print segment intersects $[L, U]$:
  $$\min(C_{j-1}, C_j) \le U \quad \text{and} \quad \max(C_{j-1}, C_j) \ge L.$$

Because the v2 model prices conditional on a non-void window (voids have zero house edge, returning 100% of stake), the exact touch event on a 1-second path $\{C_t\}$ for window $[T_{\text{start}}, T_{\text{end}})$ is:
$$E(c) = \left\{\exists\, s \in \{T_{\text{start}}, \dots, T_{\text{end}}-1\} : \min(C_{s-1}, C_s) \le U \;\wedge\; \max(C_{s-1}, C_s) \ge L\right\}.$$
Note that even the existing `TouchMode::Discrete` (`src/grid/touch.rs:5-44`) only checks point membership $L \le C_s \le U$ and misses jump-overs where $C_{s-1} < L$ and $C_s > U$; v2 uses the exact segment-crossing predicate above.

### 2.2 Exact Algorithm (1-D Markov DP and Discrete Integer-Cent MC)

Because the price changes only every 3 seconds, a 30-second horizon has at most **10–11** jump steps (and a 2-minute horizon at most **41** steps). Two equivalent ways to compute $P(c) = \mathbb{P}(E(c))$:

#### Method A — Exact 1-D Dynamic Programming (zero MC variance)
Let $n(t, \phi) = \left|\{s \in \{1,\dots,t\} : (s - \phi) \equiv 0 \pmod 3\}\right|$ be the number of oracle ticks up to second $t$. For fixed phase $\phi \in \{0, 1, 2\}$:
1. **Before the window:** propagate the state distribution $u_k(x) = \mathbb{P}(C = x \text{ after } k \text{ ticks})$ from $u_0 = \delta_{C_0}$ to $k_0 = n(T_{\text{start}} - 1, \phi)$ via discrete convolution $u_{k+1} = u_k * \pi$.
2. **Inside the window** ($m = n(T_{\text{end}} - 1, \phi) - k_0 \in \{1, 2\}$ ticks): track the **unhit** distribution $v_0(x) = u_{k_0}(x) \cdot \mathbf{1}\{x \notin [L, U]\}$ (if $x \in [L, U]$ at the start of the window, the first tick's segment already starts inside $[L, U]$ and hits immediately!). For each of the $m$ ticks inside $[T_{\text{start}}, T_{\text{end}})$, transition only through **non-crossing** jumps:
   $$v_{j+1}(y) = \sum_{\substack{x \notin [L,U],\; y \notin [L,U] \\ \text{same side of } [L,U]}} v_j(x)\,\pi(y - x),$$
   where "same side" means $(x < L \wedge y < L)$ or $(x > U \wedge y > U)$ — any jump across $[L, U]$ is absorbed as a hit!
3. **Touch probability:** $P(c \mid \phi) = 1 - \sum_y v_m(y)$, and average uniformly over the 3 phases:
   $$P(c) = \frac{1}{3}\sum_{\phi=0}^{2} P(c \mid \phi).$$

#### Method B — Exact Discrete Integer-Cent Monte Carlo (preserves pathwise CVaR for `K_safety`)
```text
Input: spot C_0 (cents), bands {[L_r, U_r]}, windows {[T_s, T_e)}, preset (p_move, w, p_1, p_2), paths N
For n = 1 .. N:
  c <- C_0
  phi <- Uniform{0, 1, 2}
  For each window w_idx: w_lo[w_idx] <- +inf, w_hi[w_idx] <- -inf
  For s = 1 .. T_max:
    prev <- c
    If (s - phi) mod 3 == 0 and Uniform(0,1) < p_move:
      p_use <- p_1 if (w == 1 or Uniform(0,1) < w) else p_2
      K <- 1 + floor(ln(Uniform(0,1)) / ln(1 - p_use))     # exact Geom(p_use)
      c <- c + K if Uniform(0,1) < 0.5 else c - K
    If s falls in window w_idx (T_s <= s < T_e):
      w_lo[w_idx] <- min(w_lo[w_idx], prev, c)
      w_hi[w_idx] <- max(w_hi[w_idx], prev, c)
  For each cell (r, w_idx):
    If w_lo[w_idx] <= U_r and w_hi[w_idx] >= L_r:
      hits[r, w_idx] += 1
      scenario_payout[n] += liability[r, w_idx]
Return P[r, w_idx] = hits[r, w_idx] / N, and scenario_payout[1..N]
```
With `N = 5,000` paths and only ~10–41 integer steps (zero `exp`/`ln` bridge evaluations across 480 cells), Method B runs in `<1 ms` per ticker while remaining compatible with QTAP's pathwise CVaR `K_safety`.

### 2.3 Empirical Validation of v2 vs v1 (Zero Calibration Factors)

Comparing **causal empirical 5-second touch rates** (classified strictly by the *previous* completed minute's regime across `8,491` minutes) against **v2** (no tuning, directly from Section 4 presets) and **v1 raw Brownian Bridge** (using each regime's exact `RMS` cents/s, zero patches):

| Ticker & Regime | Window | Source | `d=0` | `d=1` | `d=2` | `d=3` | `d=4` | `d=5` | Max `\|err\|` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| NVDA `quiet` (3c) | `5–10s` | Empirical | `0.832` | `0.347` | `0.074` | `0.024` | `0.009` | `0.004` | — |
| | | **v2 (3s Geom)** | **`0.860`** | **`0.362`** | **`0.040`** | **`0.003`** | **`0.000`** | **`0.000`** | **`3.4 pp`** |
| | | v1 Raw BB | `0.838` | `0.433` | `0.052` | `0.001` | `0.000` | `0.000` | `8.7 pp` |
| NVDA `normal` (3c) | `5–10s` | Empirical | `0.622` | `0.438` | `0.226` | `0.108` | `0.049` | `0.023` | — |
| | | **v2 (3s Geom)** | **`0.669`** | **`0.483`** | **`0.229`** | **`0.093`** | **`0.036`** | **`0.012`** | **`4.7 pp`** |
| | | v1 Raw BB | `0.695` | `0.585` | `0.351` | `0.155` | `0.050` | `0.013` | `14.7 pp` |
| NVDA `normal` (3c) | `30–35s` | Empirical | `0.283` | `0.259` | `0.214` | `0.165` | `0.121` | `0.087` | — |
| | | **v2 (3s Geom)** | **`0.293`** | **`0.276`** | **`0.239`** | **`0.188`** | **`0.135`** | **`0.089`** | **`2.5 pp`** |
| | | v1 Raw BB | `0.367` | `0.348` | `0.298` | `0.237` | `0.175` | `0.117` | `8.9 pp` |
| NVDA `volatile` (3c) | `5–10s` | Empirical | `0.452` | `0.398` | `0.311` | `0.233` | `0.171` | `0.124` | — |
| | | **v2 (3s Geom)** | **`0.505`** | **`0.455`** | **`0.360`** | **`0.268`** | **`0.188`** | **`0.129`** | **`5.7 pp`** |
| | | v1 Raw BB | `0.601` | `0.577` | `0.517` | `0.436` | `0.342` | `0.253` | `20.6 pp` |
| TSLA `normal` (8c) | `5–10s` | Empirical | `0.703` | `0.340` | `0.116` | `0.042` | `0.017` | `0.007` | — |
| | | **v2 (3s Geom)** | **`0.730`** | **`0.378`** | **`0.104`** | **`0.023`** | **`0.004`** | **`0.001`** | **`3.8 pp`** |
| | | v1 Raw BB | `0.756` | `0.541` | `0.197` | `0.038` | `0.004` | `0.000` | `20.1 pp` |
| TSLA `volatile` (8c) | `5–10s` | Empirical | `0.547` | `0.376` | `0.223` | `0.131` | `0.076` | `0.042` | — |
| | | **v2 (3s Geom)** | **`0.572`** | **`0.433`** | **`0.261`** | **`0.140`** | **`0.071`** | **`0.036`** | **`5.7 pp`** |
| | | v1 Raw BB | `0.651` | `0.590` | `0.443` | `0.282` | `0.148` | `0.069` | `22.0 pp` |
| TSLA `news` (8c) | `30–35s` | Empirical | `0.169` | `0.161` | `0.157` | `0.154` | `0.138` | `0.122` | — |
| | | **v2 (2-Geom)** | **`0.166`** | **`0.165`** | **`0.164`** | **`0.162`** | **`0.154`** | **`0.147`** | **`2.5 pp`** |
| | | v1 Raw BB | `0.275` | `0.273` | `0.271` | `0.264` | `0.257` | `0.249` | `12.8 pp` |

v2 cuts maximum probability error by **3×–5×** (`15–22 pp → 2.5–5.7 pp`) with **zero** calibration factors. The small remaining excess in causal empirical `quiet` outer rows (`d2–d3`: `0.074` vs `0.040` NVDA) comes from **1-minute regime transitions** (`17.1%` of `quiet` minutes transition to `normal` next minute; see Open Question 4).

---

## 3. Pricing Formula

Following the owner's QTAP paper (`3_proposed.tex`, Section 3.4) with **no** per-row, per-window, or per-ticker correction factors, logit shifts, or off-hours margin overrides:

$$M_{\text{base}}(c) = \frac{1 - m_{\text{eff}}}{\max(P(c), \varepsilon)}, \qquad M(c) = \operatorname{clamp}\!\big(M_{\text{base}}(c) \cdot K_{\text{skew}}(c) \cdot K_{\text{safety}},\; M_{\min},\; M_{\max}\big)$$

with $\varepsilon = 10^{-6}$, $M_{\min} = 1.05$, $M_{\max} = 25.0$, and the existing QTAP risk controls:

$$K_{\text{skew}}(c) = \frac{1}{1 + \beta_{\text{skew}} \cdot L_{\text{used}}(c) / C_{\text{pool}}}, \qquad K_{\text{safety}} = \operatorname{clamp}\!\left(\frac{B_{\text{risk}}}{\text{CVaR}_{99}},\, 0,\, 1\right).$$

### Effective Margin $m_{\text{eff}}$

$$m_{\text{eff}} = m_{\text{base}} + m_{\text{risk}}, \qquad m_{\text{risk}} = \operatorname{clamp}\!\left(c_r \cdot \max\!\left(\frac{R_{60}}{R_{\text{ref}}} - 1,\; 0\right),\; 0,\; m_{\text{risk,max}}\right)$$

- **$m_{\text{base}} = 0.025$ (2.5%)** — uniform across all tickers, all sessions (US and off-hours), and all cells, sitting squarely inside the owner's 2–5% target.
- **$c_r = 0.005$, $m_{\text{risk,max}} = 0.025$** — dimensionless volatility risk premium, keeping $m_{\text{eff}} \in [2.5\%, 5.0\%]$. *(Fixes a unit bug in v1 `src/grid/pricing.rs:20`, where `a_sigma * sigma * (sigma/sigma_ref - 1)` multiplied by `sigma ≈ 2e-4`, collapsing `m_risk` to `~0.01%` even in extreme shocks.)*
- **Price-level scaling $f(P)$:** Because v2 simulates directly in integer cents matching the fixed cent band width $B$, price-level geometry is exact at the calibration price. If a ticker's spot price $P$ drifts materially from its preset reference $P_{\text{ref}}$ (e.g. `|P / P_ref - 1| > 15%`), scale the jump-size mean linearly with $P / P_{\text{ref}}$ (`p_g' = 1 / (1 + (1/p_g - 1) · P / P_ref)`) rather than applying an ad-hoc `sigma_scale`.

---

## 4. Scenario Presets & Online Regime Detection

### 4.1 Why Four Presets & Why `R_60` Is the Metric

Across the `8,491` valid minutes of 7-day data, the minute-level realized 1-second RMS cent move `R_60` spans two orders of magnitude (`0.26c → 43.60c` NVDA; `0.22c → 31.34c` TSLA) and cleanly separates four market states using **pure empirical quantiles (`p50 / p85 / p99`)**, not tuned thresholds:

| Regime | Quantile Band | Share of Minutes | NVDA `R_60` Range (`σ_1s` bps) | TSLA `R_60` Range (`σ_1s` bps) | US Regular Share (NVDA / TSLA) | 1-Min Persistence (NVDA / TSLA) |
|---|---|---:|---|---|---:|---:|
| **`quiet`** | `≤ p50` | `50.1%` | `≤ 1.16c` (`≤ 0.80 bp`) | `≤ 1.81c` (`≤ 0.49 bp`) | `2.0% / 1.3%` | `81.6% / 82.4%` |
| **`normal`** | `(p50, p85]` | `34.9%` | `(1.16, 2.41]c` (`0.80–1.68 bp`) | `(1.81, 4.36]c` (`0.49–1.17 bp`) | `26.9% / 26.9%` | `64.4% / 62.0%` |
| **`volatile`** | `(p85, p99]` | `14.0%` | `(2.41, 8.39]c` (`1.68–5.42 bp`) | `(4.36, 11.51]c` (`1.17–3.02 bp`) | `64.2% / 63.8%` | `67.4% / 56.7%` |
| **`news`** | `> p99` | `1.0%` | `> 8.39c` (`> 5.42 bp`) | `> 11.51c` (`> 3.02 bp`) | `43.5% / 80.0%` | `63.5% / 48.2%` |

*(For new tickers like `AAPL` and `SP500` once `/opt/.../hl-ticks-extra` accrues ≥3 days, compute their exact `p50 / p85 / p99` `R_60` cutoffs and MLE preset parameters from their own ticks using the identical formula.)*

### 4.2 Full Data-Derived Preset Parameter Tables

Every parameter below is the direct MLE on that regime's 1-second tape (`logs/calibration/m1-tick-stats.txt`), with $R_{\text{ref}} = \text{median } R_{60}$ of `normal` (`1.60c` NVDA, `2.77c` TSLA):

#### NVDA (`B = 3c`, `P_ref = $140`, `R_ref = 1.60c`)

| Preset | `q` (1s) | `p_move = min(1, 3q)` | Jump Size `K` (cents) | `E[K]` | `R_med` | `m_base` | `m_risk` | `m_eff` |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **`quiet`** | `0.1426` | `0.4278` | `Geom(p_g = 0.6517)` | `1.53c` | `0.58c` | `2.5%` | `0.0%` | **`2.5%`** |
| **`normal`** | `0.2602` | `0.7806` | `Geom(p_g = 0.3767)` | `2.65c` | `1.60c` | `2.5%` | `0.0%` | **`2.5%`** |
| **`volatile`** | `0.3017` | `0.9051` | `Geom(p_g = 0.1850)` | `5.40c` | `3.36c` | `2.5%` | `0.7%` | **`3.2%`** |
| **`news`** | `0.2949` | `0.8847` | `0.815·Geom(0.0643) + 0.185·Geom(0.0242)` | `20.33c` | `11.79c` | `2.5%` | `2.5%` | **`5.0%`** |

#### TSLA (`B = 8c`, `P_ref = $364`, `R_ref = 2.77c`)

| Preset | `q` (1s) | `p_move = min(1, 3q)` | Jump Size `K` (cents) | `E[K]` | `R_med` | `m_base` | `m_risk` | `m_eff` |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **`quiet`** | `0.1108` | `0.3324` | `Geom(p_g = 0.4643)` | `2.15c` | `0.63c` | `2.5%` | `0.0%` | **`2.5%`** |
| **`normal`** | `0.2391` | `0.7173` | `Geom(p_g = 0.2202)` | `4.54c` | `2.77c` | `2.5%` | `0.0%` | **`2.5%`** |
| **`volatile`** | `0.2900` | `0.8700` | `Geom(p_g = 0.1183)` | `8.45c` | `5.63c` | `2.5%` | `0.6%` | **`3.1%`** |
| **`news`** | `0.3027` | `0.9081` | `0.771·Geom(0.0645) + 0.229·Geom(0.0297)` | `19.67c` | `14.20c` | `2.5%` | `2.5%` | **`5.0%`** |

### 4.3 Online Regime-Detection Rule (Strictly Causal)

At every quote second `t`, compute `R_60(t)` over all valid 1-second steps (`dt ≤ 3 s`) in the trailing 60-second buffer `[t - 60, t]`:
- **Base classification:** map `R_60(t)` through the ticker's empirical `p50 / p85 / p99` cutoffs (`quiet / normal / volatile / news`).
- **Fast upward escalation (zero lag on shocks):**
  - Upgrades take effect **immediately** on the first second `R_60(t)` crosses an upper threshold.
  - **Single-print shock trigger:** if a single valid step satisfies `|ΔC_t| > 3 · θ_p99` (`> 25c` NVDA, `> 35c` TSLA — corresponding to `~p99.5` single-second prints), immediately promote to `news` without waiting for `R_60` to accumulate. On `2026-11-23`, NVDA's `13:55Z` (`max1s = 44c`, `R_60 = 5.73c`) and `13:56Z` (`max1s = 143c`, `R_60 = 39.05c`) trigger `news` on the very first shock print, and TSLA's `13:56Z` (`max1s = 96c`) and `14:30Z` open (`max1s = 161c`) do the same.
- **Downgrade hysteresis:** require `R_60(t)` to remain below `0.85 × threshold` for `15` consecutive seconds before stepping down one regime (prevents chattering across boundaries; matches the `57–82%` 1-minute regime persistence).
- **Gap & cold-start handling:** steps with `dt > 3 s` are excluded from `R_60(t)` (never treated as jumps); during feed staleness (`> 3 s` silence), existing stale logic clears all quotes and holds the prior regime; on cold start (`< 30` valid steps), initialize from candle warmup `R_warmup` (or default to `normal`).

### 4.4 Worked Examples (One Cell per Preset)

Evaluated with `K_skew = K_safety = 1.0`, window 0 `[5 s, 10 s)`, `N = 100,000` exact v2 paths (`d = +1` is the first band strictly above spot; `d = 0` shown in parentheses for comparison):

- **NVDA (`B = 3c`, spot `$139.99`, `d = +1` band `[$140.01, $140.04]`, `d = 0` band `[$139.98, $140.01]`):**
  - **`quiet`:** `P(d=+1) = 0.2145`, `m_eff = 2.5%` → **`M = (1 - 0.025) / 0.2145 = 4.55x`** *(price row `d=0`: `P = 0.8942`, `M = 1.09x`)*.
  - **`normal`:** `P(d=+1) = 0.4276`, `m_eff = 2.5%` → **`M = 0.975 / 0.4276 = 2.28x`** *(`d=0`: `P = 0.6818`, `M = 1.43x`)*.
  - **`volatile`:** `P(d=+1) = 0.4376`, `m_eff = 3.2%` → **`M = 0.968 / 0.4376 = 2.21x`** *(`d=0`: `P = 0.5090`, `M = 1.90x`)*.
  - **`news`:** `P(d=+1) = 0.3591`, `m_eff = 5.0%` → **`M = 0.950 / 0.3591 = 2.65x`** *(`d=0`: `P = 0.3961`, `M = 2.40x`)*.
- **TSLA (`B = 8c`, spot `$364.04`, `d = +1` band `[$364.08, $364.16]`, `d = 0` band `[$364.00, $364.08]`):**
  - **`quiet`:** `P(d=+1) = 0.0846`, `m_eff = 2.5%` → **`M = 0.975 / 0.0846 = 11.53x`** *(`d=0`: `P = 0.9668`, `M = 1.05x` floor)*.
  - **`normal`:** `P(d=+1) = 0.3564`, `m_eff = 2.5%` → **`M = 0.975 / 0.3564 = 2.74x`** *(`d=0`: `P = 0.7600`, `M = 1.28x`)*.
  - **`volatile`:** `P(d=+1) = 0.4340`, `m_eff = 3.1%` → **`M = 0.969 / 0.4340 = 2.23x`** *(`d=0`: `P = 0.5871`, `M = 1.65x`)*.
  - **`news`:** `P(d=+1) = 0.3991`, `m_eff = 5.0%` → **`M = 0.950 / 0.3991 = 2.38x`** *(`d=0`: `P = 0.4571`, `M = 2.08x`)*.

Notice the natural physical shape without any ad-hoc shifts: `P(d=0)` falls monotonically from `quiet` to `news`, while `P(d=+1)` peaks in `volatile` and dips in `news` as price diffuses past the adjacent band before `t = 5 s`.

---

## 5. Validation Plan

Before changing any live defaults, validate v2 offline on all 7 recorded days (`2026-11-17`..`2026-11-23`) for NVDA and TSLA, and repeat for AAPL and SP500 once `≥ 3` days accrue in `hl-ticks-extra`:

1. **Causal minute classification:** Label every minute with `≥ 30` valid 1s steps using the **trailing** 60-second `R_60` (zero lookahead). Exclude `> 3 s` void windows using `eval.rs:window_is_void`.
2. **Stage 1 — Static preset validation (per-regime isolation, matching QTAP `4_experimental.tex` Table 2):**
   - For each of the 4 presets (`quiet`, `normal`, `volatile`, `news`), run the 6 QTAP order-selection policies (`closest`, `three_rows`, `row_sweep`, `long_bias`, `short_bias`, `random_near`) plus `momentum` and `chase_far` on **only that preset's classified minutes**, using exact `eval.rs` print-crossing settlement.
   - Run on both 6-window (`30 s`) and 24-window (`2 min`) horizons, reporting pooled edge, US vs OFF edge, 25x-cap share, and calibration Brier/bucket gaps.
3. **Stage 2 — End-to-end online switching validation:**
   - Replay all 7 full days continuously with the causal online detector (Section 4.3, including fast shock escalation and 15 s downgrade hysteresis), plus the dense `2026-11-23` (`13:40–15:00 UTC`) news-shock stress test.
   - Verify no degradation across regime transitions (`quiet → normal`, `normal → news`, `news → volatile`).
4. **Pass Criterion (the owner's Bar):**
   - **Per-regime & pooled:** every QTAP policy's pooled house edge **`> 0%`**, and the 6-policy average in **`2%–5%`** (separately reporting the effect of the `25x` cap on `quiet` outer bands — see Open Question 1).
   - **Shock solvency:** positive combined house P&L on `2026-11-23` `13:56` (60s/300s) and `14:30` US open, with `chase_far` protected.

---

## 6. Removal List (Calibration & Patch Layers Deleted by v2)

v2 deletes every post-hoc probability distortion, correction table, and per-ticker margin hack:

| Layer / Patch to Delete | File & Lines | Why v2 Removes It |
|---|---|---|
| Baked `NVDA_FAR_FACTORS` 18×7 table | `st-engine/src/grid/stock.rs:11-30` | Replaced by v2 discrete process (accurate at `w=0..23` without factors). |
| `set_distance_factors` & `set_far_distance_factors` | `st-engine/src/grid/stock.rs:32-65` | No per-row or per-window multipliers on `P`. |
| Per-ticker `sigma_scale`, `m_base`, `off_hours_m_base`, `current_row_logit_shift`, distance-factor calls | `st-engine/src/grid/stock.rs:75, 95-100, 115-148` | Replaced by uniform `m_base = 0.025`, dimensionless `m_risk`, and ticker scenario presets. |
| `effective_m_base` (`off_hours_m_base` override) | `st-engine/src/grid/pricing.rs:3-9` | Quiet off-hours is a regime preset (`quiet`), not a clock-based margin hack. |
| `p_raw_gamma`, `calibration.factors` lookup, `p_near_boost`, `p_time_boost`, `current/adjacent/zone_vol` logit shifts | `st-engine/src/grid/pricing.rs:51-118, 169-177` | All probability distortion layers deleted; `P(c)` passes straight into `(1 - m_eff) / P(c)`. |
| `CalibrationKey`, `CalibrationConfig`, `factors` map & obsolete `PricingConfig`/`VolatilityConfig` patch fields (`off_hours_m_base`, `p_raw_gamma`, `p_near_boost`, `p_time_boost`, `calibration_dist_max`, `calibration_sigma_edges`, `current_row_logit_shift`, `adjacent_row_logit_shift`, `zone_vol_shift_scale`, `sigma_scale`, `parkinson_weight`, `price_adj_*`) | `st-engine/src/grid/types.rs:306-315, 338, 348-355, 392-410, 462-466` | Replaced by clean `RegimePreset` and `RegimeDetectorConfig` structs. |
| Continuous EWMA `sigma_scale` & `price_adjustment` | `st-engine/src/grid/state.rs:166-167, 193-213` | Replaced by causal `R_60` trailing RMS detector (no Parkinson/EWMA `sigma_scale`). |
| Continuous Brownian Bridge (`window_hit_probability`, `window_hit_probability_log_steps`) & point-only `window_hit_discrete` | `st-engine/src/grid/touch.rs:5-343` | Replaced by exact print-to-print segment-crossing operator matching `eval.rs`. |
| Log-space Brownian Bridge path simulator | `st-engine/src/grid/simulator.rs` (continuous path & bridge loops) | Replaced by 3-second-clock integer-cent jump DP/MC (Section 2.2). |
| Factor-window clamping in `QuoteEngine` | `st-engine/src/quote/mod.rs:253, 267-276` | No factor table to look up. |
| Runtime factor-table parser/applicator (`parse_factor_table`, `apply_factor_table`, `row_dist_bucket`) | `st-engine/src/horizon.rs:127-190` | No external calibration factor files in v2. |
| Calibration-factor / distortion CLI flags (`--distance-factor`, `--factors-file`, `--sigma-scale`, `--p-raw-gamma`, `--p-time-boost`, `--current-shift`, `--adjacent-shift`, `--off-hours-m-base`) | `st-engine/src/bin/calibrate.rs:21-31, 95-115, 177-230` and `st-engine/src/bin/policy_backtest.rs:40-42, 55, 115-120, 155, 213-239` | Replaced by `--preset <quiet|normal|volatile|news|auto>` validation flags. |

---

## 7. Open Questions for the owner (Paper vs Data Disagreements)

1. **The `25x` cap vs the `2–5%` average edge bar in `quiet`:** On Binance BTC (`4_experimental.tex`), volatility was high enough relative to the `$10` band that `three_rows` and `random` rarely hit the `25x` cap (`+2.8%` to `+4.5%`). On Hyperliquid `xyz` `quiet` minutes (50% of the day, mostly off-hours), NVDA `3c` has `P(d=2) = 0.04` and TSLA `8c` has `P(d=2) = 0.002–0.02`, so `M_max = 25x` automatically forces `+50%` to `+95%` house edge on `three_rows` and `random` **even when `P(c)` is 100% exact**. Should we (a) adapt band width `B` by regime (e.g. `1–2c` in `quiet`), (b) raise `M_max` or filter unplayable `<4%` cells, or (c) evaluate the `2–5%` bar on uncapped cells while keeping `M_max = 25x` as a tail-risk cap?
2. **Oracle clock phase `φ` — infer or average?** Because `markPx` updates every ~3 seconds, the engine can observe when the last non-zero print occurred (knowing `φ`), which changes `w = 0` (`5–10 s`) between 1 and 2 expected updates. However, a sharp user watching the same feed also knows when the last update happened. Should v2 average uniformly over `φ ∈ {0, 1, 2}` (smoother multipliers, slight phase exposure) or condition on the observed time since last print `τ_since ∈ {0, 1, 2}` (exact conditional probability, eliminates phase sniping)?
3. **Mild 3-second sign persistence (`55.3%` NVDA, `56.4%` TSLA):** The QTAP paper assumes independent symmetric steps (`0.5 / 0.5`). The `xyz` feed has `Corr(ΔC_t, ΔC_{t-3}) = +0.088` (NVDA) / `+0.104` (TSLA), which is why `momentum` won on directional days in E10–E11. Should v2 keep symmetric `0.5 / 0.5` steps and let `m_eff ∈ [2.5%, 5.0%]` absorb the small autocorrelation, or include a first-order Markov sign-persistence parameter `ρ_sign` in the jump step?
4. **Within-horizon regime switching (`quiet → normal` is `16–17%/min`):** Static-preset simulation (Section 2.3) matches `normal`, `volatile`, and `news` within `2.5–5.7 pp`, but under-predicts causal `quiet` outer bands (`d2–d3`) slightly when a `quiet` minute wakes up into `normal` inside the next 5–120 s. Should v2 stay purely stationary within each preset (simpler, matches `4_experimental.tex`) or allow a 1-step Markov regime transition along the path for multi-window horizons (especially 2-minute `24w`)?
5. **Exact 1-D DP vs integer-cent MC:** Exact 1-D DP (Section 2.2 Method A) eliminates Monte Carlo variance completely and runs in `<0.5 ms` even for 24 windows, while Method B (integer-cent MC) naturally produces pathwise `scenario_payouts` for QTAP's CVaR `K_safety`. Do you prefer pure integer-cent MC (`N = 5,000`), or exact DP for `P(c)` with lightweight MC only when open liability is non-zero?
6. **`news` preset vs E10 tap pause (`k > 10`):** With the single-print shock trigger (`|ΔC_t| > 3 · θ_p99`), v2 promotes immediately to the `news` preset (`m_eff = 5.0%`, 2-Geom heavy tail). Should we still keep E10's 60-second `k > 10` tap pause as an outer circuit-breaker, or let the `news` preset + `K_safety` handle shock minutes without halting taps?
