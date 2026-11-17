# Port Plan: Bring `tap2stacks-rs` in Line with `tap2stacks-model`

## Mục tiêu

Tài liệu này mô tả trạng thái port hiện tại của Rust so với implementation chuẩn Python ở:

- [tap2stacks-model/fortress/engine.py](tap2stacks-model/fortress/engine.py)

Mục tiêu không phải chỉ "giống docs", mà là giống behavior của bản Python đang được xem là implementation chuẩn của thuật toán pricing/grid.

---

## 1. Status hiện tại

### Đã port xong cho single-mode `10`

- `4.1` refactor config/state model
- `4.2` grid construction theo Python MC branch
- `4.3` state estimator gần Python
- `4.4` jump sampler empirical memory
- `4.5` path simulation theo `mu + sigma * eps + jump`
- `4.6` adaptive Monte Carlo
- `4.7.1` `m_eff`
- `4.7.2` distance tempering
- `4.7.3` near boost
- `4.7.4` time boost
- `4.7.5` liability skew
- `4.7.6` safety factor
- `4.7.7` calibration
- `4.7.8` zone logit shifts

### Khoảng cách lớn còn lại so với Python

- chưa có `multi-mode`
- `k_safety` đã là scenario/tail-based, nhưng chưa parity tuyệt đối với bản Python vectorized
- chưa có parity tests trực tiếp với fixture/backtest của Python

---

## 2. Kiến trúc Rust hiện tại

Các file hiện tại:

- [src/grid/mod.rs](src/grid/mod.rs)
- [src/grid/types.rs](src/grid/types.rs)
- [src/grid/state.rs](src/grid/state.rs)
- [src/grid/simulator.rs](src/grid/simulator.rs)
- [src/grid/touch.rs](src/grid/touch.rs)
- [src/grid/pricing.rs](src/grid/pricing.rs)

### Các khối chính

- `state`: EWMA + Parkinson + adaptive jump threshold + Hawkes-like intensity
- `simulator`: path generation + adaptive MC + scenario sampling
- `touch`: Brownian Bridge hit probability và window aggregation
- `pricing`: quote shaping + multiplier transforms
- `mod`: orchestration, grid build, liability-aware quoting

### Config/state model hiện tại

Rust đã tách config/state thành:

- `GridSpec`
- `SimulationConfig`
- `VolatilityConfig`
- `PricingConfig`
- `CalibrationConfig`
- `RuntimeState`

Điều này gần hơn với Python:

- `ModeConfig`
- `GlobalConfig`
- `ModeState`

Nhưng Rust hiện vẫn là mode đơn, chưa phải multi-mode engine.

---

## 3. Những gì đã port

## 3.1 Grid construction

Đã port theo Python MC branch:

- `center_row = (rows - 1) / 2`
- `anchor = floor(price / band_width) * band_width`
- grid kiểu `center-straddling`

Tham chiếu:

- [src/grid/mod.rs](src/grid/mod.rs)
- [tap2stacks-model/fortress/engine.py:514](tap2stacks-model/fortress/engine.py#L514)

## 3.2 State estimator

Đã có:

- `sigma_ref`
- adaptive `kappa_s`
- `price_adj`
- `sigma_raw`
- `sigma`
- `jump_flag`
- `jump_intensity`
- stateful update qua `update_runtime_state(...)`

Tham chiếu:

- [src/grid/state.rs](src/grid/state.rs)
- [tap2stacks-model/fortress/engine.py:227](tap2stacks-model/fortress/engine.py#L227)

## 3.3 Jump sampler

Đã có:

- `JumpSamplerState`
- `record(log_return)`
- `sample(n, rng, fallback_sigma)`

Tham chiếu:

- [src/grid/types.rs](src/grid/types.rs)
- [tap2stacks-model/fortress/engine.py:91](tap2stacks-model/fortress/engine.py#L91)

## 3.4 Path simulation

Đã port phần core:

- `returns = mu + sigma * eps`
- `p_jump = 1 - exp(-lambda_intensity)`
- cộng jump sampled vào return
- tích lũy trên log-price

Đã bỏ Hawkes update nội-path.

Tham chiếu:

- [src/grid/simulator.rs](src/grid/simulator.rs)
- [tap2stacks-model/fortress/engine.py:652](tap2stacks-model/fortress/engine.py#L652)

## 3.5 Adaptive Monte Carlo

Đã có:

- `mc_n_min`
- `mc_n_max`
- `se_abs`
- `se_rel`
- `p_floor`
- `adaptive_mc`

Simulator chạy theo batch và dừng sớm nếu mọi cell đạt threshold.

## 3.6 Touch probability

Đã có:

- Brownian Bridge hit probability theo interval
- `P_raw = 1 - Π(1 - P_hit_t)` cho từng window

Tham chiếu:

- [src/grid/touch.rs](src/grid/touch.rs)
- [tap2stacks-model/fortress/engine.py:744](tap2stacks-model/fortress/engine.py#L744)

## 3.7 Quote shaping / pricing transforms

Đã có:

- `m_eff = m_base + m_risk`
- `p_raw_gamma`
- `p_near_boost`
- `p_time_boost`
- `k_skew`
- `k_safety`
- calibration
- zone logit shifts

Tham chiếu:

- [src/grid/pricing.rs](src/grid/pricing.rs)
- [tap2stacks-model/fortress/engine.py:365](tap2stacks-model/fortress/engine.py#L365)

---

## 4. Những điểm còn chưa parity tuyệt đối

## 4.1 Multi-mode

Python là multi-mode engine:

- mode `10`
- mode `50`
- mode `100`

Rust hiện chỉ port mode đơn, tương đương mode `10`.

## 4.2 Antithetic variates

Status: done

Rust hiện đã port:

- Gaussian shock pair `eps` và `-eps`
- reuse cùng jump sequence cho path đối xứng
- batching vẫn tương thích với adaptive MC

Tham chiếu:

- [src/grid/simulator.rs](src/grid/simulator.rs)
- [tap2stacks-model/fortress/engine.py:652](tap2stacks-model/fortress/engine.py#L652)

## 4.3 Liability smoothing

Status: done cho single-mode engine

Rust hiện đã có:

- `LiabilityBook`
- `GridEngine` stateful
- `liability_used` memory theo cell
- smoothing:
  - nếu `l_inst >= l_prev` thì dùng `l_inst`
  - nếu `l_inst < l_prev` thì dùng
    `liability_rho * l_prev + (1 - liability_rho) * l_inst`

Lưu ý:

- API stateless `quote_grid_with_liabilities(...)` vẫn giữ behavior external/raw
- smoothing chỉ áp trong `GridEngine`, tương ứng với internal liability state của Python

## 4.4 Safety factor exact parity

Rust hiện đã nâng `k_safety` từ proxy `total_liability / risk_budget` lên bản scenario-based:

- simulator giữ `sampled_wins`
- pricing tính tail payout từ scenarios

Tuy nhiên vẫn chưa parity tuyệt đối với Python vì:

- Python dùng `win_matrix` vectorized từ `pwin_matrix`
- Python lấy `q99` và `CVaR` trên phân phối payout từ scenario engine gốc

Rust đã cùng loại mô hình, nhưng chưa chắc cùng số liệu path-for-path.

## 4.5 Exact parity tests

Rust hiện đã có unit tests cho:

- grid alignment
- sampler behavior
- `m_eff`
- `k_skew`
- `k_safety`
- gamma / near / time / calibration / zone shift

Nhưng chưa có:

- parity test trực tiếp với fixture Python
- đối chiếu `P_raw` / `M_final` trên cùng data feed

---

## 5. Những mục đã hoàn tất trong plan gốc

### 5.1 `4.1` Refactor config/state model

Status: done

### 5.2 `4.2` Grid construction

Status: done theo Python MC branch

### 5.3 `4.3` State estimator

Status: done ở mức single-mode usable

### 5.4 `4.4` Jump sampler

Status: done

### 5.5 `4.5` Path simulation

Status: done phần core

Gap còn lại:

- exact parity path-for-path với Python vectorized

### 5.6 `4.6` Adaptive Monte Carlo

Status: done

### 5.7 `4.7.1` Effective margin / risk loading

Status: done

### 5.8 `4.7.2` Distance tempering

Status: done

### 5.9 `4.7.3` Near boost

Status: done

### 5.10 `4.7.4` Time boost

Status: done

### 5.11 `4.7.5` Liability skew

Status: done, gồm cả external liability input và internal smoothing qua `GridEngine`

### 5.12 `4.7.6` Safety factor

Status: done ở mức scenario/tail-based gần Python

### 5.13 `4.7.7` Calibration

Status: done ở mức single-mode

Key hiện tại:

- `(window_index, dist_bucket, sigma_bucket)`

Không có `mode_id` vì engine hiện là mode đơn.

### 5.14 `4.7.8` Zone logit shifts

Status: done

---

## 6. Roadmap tiếp theo

Nếu tiếp tục đưa Rust sát Python hơn, thứ tự có giá trị cao nhất là:

1. thêm parity tests với fixture Python
2. xem xét exact parity cho `k_safety`
3. nếu cần thì mới mở sang `multi-mode`

---

## 7. Kết luận

Rust hiện tại không còn chỉ là scaffold. Nó đã port được phần lớn pipeline single-mode của Python mode `10`.

Các khối đã usable:

- state estimation
- path simulation
- Brownian Bridge window probability
- adaptive MC
- pricing transforms
- liability-aware quoting
- calibration
- zone shaping

Khoảng cách còn lại so với Python không còn nằm ở các công thức lõi chính, mà nằm ở:

- exact parity / validation
- multi-mode support
