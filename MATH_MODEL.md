# Mô hình toán học: One-Touch Settlement

---

## 1. Tổng quan

Player đặt cược: giá BTC sẽ **chạm** vào band [L, R] trong window [T_start, T_end].

**Thắng nếu:** bất kỳ bar nào trong window có `high ≥ L` VÀ `low ≤ R`.

```
Input:  Giá hiện tại P₀, band [L, R], window [T_start, T_end]
Output: Multiplier M (đặt $1, thắng $M)
```

---

## 2. Pricing: Brownian Bridge Monte Carlo

### Bước 1: Mô phỏng đường giá

Tạo N đường giá ngẫu nhiên (N = 5,000-10,000) theo mô hình:

```
P_{t+1} = P_t × exp(σ × Z_t + J_t)

Z_t ~ N(0,1)           # Biến động thường
J_t ~ Jump(λ, sampler)  # Jump (Hawkes process, tần suất λ)
σ = EWMA volatility      # Cập nhật mỗi giây từ H/L data
```

### Bước 2: Brownian Bridge hit probability

Với mỗi cặp (P_start, P_end) liên tiếp trên đường giá, tính xác suất bar H/L chạm band [L, R]:

```
Nếu P_start hoặc P_end nằm trong [L, R]: P_hit = 1.0

Nếu cả hai dưới L:
    P_hit = exp(-2 × ln(L/P_start) × ln(L/P_end) / σ²)

Nếu cả hai trên R:
    P_hit = exp(-2 × ln(R/P_start) × ln(R/P_end) / σ²)
```

Đây là công thức Brownian Bridge chuẩn — tính xác suất giá liên tục vượt qua mức L hoặc R giữa 2 điểm đã biết.

### Bước 3: One-touch trên window

```
P_not_touch = Π(1 - P_hit_t)  cho t ∈ [T_start, T_end]
P_raw = 1 - P_not_touch
```

Trung bình P_raw qua N đường → P_raw cho cell đó.

### Bước 4: Tính multiplier

```
m_eff = m_base + m_risk     # margin (2-2.5%)
M_base = (1 - m_eff) / P_raw
K_skew = liability adjustment  # giảm M khi cell bị bet nhiều
K_safety = CVaR safety factor  # giới hạn rủi ro tail

M_final = clamp(M_base × K_skew × K_safety, m_min, m_max)
```

- `m_min = 1.05` — sàn multiplier, không cell nào dưới 1.05x
- `m_max = 25.0` — trần multiplier

---

## 3. Grid structure

### Boundary-aligned bands

```
anchor = round(price / band_width) × band_width
Row i: L = anchor + (i - center_row) × band_width
       R = L + band_width
```

20 rows × 5 windows = 100 cells. $10 bands, $200 tổng phạm vi.

### Betable rule

Cell betable nếu `T_start ≥ 5s` từ thời điểm hiện tại. Tất cả price ranges đều betable.

### Per-window multiplier

Mỗi cột dùng window khác nhau:

| Cột | Window | Ý nghĩa |
|-----|--------|---------|
| 0 | (5, 10) | Chạm trong giây 5-10 |
| 1 | (10, 15) | Chạm trong giây 10-15 |
| 2 | (15, 20) | Chạm trong giây 15-20 |
| 3 | (20, 25) | Chạm trong giây 20-25 |
| 4 | (25, 30) | Chạm trong giây 25-30 |

Window xa hơn ≠ dễ hơn. One-touch P phụ thuộc vào khoảng cách giá phải di chuyển VÀ thời gian còn lại.

---

## 4. Settlement

Khi window kết thúc:

```python
for bar in range(lock + T_start, lock + T_end):
    if highs[bar] >= L and lows[bar] <= R:
        return WIN  # payout = stake × M_final

return LOSE  # mất stake
```

Dùng OHLC high/low — không phải real-time tick. Chống manipulation vì cần toàn bộ bar H/L, không chỉ 1 tick.

---

## 5. Tại sao house lãi

### Brownian Bridge overestimate

BB giả sử giá là Brownian motion liên tục. Thực tế BTC di chuyển rời rạc (discrete 1s bars). BB **overestimates** touch probability → M thấp hơn fair → house lãi.

| Metric | BB Model | Thực tế |
|--------|----------|---------|
| P (cell gần nhất) | ~0.85 | ~0.65 |
| M (implied) | 1.15x | Fair 1.54x |
| House charge | 1.15x | — |
| Player gets | 0.65 × 1.15 = $0.75 | — |
| **House edge** | — | **+25%** |

### m_min floor

Khi P_raw rất cao (>0.95), M_base < 1.05. Floor `m_min=1.05` đảm bảo không cell nào vô dụng.

---

## 6. Volatility model

### EWMA variance

```
v_obs = (1-w) × r² + w × parkinson_var
v_ewma = (1-decay) × v_obs + decay × v_prev
σ = sqrt(v_ewma) × sigma_scale
```

- `decay = exp(-ln2 / hl_seconds)` — half-life 30s
- `parkinson_var = ln(H/L)² / (4·ln2)` — H/L-based vol
- `sigma_scale = 0.35` — scale factor

### Jump detection (Hawkes)

```
z = |return| / σ
jump = (z > κ)
λ_next = μ₀ + exp(-β) × (λ_prev - μ₀) + α × jump
```

Jumps cluster (Hawkes process) — sau 1 jump, xác suất jump tiếp cao hơn. MC sử dụng jump trong simulation.
