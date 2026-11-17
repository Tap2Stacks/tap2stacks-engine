# Parity Fixtures

Thư mục này chứa fixture dùng để đối chiếu Rust với implementation Python trong `tap2stacks-model`.

## Naming

Mỗi case dùng 2 file:

- `name.input.json`
- `name.expected.json`

`expected` là optional. Nếu chưa có file expected, test harness vẫn chạy case đó để đảm bảo Rust quote thành công và shape grid hợp lệ.

## Input schema

```json
{
  "spot": 50000.0,
  "history": [
    { "open": 50000.0, "high": 50020.0, "low": 49980.0, "close": 50010.0 }
  ],
  "config": {
    "simulation": {
      "mc_n_min": 512,
      "mc_n_max": 512,
      "adaptive_mc": false,
      "use_antithetic": true
    },
    "volatility": {
      "jump": { "enabled": false }
    }
  },
  "liabilities": [0.0, 0.0],
  "use_internal_liability_book": false
}
```

## Expected schema

```json
{
  "anchor": 50000.0,
  "volatility": {
    "sigma": 0.0019,
    "sigma_raw": 0.0054,
    "jump_intensity": 0.02
  },
  "cells": [
    {
      "row_index": 9,
      "window_index": 0,
      "hit_probability": 0.412,
      "multiplier": 2.38
    }
  ],
  "tolerances": {
    "default_abs": 0.000001,
    "default_rel": 0.01,
    "hit_probability_abs": 0.01,
    "multiplier_rel": 0.03
  }
}
```

## Recommended rollout

1. thêm deterministic fixtures trước: tắt jump, fixed MC size
2. xuất expected từ Python cho `anchor`, `sigma`, vài cell representative
3. thêm fixture có liabilities để test `k_skew` / `k_safety`
4. thêm fixture `use_internal_liability_book=true` nếu muốn parity cho `liability_rho`

## Generate expected from Python

Có thể dùng:

```bash
python3 scripts/generate_parity_expected.py fixtures/parity/mode10_smoke.input.json
```

Hoặc generate toàn bộ fixture:

```bash
python3 scripts/generate_parity_expected.py
```
