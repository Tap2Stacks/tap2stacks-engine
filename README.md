# tap2stacks engine

The price and odds engine behind **tap2stacks**, a tap-to-predict game on live stock prices, built on Base.

Players see a live chart for a stock (NVDA, TSLA) with a grid of cells: each cell is a price row and a 5-second time window. Tapping a cell stakes test USDC on "the price touches this row during this window". This engine ingests the price feed, computes the probability of every cell being touched, turns it into a multiplier, and streams the board to the backend ten times a second.

| Repo | Role |
|---|---|
| [tap2stacks-web](https://github.com/Tap2Stacks/tap2stacks-web) | Player app (React, wallets, live board) |
| [tap2stacks-backend](https://github.com/Tap2Stacks/tap2stacks-backend) | Accounts, taps, settlement, proofs, on-chain money flows |
| **tap2stacks-engine** | Price feed, odds model, quote stream (this repo) |
| [tap2stacks-contracts](https://github.com/Tap2Stacks/tap2stacks-contracts) | Vault, test USDC and proof anchor on Base Sepolia |

## What it does

- **Price feeds.** Hyperliquid trade.xyz perps (`xyz:NVDA`, `xyz:TSLA`, the current default) and the Coinbase Derivatives (Deribit) perpetual index at 1 s (`ENGINE_PRICE_SOURCE=deribit`). The Deribit feed emits one sample per second: new index prints immediately, and the last value is carried forward only while the socket is proven alive by a 1 s ping.
- **Odds.** For every cell the engine computes the probability that the price path touches the row inside the window. The straight segment between two consecutive prices counts, and a price on a row edge touches both rows. The multiplier is
  `M = clamp((1 − m_eff) / P × K_skew × K_safety, 1.05, cap)`,
  where `m_eff` is the house margin for the current market preset, and `K_skew` / `K_safety` protect against one-sided exposure.
- **Model v2.** An exact dynamic programme over whole-cent price steps on the feed's clock (3 s for Hyperliquid, 1 s for the index). It has four data-fitted presets (quiet, normal, volatile, news) chosen from the last 60 s of moves. On the 1 s index it adds two measured terms: the perp order book leads the index by about 3 s, so the board starts from where the index is heading; and steps before a window differ from steps inside it. US regular hours and off hours use separate fitted tables. The maths is in [`docs/`](docs).
- **Quote stream.** A WebSocket server pushes `quote`, `sample` and `health` frames to the backend. Clients authenticate with the `x-engine-token` header.

## Run

Rust (edition 2024).

```sh
cargo test --release            # unit + integration tests
cargo run --release --bin engine_server
```

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE_TOKEN` | (required) | Shared secret the backend sends in `x-engine-token` |
| `ENGINE_BIND_ADDR` | `127.0.0.1:8090` | WebSocket listen address |
| `ENGINE_TICKERS` | NVDA, TSLA, AAPL, SP500 | `TICKER=coin:row_cents`, comma separated |
| `ENGINE_PRICE_SOURCE` | `hyperliquid` | `hyperliquid` or `deribit` |
| `ENGINE_PRICING_MODEL` | `v1` | `v1` or `v2` (`m4` is experimental) |
| `ENGINE_MAX_MULTIPLIER` | `25` | Multiplier cap (at least 1.05) |
| `ENGINE_QUOTE_WINDOWS` | `6` | Number of 5-second windows quoted |
| `ENGINE_REPLAY_FILE` | unset | Replay a recorded JSONL feed instead of the live one |
| `HL_WS_URL`, `HL_INFO_URL`, `DERIBIT_WS_URL` | public endpoints | Feed endpoints |

A `Dockerfile` and `fly.toml` are included for container deployment.

## Research and validation tools

| Binary | Purpose |
|---|---|
| `policy_backtest` | Replays recorded feeds and plays nine player styles against the live quote path, reporting the house edge per style, preset and window. Supports `--source deribit`, `--v2-params-file`, `--dump-v2-params` and `--emit-rust-table`. |
| `calibrate` | Accuracy of predicted against realised touch rates on recorded data |
| `bench_horizon`, `bench_grid` | Quote latency benchmarks |

The pass rule used for any odds change: every player style keeps a positive pooled house edge, with an average around 2-5%.

## Repository rules

Commits use the project identity and are checked by hooks and CI. See [`AGENTS.md`](AGENTS.md).
