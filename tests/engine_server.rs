use std::collections::BTreeSet;
use std::net::TcpListener as StdTcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use st_engine::protocol::{EngineIn, EngineOut, ExposureCell};
use st_engine::quote::{StockCell, StockQuote};
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::StatusCode;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};

struct ChildGuard {
    child: Child,
}

impl ChildGuard {
    fn spawn(mut cmd: Command) -> Self {
        let child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn engine_server binary");
        Self { child }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn idle_engine_slot() -> std::sync::MutexGuard<'static, ()> {
    static SLOT: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SLOT.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn free_addr() -> String {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().unwrap().to_string()
}

async fn connect_with_token(
    addr: &str,
    token: Option<&str>,
) -> Result<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    WsError,
> {
    let url = format!("ws://{addr}");
    let mut req = url.into_client_request().unwrap();
    if let Some(tok) = token {
        req.headers_mut()
            .insert("x-engine-token", tok.parse().unwrap());
    }
    let (ws, _) = connect_async(req).await?;
    Ok(ws)
}

async fn wait_and_connect(
    addr: &str,
    token: &str,
) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>
{
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match connect_with_token(addr, Some(token)).await {
            Ok(ws) => return ws,
            Err(WsError::Io(_)) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            Err(err) => panic!("failed to connect to engine_server at {addr}: {err:?}"),
        }
    }
}

async fn wait_for_port_ready(addr: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for engine_server to listen on {addr}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[test]
fn quote_serialises_with_exact_wire_names() {
    let sample = EngineOut::Sample {
        ticker: "NVDA".into(),
        ts_ms: 1,
        price: 157.53,
    };
    let v = serde_json::to_value(sample).unwrap();
    assert_eq!(
        v,
        json!({"type":"sample","ticker":"NVDA","tsMs":1,"price":157.53})
    );

    let quote = EngineOut::Quote(StockQuote {
        ticker: "NVDA".into(),
        server_ts_ms: 1_795_000_000_100,
        spot: 157.53,
        band_cents: 3,
        cells: vec![StockCell {
            window_start_ms: 1_795_000_010_000,
            window_end_ms: 1_795_000_015_000,
            lower_cents: 15_750,
            upper_cents: 15_753,
            hit_probability: 0.41,
            multiplier: 2.31,
        }],
    });
    let qv = serde_json::to_value(quote).unwrap();
    assert_eq!(
        qv,
        json!({
            "type": "quote",
            "ticker": "NVDA",
            "serverTsMs": 1_795_000_000_100u64,
            "spot": 157.53,
            "bandCents": 3,
            "cells": [{
                "windowStartMs": 1_795_000_010_000u64,
                "windowEndMs": 1_795_000_015_000u64,
                "lowerCents": 15_750,
                "upperCents": 15_753,
                "hitProbability": 0.41,
                "multiplier": 2.31
            }]
        })
    );

    let health = EngineOut::Health {
        ticker: "NVDA".into(),
        server_ts_ms: 1_795_000_001_000,
        last_sample_age_ms: Some(127),
        stale: false,
        prev_day_px: None,
    };
    let hv = serde_json::to_value(health).unwrap();
    assert_eq!(
        hv,
        json!({
            "type": "health",
            "ticker": "NVDA",
            "serverTsMs": 1_795_000_001_000u64,
            "lastSampleAgeMs": 127,
            "stale": false,
            "prevDayPx": null
        })
    );
}

#[test]
fn health_frame_serializes_prev_day_px() {
    let health = EngineOut::Health {
        ticker: "NVDA".into(),
        server_ts_ms: 1_795_000_001_000,
        last_sample_age_ms: Some(127),
        stale: false,
        prev_day_px: Some(154.03),
    };
    let hv = serde_json::to_value(health).unwrap();
    assert_eq!(hv["prevDayPx"], 154.03);
}

#[test]
fn exposure_parses_from_backend_json() {
    let raw = r#"{"type":"exposure","ticker":"NVDA","cells":[{"windowStartMs":1795000010000,"lowerCents":15750,"liability":57.75}]}"#;
    let parsed: EngineIn = serde_json::from_str(raw).unwrap();
    assert_eq!(
        parsed,
        EngineIn::Exposure {
            ticker: "NVDA".into(),
            cells: vec![ExposureCell {
                window_start_ms: 1_795_000_010_000,
                lower_cents: 15_750,
                liability: 57.75,
            }],
        }
    );
}

#[test]
fn refuses_to_start_without_engine_token() {
    let output = Command::new(env!("CARGO_BIN_EXE_engine_server"))
        .env_remove("ENGINE_TOKEN")
        .env("ENGINE_BIND_ADDR", free_addr())
        .output()
        .expect("failed to execute engine_server");
    assert!(
        !output.status.success(),
        "engine_server must refuse to start without ENGINE_TOKEN"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("ENGINE_TOKEN"),
        "expected error mentioning ENGINE_TOKEN, got: {stderr}"
    );
}

#[tokio::test]
async fn rejects_connection_without_token() {
    let addr = free_addr();
    let token = "test-secret-token-123";
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_engine_server"));
    cmd.env("ENGINE_BIND_ADDR", &addr)
        .env("ENGINE_TOKEN", token)
        .env("ENGINE_REPLAY_FILE", "tests/fixtures/hl_sample.jsonl");
    let _slot = idle_engine_slot();
    let _guard = ChildGuard::spawn(cmd);

    wait_for_port_ready(&addr).await;

    // Missing header -> HTTP 401
    let missing_err = connect_with_token(&addr, None)
        .await
        .expect_err("connection without x-engine-token must fail");
    match missing_err {
        WsError::Http(resp) => assert_eq!(resp.status(), StatusCode::UNAUTHORIZED),
        other => panic!("expected HTTP 401 rejection, got: {other:?}"),
    }

    // Wrong token -> HTTP 401
    let wrong_err = connect_with_token(&addr, Some("wrong-token"))
        .await
        .expect_err("connection with wrong x-engine-token must fail");
    match wrong_err {
        WsError::Http(resp) => assert_eq!(resp.status(), StatusCode::UNAUTHORIZED),
        other => panic!("expected HTTP 401 rejection, got: {other:?}"),
    }

    // Valid token -> accepted
    let mut ws = connect_with_token(&addr, Some(token))
        .await
        .expect("valid token must succeed");
    let _ = ws.close(None).await;
}

#[tokio::test]
async fn replay_client_gets_quote_within_200ms_and_exposure_moves_it() {
    let addr = free_addr();
    let token = "replay-test-token";
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_engine_server"));
    cmd.env("ENGINE_BIND_ADDR", &addr)
        .env("ENGINE_TOKEN", token)
        .env("ENGINE_TICKERS", "NVDA=xyz:NVDA:3,TSLA=xyz:TSLA:8")
        .env("ENGINE_REPLAY_FILE", "tests/fixtures/hl_sample.jsonl");
    let _slot = idle_engine_slot();
    let _guard = ChildGuard::spawn(cmd);

    let mut ws = wait_and_connect(&addr, token).await;

    // Receive non-empty NVDA quote within 200 ms of connect (allow brief startup jitter if needed,
    // actually connect happens after bind, so measure from first message or from connect).
    let start = Instant::now();
    let first_quote: StockQuote = tokio::time::timeout(Duration::from_millis(200), async {
        while let Some(msg) = ws.next().await {
            let msg = msg.expect("ws read error");
            if let Message::Text(text) = msg {
                let parsed: Value = serde_json::from_str(&text).unwrap();
                if parsed["type"] == "quote" && parsed["ticker"] == "NVDA" {
                    let q: EngineOut = serde_json::from_str(&text).unwrap();
                    if let EngineOut::Quote(q) = q {
                        if !q.cells.is_empty() {
                            return q;
                        }
                    }
                }
            }
        }
        panic!("ws stream ended before quote");
    })
    .await
    .expect("expected non-empty NVDA quote within 200ms");
    assert!(
        start.elapsed() <= Duration::from_millis(200),
        "quote took {:?}",
        start.elapsed()
    );
    assert_eq!(first_quote.band_cents, 3);
    assert!(!first_quote.cells.is_empty());

    // Pick a target cell whose multiplier is well above floor 1.05 so adding exposure moves it down.
    let target = first_quote
        .cells
        .iter()
        .find(|c| c.multiplier > 1.2 && c.multiplier < 25.0)
        .expect("expected a cell with multiplier in (1.2, 25.0)")
        .clone();

    let exposure_msg = EngineIn::Exposure {
        ticker: "NVDA".into(),
        cells: vec![ExposureCell {
            window_start_ms: target.window_start_ms,
            lower_cents: target.lower_cents,
            liability: 5_000.0,
        }],
    };
    ws.send(Message::Text(
        serde_json::to_string(&exposure_msg).unwrap().into(),
    ))
    .await
    .unwrap();

    // Wait up to 500ms for a subsequent NVDA quote where the targeted cell's multiplier dropped.
    let updated_mult = tokio::time::timeout(Duration::from_millis(500), async {
        while let Some(msg) = ws.next().await {
            let msg = msg.expect("ws read error");
            if let Message::Text(text) = msg {
                if let Ok(EngineOut::Quote(q)) = serde_json::from_str::<EngineOut>(&text) {
                    if q.ticker == "NVDA" {
                        if let Some(cell) = q.cells.iter().find(|c| {
                            c.window_start_ms == target.window_start_ms
                                && c.lower_cents == target.lower_cents
                        }) {
                            if cell.multiplier < target.multiplier - 1e-6 {
                                return cell.multiplier;
                            }
                        }
                    }
                }
            }
        }
        panic!("ws stream ended before exposure affected quote");
    })
    .await
    .expect("expected exposure update to lower target cell multiplier");

    assert!(
        updated_mult < target.multiplier,
        "expected {updated_mult} < {}",
        target.multiplier
    );

    // Also verify sample and health frames arrive over the stream.
    let mut saw_sample = false;
    let mut saw_health = false;
    tokio::time::timeout(Duration::from_millis(1_500), async {
        while let Some(msg) = ws.next().await {
            if let Message::Text(text) = msg.expect("ws read error") {
                match serde_json::from_str::<EngineOut>(&text).unwrap() {
                    EngineOut::Sample { ticker, price, .. } if ticker == "NVDA" && price > 0.0 => {
                        saw_sample = true;
                    }
                    EngineOut::Health { ticker, stale, .. } if ticker == "NVDA" && !stale => {
                        saw_health = true;
                    }
                    _ => {}
                }
                if saw_sample && saw_health {
                    break;
                }
            }
        }
    })
    .await
    .expect("expected sample and health messages in replay mode");
    assert!(saw_sample);
    assert!(saw_health);
}

#[tokio::test]
async fn slow_client_does_not_block_fast_client() {
    let addr = free_addr();
    let token = "slow-client-token";
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_engine_server"));
    cmd.env("ENGINE_BIND_ADDR", &addr)
        .env("ENGINE_TOKEN", token)
        .env("ENGINE_TICKERS", "NVDA=xyz:NVDA:3")
        .env("ENGINE_REPLAY_FILE", "tests/fixtures/hl_sample.jsonl");
    let _slot = idle_engine_slot();
    let _guard = ChildGuard::spawn(cmd);

    // Connect a slow client and intentionally never read from it.
    let _slow_ws = wait_and_connect(&addr, token).await;

    // Connect a fast client and verify it still gets multiple quotes promptly.
    let mut fast_ws = connect_with_token(&addr, Some(token))
        .await
        .expect("fast client must connect");

    let mut quote_count = 0usize;
    tokio::time::timeout(Duration::from_millis(600), async {
        while let Some(msg) = fast_ws.next().await {
            if let Message::Text(text) = msg.expect("ws read error") {
                if let Ok(EngineOut::Quote(q)) = serde_json::from_str::<EngineOut>(&text) {
                    if q.ticker == "NVDA" && !q.cells.is_empty() {
                        quote_count += 1;
                        if quote_count >= 3 {
                            break;
                        }
                    }
                }
            }
        }
    })
    .await
    .expect("fast client must keep receiving quotes while slow client stalls");
    assert!(quote_count >= 3);
}

#[tokio::test]
async fn stale_ticker_sends_empty_cells_and_stale_health() {
    // Mock Hyperliquid WS server that sends one NVDA tick and then goes silent.
    let hl_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hl_addr = hl_listener.local_addr().unwrap();

    let hl_server = tokio::spawn(async move {
        let (stream, _) = hl_listener.accept().await.unwrap();
        let ws = accept_async(stream).await.unwrap();
        let (mut write, mut read) = ws.split();

        // Read the subscribe message for xyz:NVDA
        let _sub = read.next().await;

        // Send one NVDA mark price tick, then stay open and silent > 3s.
        write
            .send(Message::Text(
                r#"{"channel":"activeAssetCtx","data":{"coin":"xyz:NVDA","ctx":{"markPx":"157.53"}}}"#
                    .into(),
            ))
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_secs(6)).await;
    });

    let addr = free_addr();
    let token = "stale-test-token";
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_engine_server"));
    cmd.env("ENGINE_BIND_ADDR", &addr)
        .env("ENGINE_TOKEN", token)
        .env("ENGINE_TICKERS", "NVDA=xyz:NVDA:3")
        .env("HL_WS_URL", format!("ws://{hl_addr}"));
    let _slot = idle_engine_slot();
    let _guard = ChildGuard::spawn(cmd);

    let mut ws = wait_and_connect(&addr, token).await;

    // 1. Wait for initial non-empty quote and fresh health.
    let mut saw_fresh_quote = false;
    tokio::time::timeout(Duration::from_secs(2), async {
        while let Some(msg) = ws.next().await {
            if let Message::Text(text) = msg.unwrap() {
                if let Ok(EngineOut::Quote(q)) = serde_json::from_str::<EngineOut>(&text) {
                    if q.ticker == "NVDA" && !q.cells.is_empty() {
                        saw_fresh_quote = true;
                        break;
                    }
                }
            }
        }
    })
    .await
    .expect("expected initial non-empty NVDA quote");
    assert!(saw_fresh_quote);

    // 2. After > 3s of silence, engine must emit stale health and empty-cells quote.
    let mut saw_stale_health = false;
    let mut saw_empty_quote_while_stale = false;
    tokio::time::timeout(Duration::from_secs(6), async {
        while let Some(msg) = ws.next().await {
            if let Message::Text(text) = msg.unwrap() {
                match serde_json::from_str::<EngineOut>(&text) {
                    Ok(EngineOut::Health {
                        ticker,
                        stale,
                        last_sample_age_ms,
                        ..
                    }) if ticker == "NVDA" && stale => {
                        assert!(
                            last_sample_age_ms.is_some_and(|age| age > 3_000),
                            "expected lastSampleAgeMs > 3000 when stale, got {last_sample_age_ms:?}"
                        );
                        saw_stale_health = true;
                    }
                    Ok(EngineOut::Quote(q))
                        if q.ticker == "NVDA" && saw_stale_health && q.cells.is_empty() =>
                    {
                        saw_empty_quote_while_stale = true;
                    }
                    _ => {}
                }
                if saw_stale_health && saw_empty_quote_while_stale {
                    break;
                }
            }
        }
    })
    .await
    .expect("expected stale health and empty cells quote after 3s silence");

    assert!(saw_stale_health);
    assert!(saw_empty_quote_while_stale);
    hl_server.abort();
}

#[tokio::test]
async fn health_frame_reports_prev_day_px_from_hyperliquid() {
    let hl_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hl_addr = hl_listener.local_addr().unwrap();

    let hl_server = tokio::spawn(async move {
        let (stream, _) = hl_listener.accept().await.unwrap();
        let ws = accept_async(stream).await.unwrap();
        let (mut write, mut read) = ws.split();

        let _sub = read.next().await;

        write
            .send(Message::Text(
                r#"{"channel":"activeAssetCtx","data":{"coin":"xyz:NVDA","ctx":{"markPx":"154.68","prevDayPx":"154.03","oraclePx":"154.70"}}}"#
                    .into(),
            ))
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_secs(3)).await;
    });

    let addr = free_addr();
    let token = "prev-day-px-token";
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_engine_server"));
    cmd.env("ENGINE_BIND_ADDR", &addr)
        .env("ENGINE_TOKEN", token)
        .env("ENGINE_TICKERS", "NVDA=xyz:NVDA:3")
        .env("HL_WS_URL", format!("ws://{hl_addr}"));
    let _slot = idle_engine_slot();
    let _guard = ChildGuard::spawn(cmd);

    let mut ws = wait_and_connect(&addr, token).await;

    let observed_prev_day_px = tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(msg) = ws.next().await {
            if let Message::Text(text) = msg.unwrap() {
                if let Ok(EngineOut::Health {
                    ticker,
                    prev_day_px: Some(px),
                    ..
                }) = serde_json::from_str::<EngineOut>(&text)
                {
                    if ticker == "NVDA" {
                        return px;
                    }
                }
            }
        }
        panic!("ws stream ended before health frame with prevDayPx");
    })
    .await
    .expect("expected NVDA health frame with prevDayPx");

    assert_eq!(observed_prev_day_px, 154.03);
    hl_server.abort();
}

#[test]
fn refuses_to_start_with_invalid_quote_windows_env() {
    let cases = [
        (
            vec![("ENGINE_QUOTE_WINDOWS_BY_TICKER", "MSFT=24")],
            "MSFT",
        ),
        (
            vec![("ENGINE_QUOTE_WINDOWS_BY_TICKER", "NVDA=0")],
            "ENGINE_QUOTE_WINDOWS_BY_TICKER",
        ),
        (
            vec![("ENGINE_QUOTE_WINDOWS_BY_TICKER", "NVDA=abc")],
            "ENGINE_QUOTE_WINDOWS_BY_TICKER",
        ),
        (vec![("ENGINE_QUOTE_WINDOWS", "0")], "ENGINE_QUOTE_WINDOWS"),
    ];

    for (envs, expected_substr) in cases {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_engine_server"));
        cmd.env("ENGINE_TOKEN", "bad-windows-token")
            .env("ENGINE_BIND_ADDR", free_addr())
            .env("ENGINE_TICKERS", "NVDA=xyz:NVDA:3,TSLA=xyz:TSLA:8")
            .env_remove("ENGINE_QUOTE_WINDOWS")
            .env_remove("ENGINE_QUOTE_WINDOWS_BY_TICKER");
        for (k, v) in &envs {
            cmd.env(k, v);
        }
        let output = cmd.output().expect("failed to execute engine_server");
        assert!(
            !output.status.success(),
            "engine_server must refuse to start for {envs:?}"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(expected_substr),
            "expected stderr for {envs:?} to contain '{expected_substr}', got: {stderr}"
        );
    }
}

#[tokio::test]
async fn per_ticker_quote_windows_gives_nvda_24_and_tsla_6() {
    let addr = free_addr();
    let token = "per-ticker-windows-token";
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_engine_server"));
    cmd.env("ENGINE_BIND_ADDR", &addr)
        .env("ENGINE_TOKEN", token)
        .env("ENGINE_TICKERS", "NVDA=xyz:NVDA:3,TSLA=xyz:TSLA:8")
        .env_remove("ENGINE_QUOTE_WINDOWS")
        .env("ENGINE_QUOTE_WINDOWS_BY_TICKER", "NVDA=24")
        .env("ENGINE_REPLAY_FILE", "tests/fixtures/hl_sample.jsonl");
    let _slot = idle_engine_slot();
    let _guard = ChildGuard::spawn(cmd);

    let mut ws = wait_and_connect(&addr, token).await;

    let mut nvda_24: Option<StockQuote> = None;
    let mut tsla_6: Option<StockQuote> = None;
    tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(msg) = ws.next().await {
            if let Message::Text(text) = msg.expect("ws read error") {
                if let Ok(EngineOut::Quote(q)) = serde_json::from_str::<EngineOut>(&text) {
                    if q.cells.is_empty() {
                        continue;
                    }
                    let windows: BTreeSet<u64> =
                        q.cells.iter().map(|cell| cell.window_start_ms).collect();
                    match q.ticker.as_str() {
                        "NVDA" => {
                            assert!(
                                windows.len() <= 24,
                                "NVDA exceeded 24 windows: {}",
                                windows.len()
                            );
                            if windows.len() == 24 && nvda_24.is_none() {
                                nvda_24 = Some(q);
                            }
                        }
                        "TSLA" => {
                            assert!(
                                windows.len() <= 6,
                                "TSLA must stay at <= 6 windows, got {}",
                                windows.len()
                            );
                            if windows.len() == 6 && tsla_6.is_none() {
                                tsla_6 = Some(q);
                            }
                        }
                        _ => {}
                    }
                    if nvda_24.is_some() && tsla_6.is_some() {
                        break;
                    }
                }
            }
        }
    })
    .await
    .expect("expected 24-window NVDA quote and 6-window TSLA quote within 3s");

    let nvda = nvda_24.expect("missing 24-window NVDA quote");
    let tsla = tsla_6.expect("missing 6-window TSLA quote");
    assert_eq!(nvda.band_cents, 3);
    assert_eq!(nvda.cells.len(), 20 * 24);
    assert_eq!(tsla.band_cents, 8);
    assert_eq!(tsla.cells.len(), 20 * 6);
}
