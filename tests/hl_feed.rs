use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use st_engine::price::{
    HyperliquidConfig, read_hl_jsonl, spawn_hyperliquid_stream,
};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

#[test]
fn replay_filters_by_coin_and_skips_bbo() {
    let path = Path::new("tests/fixtures/hl_sample.jsonl");
    let nvda = read_hl_jsonl(path, "xyz:NVDA").unwrap();
    let tsla = read_hl_jsonl(path, "xyz:TSLA").unwrap();

    assert_eq!(nvda.len(), 10);
    assert_eq!(tsla.len(), 6);
    assert_eq!(nvda[0].ts_ms, 1_794_893_915_364);
    assert_eq!(nvda[0].price, 157.53);
    assert_eq!(nvda[9].price, 157.59);
    assert_eq!(tsla[0].ts_ms, 1_794_893_915_396);
    assert_eq!(tsla[0].price, 379.3);
}

#[tokio::test]
async fn stream_subscribes_emits_ticks_pings_and_detects_silent_stale() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let ws = accept_async(stream).await.unwrap();
        let (mut write, mut read) = ws.split();

        let mut subs = Vec::new();
        for _ in 0..2 {
            let msg = read.next().await.unwrap().unwrap();
            let text = msg.into_text().unwrap();
            let val: Value = serde_json::from_str(&text).unwrap();
            subs.push(val);
        }

        // Send a BBO frame (must be ignored) followed by a mark price frame.
        write
            .send(Message::Text(
                r#"{"channel":"bbo","data":{"coin":"xyz:NVDA"}}"#.into(),
            ))
            .await
            .unwrap();
        write
            .send(Message::Text(
                r#"{"channel":"activeAssetCtx","data":{"coin":"xyz:NVDA","ctx":{"markPx":"157.53"}}}"#
                    .into(),
            ))
            .await
            .unwrap();

        // Keep socket open and silent long enough for a keepalive ping to arrive.
        let ping_msg = tokio::time::timeout(Duration::from_secs(2), read.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let ping_val: Value = serde_json::from_str(&ping_msg.into_text().unwrap()).unwrap();

        // Hold socket open briefly so client sees silence after ping too.
        tokio::time::sleep(Duration::from_millis(250)).await;
        (subs, ping_val)
    });

    let (nvda_tx, mut nvda_rx) = mpsc::channel(8);
    let (tsla_tx, _tsla_rx) = mpsc::channel(8);
    let mut senders = HashMap::new();
    senders.insert("xyz:NVDA".to_string(), nvda_tx);
    senders.insert("xyz:TSLA".to_string(), tsla_tx);

    let config = HyperliquidConfig {
        ws_url: format!("ws://{addr}"),
        coins: vec!["xyz:NVDA".to_string(), "xyz:TSLA".to_string()],
        stale_after: Duration::from_millis(120),
        health_interval: Duration::from_millis(25),
        ping_interval: Duration::from_millis(100),
        reconnect_base_delay: Duration::from_millis(50),
        reconnect_max_delay: Duration::from_millis(200),
    };

    let mut handle = spawn_hyperliquid_stream(config, senders);

    let tick = tokio::time::timeout(Duration::from_secs(2), nvda_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(tick.price, 157.53);

    // Wait for health_rx to flip stale while the socket stays open and silent.
    let became_stale = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            {
                let health = handle.health_rx.borrow();
                if health.get("xyz:NVDA").is_some_and(|h| h.stale) {
                    return true;
                }
            }
            if handle.health_rx.changed().await.is_err() {
                return false;
            }
        }
    })
    .await
    .unwrap();
    assert!(became_stale);

    let (subs, ping_val) = server_handle.await.unwrap();
    assert_eq!(subs.len(), 2);
    assert_eq!(subs[0]["method"], "subscribe");
    assert_eq!(subs[0]["subscription"]["type"], "activeAssetCtx");
    assert_eq!(subs[0]["subscription"]["coin"], "xyz:NVDA");
    assert_eq!(subs[1]["subscription"]["coin"], "xyz:TSLA");
    assert_eq!(ping_val["method"], "ping");

    handle.join_handle.abort();
}
