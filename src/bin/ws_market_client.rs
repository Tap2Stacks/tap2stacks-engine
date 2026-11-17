use std::env;
use std::error::Error;

use futures_util::StreamExt;
use serde_json::Value;
use st_engine::logging::init_tracing;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info, warn};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    init_tracing("info");

    let ws_url = env::var("WS_CLIENT_URL").unwrap_or_else(|_| "ws://127.0.0.1:8080".to_string());
    let pretty = env_flag("WS_CLIENT_PRETTY", true);

    info!(%ws_url, pretty, "connecting websocket client");
    let (socket, _) = connect_async(&ws_url).await?;
    let (_writer, mut reader) = socket.split();

    while let Some(message) = reader.next().await {
        match message {
            Ok(Message::Text(text)) => {
                print_payload(text.as_ref(), pretty);
            }
            Ok(Message::Binary(bytes)) => {
                if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                    print_payload(&text, pretty);
                } else {
                    warn!("received non-utf8 binary payload");
                }
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {}
            Ok(Message::Close(frame)) => {
                info!(?frame, "websocket server closed connection");
                break;
            }
            Ok(Message::Frame(_)) => {}
            Err(error) => {
                return Err(Box::new(error) as Box<dyn Error + Send + Sync>);
            }
        }
    }

    Ok(())
}

fn print_payload(raw: &str, pretty: bool) {
    match serde_json::from_str::<Value>(raw) {
        Ok(value) if pretty => match serde_json::to_string_pretty(&value) {
            Ok(pretty_json) => println!("{pretty_json}"),
            Err(_) => println!("{raw}"),
        },
        Ok(_) => println!("{raw}"),
        Err(error) => {
            debug!(%error, "failed to parse payload as json");
            println!("{raw}");
        }
    }
}

fn env_flag(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}
