FROM rust:1.86-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY tests ./tests
COPY fixtures ./fixtures

RUN cargo build --release --bin ws_market_server

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/ws_market_server /usr/local/bin/ws_market_server

ENV RUST_LOG=info
ENV WS_BIND_ADDR=0.0.0.0:8080
EXPOSE 8080

CMD ["ws_market_server"]
