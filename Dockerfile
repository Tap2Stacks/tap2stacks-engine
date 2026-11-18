FROM rust:1.99-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY tests ./tests
COPY fixtures ./fixtures

RUN cargo build --release --bin engine_server

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/engine_server /usr/local/bin/engine_server

ENV RUST_LOG=info
ENV ENGINE_BIND_ADDR=[::]:8090
EXPOSE 8090

CMD ["engine_server"]
