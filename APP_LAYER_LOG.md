# App Layer Log

## Mục tiêu

Tách một `app layer` trung gian giữa:

- `price` module: ingest và aggregate market data realtime
- `grid` module: quote engine thuần
- transport layer sau này: websocket broadcast cho user

Mục tiêu của app layer là:

- subscribe latest state từ `price`
- chạy task quote `grid` riêng
- để transport layer sau này merge `price + grid` rồi forward cho user

## Quyết định kiến trúc

### 1. `price` module vẫn giữ `mpsc` chỉ cho ingress nội bộ

`mpsc<PriceTick>` chỉ dùng cho flow:

- Binance/raw market data task
- đẩy tick thô vào `PriceWorker`

Nó không phải interface chính cho module khác consume.

### 2. `price -> app` dùng `watch`

App layer không cần từng tick thô.
Nó chỉ cần latest state.

Hiện app layer consume:

- `watch<PriceSnapshot>`
- `watch<Option<GridMarketSnapshot>>`

### 3. `app` module hiện chỉ giữ `GridWorker`

#### `GridWorker`

Input:

- `watch<Option<GridMarketSnapshot>>`

Output:

- `watch<Option<GridSnapshot>>`

Trách nhiệm:

- mỗi khi market snapshot đổi thì gọi `GridEngine::quote_grid(...)`
- publish latest grid snapshot

## Kiểu dữ liệu chính

### `AppMarketView`

Chứa:

- `latest_price`
- `latest_price_ts_ms`
- `completed_bar_count`
- `grid: Option<GridSnapshot>`

Đây là DTO/helper để transport layer hoặc demo runtime tự merge local state.

## Lý do chọn `watch`

Cho outbound state:

- app/websocket thường chỉ cần latest snapshot
- không cần backlog toàn bộ history updates
- nếu consumer chậm thì chỉ lấy giá trị mới nhất là đủ

Nên `watch` phù hợp hơn `mpsc` cho:

- `price -> app`
- `grid -> app`

## Phần chưa làm

- auth/session handling
- rate limit outbound
- health aggregation ở app layer

## WebSocket transport

Đã thêm binary:

- `src/bin/ws_market_server.rs`

Thiết kế transport hiện tại:

- subscribe trực tiếp:
  - `watch<PriceSnapshot>`
  - `watch<Option<GridSnapshot>>`
  - `watch<BinanceStreamHealth>`
- merge local bằng `build_view(...)`
- serialize DTO JSON outbound
- push ra từng websocket client theo interval ngắn

Điểm quan trọng:

- không có `watch<AppMarketView>` trung gian
- websocket layer chính là consumer merge cuối cùng
- điều này giữ pipeline đơn giản hơn cho giai đoạn hiện tại

## Trạng thái hiện tại

Đã có:

- `src/app/mod.rs`
- `src/app/types.rs`
- `src/app/runtime.rs`
- `src/bin/app_market_demo.rs`

Và đã có test cơ bản xác nhận:

- app pipeline tạo được latest grid snapshot
- latest price và latest grid được merge đúng

## Demo runtime

Đã thêm binary demo:

- `cargo run --release --bin app_market_demo`

Demo này chạy full chain:

- Binance aggTrade stream
- `price` pipeline
- `app` pipeline
- app loop tự merge `price snapshot + grid snapshot`
- log ra `AppMarketView` latest state

Mục đích:

- cho thấy shape dữ liệu mà websocket layer sau này sẽ broadcast
- không cần thêm `watch<AppMarketView>` trung gian nếu transport layer là consumer duy nhất
