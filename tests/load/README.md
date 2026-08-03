# k6 WebSocket Load Test (10k Sockets)

This folder contains a profile-aware k6 script for validating websocket behavior and latency at 10,000 concurrent socket sessions against `GET /ws`.

## Script

- `tests/load/k6_ws_10k.js`

## Profiles

- `transport_only`: benchmark websocket transport/control-plane pressure with training tunnel off by default.
- `mixed_train_infer`: benchmark mixed inference + training pressure, with training tunnel on by default.

Set with `-e LOAD_PROFILE=transport_only` or `-e LOAD_PROFILE=mixed_train_infer`.

## Install k6 (Windows)

```powershell
winget install k6.k6
```

Alternative:

```powershell
choco install k6
```

## Quick Smoke Run

```bash
k6 run tests/load/k6_ws_10k.js \
  -e WS_URL=ws://127.0.0.1:8000/ws \
  -e MAX_SOCKETS=300 \
  -e STAGE_STEP_SECONDS=30 \
  -e HOLD_STAGE_SECONDS=90
```

## Full 10k Run (Transport-Only Profile)

```bash
k6 run tests/load/k6_ws_10k.js \
  -e WS_URL=ws://127.0.0.1:8000/ws \
  -e LOAD_PROFILE=transport_only \
  -e MAX_SOCKETS=10000 \
  -e STAGE_STEP_SECONDS=180 \
  -e HOLD_STAGE_SECONDS=900 \
  -e RAMP_DOWN_SECONDS=180 \
  -e STATUS_PROBE_RATIO=0.10 \
  -e ENABLE_TRAIN_TUNNEL=false
```

## Full 10k Run (Mixed Inference + Training Profile)

```bash
k6 run tests/load/k6_ws_10k.js \
  -e WS_URL=ws://127.0.0.1:8000/ws \
  -e LOAD_PROFILE=mixed_train_infer \
  -e MAX_SOCKETS=10000 \
  -e STAGE_STEP_SECONDS=180 \
  -e HOLD_STAGE_SECONDS=900 \
  -e RAMP_DOWN_SECONDS=180 \
  -e STATUS_PROBE_RATIO=0.10 \
  -e ENABLE_TRAIN_TUNNEL=true
```

## Target SLA Thresholds (10k Concurrency)

These are encoded directly in the script under `options.thresholds` and selected by `LOAD_PROFILE`.

### `transport_only`

| Metric | Target |
| --- | --- |
| `checks` (HTTP 101 upgrade) | `rate >= 99.5%` |
| `handshake_failure_rate` | `<= 0.5%` |
| `session_completed_rate` | `>= 98.5%` |
| `ready_message_rate` | `>= 99.5%` |
| `subscribed_message_rate` | `>= 99.5%` |
| `status_message_rate` (sampled sockets) | `>= 98%` |
| `app_error_rate` | `<= 1.0%` |
| `parse_error_rate` | `<= 0.1%` |
| `unexpected_close_rate` | `<= 1.0%` |
| `ws_connecting` | `p95 < 1200 ms`, `p99 < 2500 ms` |
| `subscribe_ack_ms` | `p95 < 400 ms`, `p99 < 900 ms` |
| `status_rtt_ms` | `p95 < 600 ms`, `p99 < 1200 ms` |

### `mixed_train_infer`

| Metric | Target |
| --- | --- |
| `checks` (HTTP 101 upgrade) | `rate >= 99.0%` |
| `handshake_failure_rate` | `<= 1.0%` |
| `session_completed_rate` | `>= 97.0%` |
| `ready_message_rate` | `>= 99.0%` |
| `subscribed_message_rate` | `>= 99.0%` |
| `status_message_rate` (sampled sockets) | `>= 96%` |
| `app_error_rate` | `<= 2.0%` |
| `parse_error_rate` | `<= 0.1%` |
| `unexpected_close_rate` | `<= 2.0%` |
| `ws_connecting` | `p95 < 1800 ms`, `p99 < 3200 ms` |
| `subscribe_ack_ms` | `p95 < 900 ms`, `p99 < 1800 ms` |
| `status_rtt_ms` | `p95 < 1200 ms`, `p99 < 2200 ms` |

## Notes

- `ENABLE_TRAIN_TUNNEL` defaults to `false` for `transport_only`, and `true` for `mixed_train_infer`.
- Keep `STATUS_PROBE_RATIO` between `0.05` and `0.20` to sample control-plane latency without overloading status calls.
- A true 10k run may need distributed k6 generators depending on CPU, memory, and file descriptor limits.
