"""
FINSENT NET PRO — WebSocket Handler
Real-time price streaming via WebSocket connections.
"""

import asyncio
import json
import logging
import os
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Set
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("finsent.websocket")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNNEL_DATA_DIR = os.path.join(BASE_DIR, "data", "tunnels")
os.makedirs(TUNNEL_DATA_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
        if out != out:  # NaN guard
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class ConnectionManager:
    """Manages active WebSocket connections for real-time streaming."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        # Per websocket: {ticker: {market, capital, risk_tolerance, enable_train_tunnel}}
        self.subscriptions: Dict[WebSocket, Dict[str, Dict[str, Any]]] = {}
        self._ticker_subscribers: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._ticker_config_cache: List[Dict[str, Any]] = []
        self._ticker_config_cache_dirty = True

        self.max_subscriptions_per_socket = max(
            1,
            _safe_int(os.getenv("WS_MAX_SUBSCRIPTIONS_PER_SOCKET"), 64),
        )
        self.broadcast_chunk_size = max(
            16,
            _safe_int(os.getenv("WS_BROADCAST_CHUNK_SIZE"), 256),
        )

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = {}
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        ws_subs = self.subscriptions.pop(websocket, None) or {}
        for ticker in ws_subs.keys():
            targets = self._ticker_subscribers.get(ticker)
            if not targets:
                continue
            targets.discard(websocket)
            if not targets:
                self._ticker_subscribers.pop(ticker, None)
        self._ticker_config_cache_dirty = True
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)

    def subscribe(
        self,
        websocket: WebSocket,
        tickers: List[str],
        market: str = "SP500",
        capital: float = 100000.0,
        risk_tolerance: float = 0.5,
        enable_train_tunnel: bool = False,
    ) -> List[str]:
        if websocket not in self.subscriptions:
            return []

        ws_subs = self.subscriptions[websocket]
        safe_market = (market or "SP500").upper().strip()
        safe_capital = max(1000.0, _safe_float(capital, 100000.0))
        safe_risk = min(1.0, max(0.1, _safe_float(risk_tolerance, 0.5)))
        remaining_slots = max(0, self.max_subscriptions_per_socket - len(ws_subs))
        changed = False

        for raw_ticker in tickers:
            ticker = str(raw_ticker or "").upper().strip()
            if not ticker:
                continue

            is_new = ticker not in ws_subs
            if is_new and remaining_slots <= 0:
                continue

            ws_subs[ticker] = {
                "market": safe_market,
                "capital": safe_capital,
                "risk_tolerance": safe_risk,
                "enable_train_tunnel": bool(enable_train_tunnel),
                "subscribed_at": _now_iso(),
            }

            if is_new:
                self._ticker_subscribers[ticker].add(websocket)
                remaining_slots -= 1
            changed = True

        if changed:
            self._ticker_config_cache_dirty = True

        return sorted(ws_subs.keys())

    def unsubscribe(self, websocket: WebSocket, tickers: List[str]) -> List[str]:
        if websocket not in self.subscriptions:
            return []

        ws_subs = self.subscriptions[websocket]
        changed = False
        for raw_ticker in tickers:
            ticker = str(raw_ticker or "").upper().strip()
            if not ticker:
                continue

            if ticker in ws_subs:
                ws_subs.pop(ticker, None)
                targets = self._ticker_subscribers.get(ticker)
                if targets:
                    targets.discard(websocket)
                    if not targets:
                        self._ticker_subscribers.pop(ticker, None)
                changed = True

        if changed:
            self._ticker_config_cache_dirty = True

        return sorted(ws_subs.keys())

    def get_ticker_configs(self) -> List[Dict[str, Any]]:
        """Return unique ticker configs aggregated across all clients."""
        if not self._ticker_config_cache_dirty:
            return list(self._ticker_config_cache)

        merged: Dict[str, Dict[str, Any]] = {}
        for ws_config in self.subscriptions.values():
            for ticker, meta in ws_config.items():
                if ticker not in merged:
                    merged[ticker] = {
                        "ticker": ticker,
                        "market": meta.get("market", "SP500"),
                        "capital": _safe_float(meta.get("capital"), 100000.0),
                        "risk_tolerance": _safe_float(meta.get("risk_tolerance"), 0.5),
                        "enable_train_tunnel": bool(meta.get("enable_train_tunnel", False)),
                        "subscriber_count": 1,
                    }
                else:
                    merged[ticker]["subscriber_count"] += 1
                    merged[ticker]["enable_train_tunnel"] = (
                        merged[ticker]["enable_train_tunnel"]
                        or bool(meta.get("enable_train_tunnel", False))
                    )

        self._ticker_config_cache = list(merged.values())
        self._ticker_config_cache_dirty = False
        return list(self._ticker_config_cache)

    async def _broadcast_ticker_message(self, ticker: str, message: Dict[str, Any]):
        targets = tuple(self._ticker_subscribers.get(ticker, set()))
        if not targets:
            return

        payload = json.dumps(message, separators=(",", ":"), default=str)
        dead: List[WebSocket] = []

        for start in range(0, len(targets), self.broadcast_chunk_size):
            chunk = targets[start:start + self.broadcast_chunk_size]
            results = await asyncio.gather(
                *(ws.send_text(payload) for ws in chunk),
                return_exceptions=True,
            )
            for ws, result in zip(chunk, results):
                if isinstance(result, Exception):
                    dead.append(ws)

        for ws in set(dead):
            self.disconnect(ws)

    async def broadcast_live_tunnel(self, ticker: str, payload: Dict[str, Any]):
        await self._broadcast_ticker_message(ticker, {
            "type": "live_tunnel",
            "data": payload,
        })

    async def broadcast_prediction_tunnel(self, ticker: str, payload: Dict[str, Any]):
        await self._broadcast_ticker_message(ticker, {
            "type": "predict_tunnel",
            "data": payload,
        })

    async def broadcast_tunnel_status(self, ticker: str, status: str, details: Optional[Dict[str, Any]] = None):
        await self._broadcast_ticker_message(ticker, {
            "type": "tunnel_status",
            "data": {
                "ticker": ticker,
                "status": status,
                "details": details or {},
                "timestamp": _now_iso(),
            },
        })

    async def send_personal(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)


class DualTunnelCoordinator:
    """
    Two continuously running tunnels:

      1) Live UI tunnel: streams latest market quotes to dashboard subscribers.
      2) Train+Predict tunnel: stores current ticks in parallel, triggers auto-training,
         and emits real-time predictions.
    """

    def __init__(self, manager: ConnectionManager):
        self.manager = manager

        # Dependency singletons (injected from main)
        self.live_service = None
        self.predictor = None
        self.trainer = None
        self.fetcher = None

        # Runtime state
        self._running = False
        self._live_task: Optional[asyncio.Task] = None
        self._predict_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None
        self._redis_leader_task: Optional[asyncio.Task] = None
        self._redis_subscriber_task: Optional[asyncio.Task] = None
        self._redis_subs_sync_task: Optional[asyncio.Task] = None
        self._training_worker_tasks: List[asyncio.Task] = []
        self._active_training_tickers: Set[str] = set()
        self._queued_training_jobs: Dict[str, Dict[str, Any]] = {}
        self._training_job_seq = 0

        # Shared data store for the second tunnel
        self._tick_store: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=2000))
        self._latest_predictions: Dict[str, Dict[str, Any]] = {}
        self._last_retrain_trigger: Dict[str, float] = {}
        self._last_retrain_tick_count: Dict[str, int] = {}
        self._live_cycle_cursor = 0
        self._predict_cycle_cursor = 0

        # IO / model safety locks
        self._log_lock = threading.Lock()
        self._model_lock = threading.Lock()

        # Tuning knobs
        self.live_interval_seconds = max(1, _safe_int(os.getenv("WS_LIVE_INTERVAL_SECONDS"), 2))
        self.predict_interval_seconds = max(1, _safe_int(os.getenv("WS_PREDICT_INTERVAL_SECONDS"), 6))
        self.min_ticks_for_retrain = max(5, _safe_int(os.getenv("WS_MIN_TICKS_FOR_RETRAIN"), 20))
        self.retrain_cooldown_seconds = max(60, _safe_int(os.getenv("WS_RETRAIN_COOLDOWN_SECONDS"), 300))
        self.retrain_window_ticks = max(8, _safe_int(os.getenv("WS_RETRAIN_WINDOW_TICKS"), 40))
        self.min_retrain_abs_return = max(0.0, _safe_float(os.getenv("WS_MIN_RETRAIN_ABS_RETURN"), 0.0015))
        self.min_retrain_spike_return = max(0.0, _safe_float(os.getenv("WS_MIN_RETRAIN_SPIKE_RETURN"), 0.004))

        self.auto_train_epochs = max(1, _safe_int(os.getenv("WS_AUTO_TRAIN_EPOCHS"), 4))
        self.auto_train_period = str(os.getenv("WS_AUTO_TRAIN_PERIOD") or "2y")
        self.auto_train_patience = max(2, _safe_int(os.getenv("WS_AUTO_TRAIN_PATIENCE"), 6))
        self.auto_train_batch_size = max(8, _safe_int(os.getenv("WS_AUTO_TRAIN_BATCH_SIZE"), 32))

        self.cold_start_epochs = max(self.auto_train_epochs, _safe_int(os.getenv("WS_COLD_START_EPOCHS"), 12))
        self.cold_start_period = str(os.getenv("WS_COLD_START_PERIOD") or "5y")
        self.cold_start_patience = max(self.auto_train_patience, _safe_int(os.getenv("WS_COLD_START_PATIENCE"), 10))

        self.max_tickers_per_cycle = max(1, _safe_int(os.getenv("WS_MAX_TICKERS_PER_CYCLE"), 300))
        self.max_live_parallelism = max(1, _safe_int(os.getenv("WS_LIVE_MAX_PARALLELISM"), 24))
        self.max_predict_parallelism = max(1, _safe_int(os.getenv("WS_PREDICT_MAX_PARALLELISM"), 8))
        self.training_worker_count = max(1, _safe_int(os.getenv("WS_TRAINING_WORKER_COUNT"), 1))
        self.training_queue_maxsize = max(100, _safe_int(os.getenv("WS_TRAINING_QUEUE_MAXSIZE"), 4000))
        self.training_worker_poll_seconds = max(1, _safe_int(os.getenv("WS_TRAINING_WORKER_POLL_SECONDS"), 1))
        self.training_use_isolated_model = _safe_bool(os.getenv("WS_TRAIN_WORKER_ISOLATED_MODEL"), True)
        self.training_queue_drop_oldest = _safe_bool(os.getenv("WS_TRAINING_QUEUE_DROP_OLDEST"), True)

        self.persist_streams = _safe_bool(os.getenv("WS_PERSIST_STREAMS"), False)
        self.persist_queue_maxsize = max(1000, _safe_int(os.getenv("WS_PERSIST_QUEUE_MAXSIZE"), 20000))
        self.persist_batch_size = max(10, _safe_int(os.getenv("WS_PERSIST_BATCH_SIZE"), 200))
        self.persist_flush_ms = max(20, _safe_int(os.getenv("WS_PERSIST_FLUSH_MS"), 200))

        # Redis cross-instance fanout (optional).
        self.redis_enabled = _safe_bool(os.getenv("WS_REDIS_ENABLED"), False)
        self.redis_url = str(os.getenv("WS_REDIS_URL") or "redis://localhost:6379/0").strip()
        self.redis_channel_prefix = str(os.getenv("WS_REDIS_CHANNEL_PREFIX") or "finsent:ws:v1").strip() or "finsent:ws:v1"
        self.redis_producer_enabled = _safe_bool(os.getenv("WS_REDIS_PRODUCER_ENABLED"), True)
        self.redis_subscribe_enabled = _safe_bool(os.getenv("WS_REDIS_SUBSCRIBE_ENABLED"), True)
        self.redis_leader_election_enabled = _safe_bool(
            os.getenv("WS_REDIS_LEADER_ELECTION_ENABLED"),
            self.redis_enabled,
        )
        self.redis_leader_heartbeat_seconds = max(
            1,
            _safe_int(os.getenv("WS_REDIS_LEADER_HEARTBEAT_SECONDS"), 2),
        )
        default_lock_ttl = max(6, self.redis_leader_heartbeat_seconds * 3)
        self.redis_leader_lock_ttl_seconds = max(
            self.redis_leader_heartbeat_seconds + 1,
            _safe_int(os.getenv("WS_REDIS_LEADER_LOCK_TTL_SECONDS"), default_lock_ttl),
        )
        self.redis_leader_jitter_ms = max(
            0,
            _safe_int(os.getenv("WS_REDIS_LEADER_JITTER_MS"), 300),
        )
        self.redis_use_global_subscriptions = _safe_bool(os.getenv("WS_REDIS_USE_GLOBAL_SUBSCRIPTIONS"), True)
        self.redis_reconnect_seconds = max(1, _safe_int(os.getenv("WS_REDIS_RECONNECT_SECONDS"), 3))
        self.redis_subs_sync_seconds = max(1, _safe_int(os.getenv("WS_REDIS_SUBS_SYNC_SECONDS"), 3))
        self.redis_subs_ttl_seconds = max(
            self.redis_subs_sync_seconds + 2,
            _safe_int(os.getenv("WS_REDIS_SUBS_TTL_SECONDS"), 15),
        )
        self.redis_subs_cache_ttl_seconds = max(1, _safe_int(os.getenv("WS_REDIS_SUBS_CACHE_TTL_SECONDS"), 2))
        self.keep_running_when_idle = _safe_bool(os.getenv("WS_KEEP_RUNNING_WHEN_IDLE"), self.redis_enabled)
        custom_instance_id = str(os.getenv("WS_INSTANCE_ID") or "").strip()
        if custom_instance_id:
            self._instance_id = custom_instance_id
        else:
            host_part = str(os.getenv("HOSTNAME") or "ws")
            self._instance_id = f"{host_part}-{os.getpid()}-{uuid.uuid4().hex[:8]}"

        self._redis_live_channel = f"{self.redis_channel_prefix}:live"
        self._redis_predict_channel = f"{self.redis_channel_prefix}:predict"
        self._redis_status_channel = f"{self.redis_channel_prefix}:status"
        self._redis_leader_key = f"{self.redis_channel_prefix}:producer:leader"
        self._redis_subs_key_prefix = f"{self.redis_channel_prefix}:subs"
        self._redis_subs_key = f"{self._redis_subs_key_prefix}:{self._instance_id}"
        self._redis_pub_client = None
        self._redis_sub_client = None
        self._redis_pubsub = None
        self._redis_ready = False
        self._redis_publish_count = 0
        self._redis_receive_count = 0
        self._redis_last_error: Optional[str] = None
        self._redis_last_message_at: Optional[str] = None
        self._global_ticker_cache: List[Dict[str, Any]] = []
        self._global_ticker_cache_at: float = 0.0
        self._redis_is_leader = False
        self._redis_current_leader_id: Optional[str] = None
        self._redis_leader_last_changed_at: Optional[str] = None
        self._redis_leader_acquired_count = 0
        self._redis_leader_lost_count = 0
        self._redis_leader_heartbeat_failures = 0
        self._redis_connect_lock = asyncio.Lock()

        if not self.redis_enabled:
            self._redis_is_leader = True
            self._redis_current_leader_id = self._instance_id
            self._redis_leader_last_changed_at = _now_iso()
        elif not self.redis_leader_election_enabled and self.redis_producer_enabled:
            self._redis_is_leader = True
            self._redis_current_leader_id = self._instance_id
            self._redis_leader_last_changed_at = _now_iso()

        # Logs for parallel storage pipeline
        self._quotes_log_path = os.path.join(TUNNEL_DATA_DIR, "live_quotes_stream.jsonl")
        self._predictions_log_path = os.path.join(TUNNEL_DATA_DIR, "live_predictions_stream.jsonl")
        self._persist_queue: asyncio.Queue = asyncio.Queue(maxsize=self.persist_queue_maxsize)
        self._dropped_persist_records = 0
        self._training_queue: asyncio.Queue = asyncio.Queue(maxsize=self.training_queue_maxsize)
        self._training_worker_trainer = None
        self._training_worker_isolated_model = False
        self._training_enqueued_count = 0
        self._training_dequeued_count = 0
        self._training_completed_count = 0
        self._training_failed_count = 0
        self._training_dropped_count = 0

        self._last_live_cycle_at: Optional[str] = None
        self._last_predict_cycle_at: Optional[str] = None
        self._live_cycles = 0
        self._predict_cycles = 0

    async def _close_async_obj(self, obj):
        if obj is None:
            return
        try:
            aclose = getattr(obj, "aclose", None)
            if callable(aclose):
                maybe = aclose()
                if asyncio.iscoroutine(maybe):
                    await maybe
                return
            close = getattr(obj, "close", None)
            if callable(close):
                maybe = close()
                if asyncio.iscoroutine(maybe):
                    await maybe
        except Exception:
            pass

    async def _close_redis_transport(self):
        try:
            if (
                self.redis_enabled
                and self.redis_leader_election_enabled
                and self.redis_producer_enabled
                and self._redis_pub_client is not None
            ):
                await self._release_leader_lock()
        except Exception:
            pass

        try:
            if (
                self.redis_enabled
                and self.redis_use_global_subscriptions
                and self._redis_pub_client is not None
            ):
                await self._redis_pub_client.delete(self._redis_subs_key)
        except Exception:
            pass

        await self._close_async_obj(self._redis_pubsub)
        await self._close_async_obj(self._redis_sub_client)
        await self._close_async_obj(self._redis_pub_client)
        self._redis_pubsub = None
        self._redis_sub_client = None
        self._redis_pub_client = None
        self._redis_ready = False
        self._global_ticker_cache = []
        self._global_ticker_cache_at = 0.0
        if self.redis_enabled and self.redis_leader_election_enabled:
            self._update_leader_state(False, None, reason="transport_closed")

    async def _ensure_redis_transport(self) -> bool:
        async with self._redis_connect_lock:
            if not self.redis_enabled:
                return False
            if self._redis_ready and self._redis_pub_client is not None:
                return True

            try:
                from redis import asyncio as redis_asyncio
            except Exception as e:
                self._redis_last_error = f"redis dependency unavailable: {e}"
                logger.warning(self._redis_last_error)
                self.redis_enabled = False
                return False

            try:
                self._redis_pub_client = redis_asyncio.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    health_check_interval=30,
                )
                await self._redis_pub_client.ping()

                if self.redis_subscribe_enabled:
                    self._redis_sub_client = redis_asyncio.from_url(
                        self.redis_url,
                        encoding="utf-8",
                        decode_responses=True,
                        health_check_interval=30,
                    )

                self._redis_ready = True
                self._redis_last_error = None
                logger.info(
                    "Redis fanout transport ready | prefix=%s | producer=%s | subscribe=%s",
                    self.redis_channel_prefix,
                    self.redis_producer_enabled,
                    self.redis_subscribe_enabled,
                )
                return True
            except Exception as e:
                self._redis_last_error = str(e)
                logger.warning(f"Redis transport not ready: {e}")
                await self._close_redis_transport()
                return False

    def _is_producer_active(self) -> bool:
        if not self.redis_enabled:
            return True
        if not self.redis_producer_enabled:
            return False
        if not self.redis_leader_election_enabled:
            return True
        return self._redis_is_leader

    def _update_leader_state(self, is_leader: bool, leader_id: Optional[str], reason: str = ""):
        clean_leader_id = str(leader_id or "").strip() or None
        next_is_leader = bool(is_leader)
        prev_is_leader = self._redis_is_leader
        prev_leader_id = self._redis_current_leader_id
        changed = prev_is_leader != next_is_leader or prev_leader_id != clean_leader_id

        self._redis_is_leader = next_is_leader
        self._redis_current_leader_id = clean_leader_id

        if not changed:
            return

        self._redis_leader_last_changed_at = _now_iso()
        if next_is_leader and not prev_is_leader:
            self._redis_leader_acquired_count += 1
            logger.info(
                "Redis producer leadership acquired | instance=%s | reason=%s",
                self._instance_id,
                reason or "unknown",
            )
            return

        if prev_is_leader and not next_is_leader:
            self._redis_leader_lost_count += 1
            logger.warning(
                "Redis producer leadership lost | instance=%s | next_leader=%s | reason=%s",
                self._instance_id,
                clean_leader_id,
                reason or "unknown",
            )
            return

        logger.info(
            "Redis producer leader changed | leader=%s | reason=%s",
            clean_leader_id,
            reason or "unknown",
        )

    async def _try_acquire_leader_lock(self) -> bool:
        if self._redis_pub_client is None:
            return False

        acquired = await self._redis_pub_client.set(
            self._redis_leader_key,
            self._instance_id,
            nx=True,
            ex=self.redis_leader_lock_ttl_seconds,
        )
        return bool(acquired)

    async def _try_renew_leader_lock(self) -> bool:
        if self._redis_pub_client is None:
            return False

        result = await self._redis_pub_client.eval(
            (
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('expire', KEYS[1], tonumber(ARGV[2])) "
                "else return 0 end"
            ),
            1,
            self._redis_leader_key,
            self._instance_id,
            str(self.redis_leader_lock_ttl_seconds),
        )
        return int(result or 0) == 1

    async def _release_leader_lock(self):
        if not self.redis_enabled or not self.redis_leader_election_enabled:
            return
        if self._redis_pub_client is None:
            self._update_leader_state(False, None, reason="lock_release_no_client")
            return

        try:
            await self._redis_pub_client.eval(
                (
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('del', KEYS[1]) "
                    "else return 0 end"
                ),
                1,
                self._redis_leader_key,
                self._instance_id,
            )
        except Exception as e:
            self._redis_last_error = str(e)
        finally:
            self._update_leader_state(False, None, reason="lock_released")

    async def _run_redis_leader_election(self):
        while (
            self._running
            and self.redis_enabled
            and self.redis_producer_enabled
            and self.redis_leader_election_enabled
        ):
            try:
                if not self._redis_ready:
                    ok = await self._ensure_redis_transport()
                    if not ok:
                        self._update_leader_state(False, None, reason="redis_not_ready")
                        await asyncio.sleep(self.redis_reconnect_seconds)
                        continue

                if self._redis_pub_client is None:
                    self._update_leader_state(False, None, reason="pub_client_missing")
                    await asyncio.sleep(self.redis_reconnect_seconds)
                    continue

                if self._redis_is_leader:
                    renewed = await self._try_renew_leader_lock()
                    if renewed:
                        self._redis_leader_heartbeat_failures = 0
                        self._update_leader_state(True, self._instance_id, reason="lock_renewed")
                    else:
                        self._redis_leader_heartbeat_failures += 1
                        current_leader = await self._redis_pub_client.get(self._redis_leader_key)
                        self._update_leader_state(
                            False,
                            str(current_leader or "").strip() or None,
                            reason="lock_renew_failed",
                        )
                else:
                    acquired = await self._try_acquire_leader_lock()
                    if acquired:
                        self._redis_leader_heartbeat_failures = 0
                        self._update_leader_state(True, self._instance_id, reason="lock_acquired")
                    else:
                        current_leader = await self._redis_pub_client.get(self._redis_leader_key)
                        self._update_leader_state(
                            False,
                            str(current_leader or "").strip() or None,
                            reason="following",
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._redis_last_error = str(e)
                self._redis_leader_heartbeat_failures += 1
                if self._redis_is_leader:
                    self._update_leader_state(False, None, reason="election_error")
                logger.warning(f"Redis leader election error: {e}")
                await asyncio.sleep(self.redis_reconnect_seconds)
                continue

            jitter_seconds = 0.0
            if self.redis_leader_jitter_ms > 0:
                jitter_seconds = random.uniform(0.0, float(self.redis_leader_jitter_ms) / 1000.0)
            await asyncio.sleep(float(self.redis_leader_heartbeat_seconds) + jitter_seconds)

        if self.redis_enabled and self.redis_leader_election_enabled:
            self._update_leader_state(False, self._redis_current_leader_id, reason="election_task_stopped")

    async def _publish_redis_event(self, event_type: str, ticker: str, payload: Dict[str, Any]) -> bool:
        if not self.redis_enabled:
            return False
        if not self._is_producer_active():
            return False
        if not self._redis_ready:
            ok = await self._ensure_redis_transport()
            if not ok:
                return False

        channel_map = {
            "live_tunnel": self._redis_live_channel,
            "predict_tunnel": self._redis_predict_channel,
            "tunnel_status": self._redis_status_channel,
        }
        channel = channel_map.get(event_type)
        if not channel or self._redis_pub_client is None:
            return False

        envelope = {
            "instance_id": self._instance_id,
            "event_type": event_type,
            "ticker": str(ticker or "").upper().strip(),
            "published_at": _now_iso(),
            "payload": payload,
        }

        try:
            await self._redis_pub_client.publish(
                channel,
                json.dumps(envelope, separators=(",", ":"), default=str),
            )
            self._redis_publish_count += 1
            return True
        except Exception as e:
            self._redis_last_error = str(e)
            self._redis_ready = False
            await self._close_redis_transport()
            return False

    async def _emit_live_tunnel(self, ticker: str, payload: Dict[str, Any]):
        await self.manager.broadcast_live_tunnel(ticker, payload)
        if self.redis_enabled:
            await self._publish_redis_event("live_tunnel", ticker, payload)

    async def _emit_prediction_tunnel(self, ticker: str, payload: Dict[str, Any]):
        await self.manager.broadcast_prediction_tunnel(ticker, payload)
        if self.redis_enabled:
            await self._publish_redis_event("predict_tunnel", ticker, payload)

    async def _emit_tunnel_status(self, ticker: str, status: str, details: Optional[Dict[str, Any]] = None):
        safe_details = details or {}
        await self.manager.broadcast_tunnel_status(ticker, status=status, details=safe_details)
        if self.redis_enabled:
            await self._publish_redis_event(
                "tunnel_status",
                ticker,
                {
                    "ticker": str(ticker or "").upper().strip(),
                    "status": status,
                    "details": safe_details,
                    "timestamp": _now_iso(),
                },
            )

    async def _handle_redis_envelope(self, envelope: Dict[str, Any]):
        if not isinstance(envelope, dict):
            return
        if envelope.get("instance_id") == self._instance_id:
            return

        event_type = str(envelope.get("event_type") or "").strip()
        payload = envelope.get("payload") or {}
        ticker = str(envelope.get("ticker") or payload.get("ticker") or "").upper().strip()
        if not ticker:
            return

        if event_type == "live_tunnel":
            self._tick_store[ticker].append(payload)
            await self.manager.broadcast_live_tunnel(ticker, payload)
        elif event_type == "predict_tunnel":
            self._latest_predictions[ticker] = payload
            await self.manager.broadcast_prediction_tunnel(ticker, payload)
        elif event_type == "tunnel_status":
            await self.manager.broadcast_tunnel_status(
                ticker,
                status=str(payload.get("status") or "unknown"),
                details=payload.get("details") or {},
            )

        self._redis_receive_count += 1
        self._redis_last_message_at = _now_iso()

    async def _run_redis_subscriber(self):
        while self._running and self.redis_enabled and self.redis_subscribe_enabled:
            try:
                if not self._redis_ready or self._redis_sub_client is None:
                    ok = await self._ensure_redis_transport()
                    if not ok:
                        await asyncio.sleep(self.redis_reconnect_seconds)
                        continue

                if self._redis_sub_client is None:
                    await asyncio.sleep(self.redis_reconnect_seconds)
                    continue

                if self._redis_pubsub is None:
                    self._redis_pubsub = self._redis_sub_client.pubsub(ignore_subscribe_messages=True)
                    await self._redis_pubsub.subscribe(
                        self._redis_live_channel,
                        self._redis_predict_channel,
                        self._redis_status_channel,
                    )
                    logger.info(
                        "Redis subscriber attached | channels=%s",
                        [self._redis_live_channel, self._redis_predict_channel, self._redis_status_channel],
                    )

                msg = await self._redis_pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg:
                    await asyncio.sleep(0.05)
                    continue

                data = msg.get("data")
                if not data:
                    continue
                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="ignore")
                envelope = json.loads(data)
                await self._handle_redis_envelope(envelope)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._redis_last_error = str(e)
                logger.warning(f"Redis subscriber error: {e}")
                await self._close_async_obj(self._redis_pubsub)
                self._redis_pubsub = None
                await asyncio.sleep(self.redis_reconnect_seconds)

    def _merge_ticker_configs(self, ticker_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for cfg in ticker_configs:
            ticker = str(cfg.get("ticker") or "").upper().strip()
            if not ticker:
                continue

            if ticker not in merged:
                merged[ticker] = {
                    "ticker": ticker,
                    "market": str(cfg.get("market") or "SP500").upper().strip(),
                    "capital": max(1000.0, _safe_float(cfg.get("capital"), 100000.0)),
                    "risk_tolerance": min(1.0, max(0.1, _safe_float(cfg.get("risk_tolerance"), 0.5))),
                    "enable_train_tunnel": bool(cfg.get("enable_train_tunnel", False)),
                    "subscriber_count": max(1, _safe_int(cfg.get("subscriber_count"), 1)),
                }
            else:
                merged[ticker]["subscriber_count"] += max(1, _safe_int(cfg.get("subscriber_count"), 1))
                merged[ticker]["enable_train_tunnel"] = (
                    merged[ticker]["enable_train_tunnel"]
                    or bool(cfg.get("enable_train_tunnel", False))
                )
        return list(merged.values())

    async def _sync_local_subscriptions(self, local_configs: List[Dict[str, Any]]):
        if not self.redis_enabled or not self.redis_use_global_subscriptions:
            return

        if not self._redis_ready:
            ok = await self._ensure_redis_transport()
            if not ok:
                return
        if self._redis_pub_client is None:
            return

        snapshot = {
            "instance_id": self._instance_id,
            "updated_at": _now_iso(),
            "tickers": local_configs,
        }
        try:
            await self._redis_pub_client.set(
                self._redis_subs_key,
                json.dumps(snapshot, separators=(",", ":"), default=str),
                ex=self.redis_subs_ttl_seconds,
            )
        except Exception as e:
            self._redis_last_error = str(e)

    async def _get_global_ticker_configs(self, local_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.redis_enabled or not self.redis_use_global_subscriptions:
            return local_configs

        now = time.time()
        if now - self._global_ticker_cache_at < self.redis_subs_cache_ttl_seconds:
            return list(self._global_ticker_cache)

        merged_input: List[Dict[str, Any]] = list(local_configs)

        if not self._redis_ready:
            ok = await self._ensure_redis_transport()
            if not ok:
                merged = self._merge_ticker_configs(merged_input)
                self._global_ticker_cache = merged
                self._global_ticker_cache_at = now
                return merged

        if self._redis_pub_client is None:
            merged = self._merge_ticker_configs(merged_input)
            self._global_ticker_cache = merged
            self._global_ticker_cache_at = now
            return merged

        try:
            keys: List[str] = []
            async for key in self._redis_pub_client.scan_iter(match=f"{self._redis_subs_key_prefix}:*", count=200):
                keys.append(str(key))

            if keys:
                values = await self._redis_pub_client.mget(keys)
                for raw in values:
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    tickers = payload.get("tickers") if isinstance(payload, dict) else None
                    if isinstance(tickers, list):
                        merged_input.extend(tickers)
        except Exception as e:
            self._redis_last_error = str(e)

        merged = self._merge_ticker_configs(merged_input)
        self._global_ticker_cache = merged
        self._global_ticker_cache_at = now
        return merged

    async def _run_redis_subscriptions_sync(self):
        while self._running and self.redis_enabled and self.redis_use_global_subscriptions:
            try:
                local_configs = self.manager.get_ticker_configs()
                await self._sync_local_subscriptions(local_configs)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._redis_last_error = str(e)
                logger.debug(f"Redis subscription sync error: {e}")
            await asyncio.sleep(self.redis_subs_sync_seconds)

    def _ensure_training_worker_trainer(self):
        if self._training_worker_trainer is not None:
            return
        if self.trainer is None:
            return

        if not self.training_use_isolated_model:
            self._training_worker_trainer = self.trainer
            self._training_worker_isolated_model = False
            return

        try:
            import copy

            trainer_cls = self.trainer.__class__
            model_copy = copy.deepcopy(self.trainer.model)
            kwargs = {
                "model": model_copy,
                "fetcher": self.trainer.fetcher,
                "indicators": self.trainer.indicators,
                "device": str(getattr(self.trainer, "device", "cpu")),
                "checkpoint_dir": self.trainer.checkpoint_dir,
            }
            self._training_worker_trainer = trainer_cls(**kwargs)
            self._training_worker_isolated_model = True
            logger.info("Training worker using isolated trainer model")
        except Exception as e:
            logger.warning(f"Failed to initialize isolated training worker model: {e}")
            self._training_worker_trainer = self.trainer
            self._training_worker_isolated_model = False

    def _reason_priority(self, reason: str) -> int:
        if reason == "cold_start":
            return 2
        if reason == "continuous_retrain":
            return 1
        return 0

    async def _queue_training_job(self, job: Dict[str, Any]) -> bool:
        ticker = str(job.get("ticker") or "").upper().strip()
        if not ticker:
            return False

        # Update existing queued job for this ticker when priority increases.
        existing = self._queued_training_jobs.get(ticker)
        if existing is not None:
            if self._reason_priority(job.get("reason", "")) > self._reason_priority(existing.get("reason", "")):
                existing.update(job)
            return False

        self._queued_training_jobs[ticker] = job

        while True:
            try:
                self._training_queue.put_nowait(job)
                self._training_enqueued_count += 1
                return True
            except asyncio.QueueFull:
                if not self.training_queue_drop_oldest:
                    self._training_dropped_count += 1
                    current = self._queued_training_jobs.get(ticker)
                    if current and current.get("job_id") == job.get("job_id"):
                        self._queued_training_jobs.pop(ticker, None)
                    return False

                try:
                    dropped = self._training_queue.get_nowait()
                except asyncio.QueueEmpty:
                    self._training_dropped_count += 1
                    current = self._queued_training_jobs.get(ticker)
                    if current and current.get("job_id") == job.get("job_id"):
                        self._queued_training_jobs.pop(ticker, None)
                    return False

                dropped_ticker = str(dropped.get("ticker") or "").upper().strip()
                dropped_job_id = dropped.get("job_id")
                current = self._queued_training_jobs.get(dropped_ticker)
                if current and current.get("job_id") == dropped_job_id:
                    self._queued_training_jobs.pop(dropped_ticker, None)
                self._training_dropped_count += 1

    async def _run_training_worker(self, worker_id: int):
        while self._running:
            job = None
            try:
                if self.redis_enabled and not self._is_producer_active():
                    await asyncio.sleep(max(1, self.training_worker_poll_seconds))
                    continue

                try:
                    job = await asyncio.wait_for(
                        self._training_queue.get(),
                        timeout=float(self.training_worker_poll_seconds),
                    )
                except asyncio.TimeoutError:
                    continue

                if not isinstance(job, dict):
                    continue

                ticker = str(job.get("ticker") or "").upper().strip()
                market = str(job.get("market") or "SP500").upper().strip()
                job_id = job.get("job_id")

                if not ticker:
                    continue

                current = self._queued_training_jobs.get(ticker)
                if current and current.get("job_id") != job_id:
                    continue
                if current and current.get("job_id") == job_id:
                    self._queued_training_jobs.pop(ticker, None)

                if ticker in self._active_training_tickers:
                    continue

                trainer_obj = self._training_worker_trainer or self.trainer
                if trainer_obj is None:
                    await self._emit_tunnel_status(
                        ticker,
                        status="training_failed",
                        details={"reason": "trainer_not_configured"},
                    )
                    continue

                self._active_training_tickers.add(ticker)
                self._training_dequeued_count += 1
                started_at = time.time()

                await self._emit_tunnel_status(
                    ticker,
                    status="training_started",
                    details={
                        "reason": job.get("reason"),
                        "market": market,
                        "epochs": job.get("epochs"),
                        "period": job.get("period"),
                        "patience": job.get("patience"),
                        "worker_id": worker_id,
                    },
                )

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._train_blocking(
                        trainer_obj=trainer_obj,
                        ticker=ticker,
                        market=market,
                        epochs=int(job.get("epochs") or self.auto_train_epochs),
                        patience=int(job.get("patience") or self.auto_train_patience),
                        period=str(job.get("period") or self.auto_train_period),
                    ),
                )

                if result.get("status") == "success":
                    self._training_completed_count += 1
                    self._last_retrain_tick_count[ticker] = len(self._tick_store.get(ticker, []))
                    self._last_retrain_trigger[ticker] = time.time()
                    if self.predictor and hasattr(self.predictor, "invalidate_cache"):
                        try:
                            self.predictor.invalidate_cache(ticker)
                        except Exception:
                            pass

                    await self._emit_tunnel_status(
                        ticker,
                        status="training_completed",
                        details={
                            "worker_id": worker_id,
                            "duration_seconds": round(time.time() - started_at, 3),
                            "best_val_loss": result.get("best_val_loss"),
                            "best_val_accuracy": result.get("best_val_accuracy"),
                            "epochs_trained": result.get("epochs_trained"),
                        },
                    )
                else:
                    self._training_failed_count += 1
                    await self._emit_tunnel_status(
                        ticker,
                        status="training_failed",
                        details={
                            "worker_id": worker_id,
                            "duration_seconds": round(time.time() - started_at, 3),
                            "error": result.get("error") or result.get("reason") or "unknown",
                        },
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training worker {worker_id} loop error: {e}")
            finally:
                if isinstance(job, dict):
                    ticker = str(job.get("ticker") or "").upper().strip()
                    if ticker:
                        self._active_training_tickers.discard(ticker)
                    try:
                        self._training_queue.task_done()
                    except Exception:
                        pass

    def configure(self, live_service=None, predictor=None, trainer=None, fetcher=None):
        if live_service is not None:
            self.live_service = live_service
        if predictor is not None:
            self.predictor = predictor
        if trainer is not None:
            self.trainer = trainer
        if fetcher is not None:
            self.fetcher = fetcher

    async def ensure_started(self):
        if self._running:
            return
        self._running = True

        if self.redis_enabled and self.redis_leader_election_enabled and self.redis_producer_enabled:
            self._update_leader_state(False, self._redis_current_leader_id, reason="leader_election_wait")
            if self._redis_leader_task is None or self._redis_leader_task.done():
                self._redis_leader_task = asyncio.create_task(
                    self._run_redis_leader_election(),
                    name="finsent-redis-leader-election",
                )
        elif self.redis_enabled and not self.redis_leader_election_enabled:
            self._update_leader_state(
                self.redis_producer_enabled,
                self._instance_id if self.redis_producer_enabled else None,
                reason="static_producer_mode",
            )

        self._ensure_training_worker_trainer()
        if not self._training_worker_tasks:
            for idx in range(self.training_worker_count):
                self._training_worker_tasks.append(
                    asyncio.create_task(
                        self._run_training_worker(worker_id=idx + 1),
                        name=f"finsent-training-worker-{idx + 1}",
                    )
                )
        if self.redis_enabled and self.redis_subscribe_enabled:
            self._redis_subscriber_task = asyncio.create_task(
                self._run_redis_subscriber(),
                name="finsent-redis-subscriber",
            )
        if self.redis_enabled and self.redis_use_global_subscriptions:
            self._redis_subs_sync_task = asyncio.create_task(
                self._run_redis_subscriptions_sync(),
                name="finsent-redis-subs-sync",
            )
        self._live_task = asyncio.create_task(self._run_live_tunnel(), name="finsent-live-ui-tunnel")
        self._predict_task = asyncio.create_task(self._run_predict_tunnel(), name="finsent-train-predict-tunnel")
        if self.persist_streams:
            self._persist_task = asyncio.create_task(self._run_persist_writer(), name="finsent-persist-writer")
        logger.info("DualTunnelCoordinator started")

    async def stop(self):
        if not self._running:
            return

        self._running = False
        self._queued_training_jobs.clear()
        tasks = [
            self._live_task,
            self._predict_task,
            self._persist_task,
            self._redis_leader_task,
            self._redis_subscriber_task,
            self._redis_subs_sync_task,
            *self._training_worker_tasks,
        ]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        for task in tasks:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug(f"Tunnel task shutdown warning: {e}")

        self._live_task = None
        self._predict_task = None
        self._persist_task = None
        self._redis_leader_task = None
        self._redis_subscriber_task = None
        self._redis_subs_sync_task = None
        self._training_worker_tasks = []
        self._active_training_tickers.clear()
        while not self._training_queue.empty():
            try:
                self._training_queue.get_nowait()
            except Exception:
                break
        await self._close_redis_transport()
        logger.info("DualTunnelCoordinator stopped")

    def status(self) -> Dict[str, Any]:
        active_training = sorted(self._active_training_tickers)
        queued_training = sorted(self._queued_training_jobs.keys())
        tick_store_sizes = {
            t: len(buf) for t, buf in list(self._tick_store.items())[:50]
        }
        return {
            "running": self._running,
            "active_connections": self.manager.connection_count,
            "keep_running_when_idle": self.keep_running_when_idle,
            "subscribed_tickers": self.manager.get_ticker_configs(),
            "live_interval_seconds": self.live_interval_seconds,
            "predict_interval_seconds": self.predict_interval_seconds,
            "max_tickers_per_cycle": self.max_tickers_per_cycle,
            "max_live_parallelism": self.max_live_parallelism,
            "max_predict_parallelism": self.max_predict_parallelism,
            "training_workers": {
                "count": self.training_worker_count,
                "poll_seconds": self.training_worker_poll_seconds,
                "queue_size": self._training_queue.qsize(),
                "queue_maxsize": self.training_queue_maxsize,
                "drop_oldest": self.training_queue_drop_oldest,
                "use_isolated_model": self.training_use_isolated_model,
                "isolated_model_ready": self._training_worker_isolated_model,
                "enqueued": self._training_enqueued_count,
                "dequeued": self._training_dequeued_count,
                "completed": self._training_completed_count,
                "failed": self._training_failed_count,
                "dropped": self._training_dropped_count,
            },
            "retrain": {
                "min_ticks": self.min_ticks_for_retrain,
                "cooldown_seconds": self.retrain_cooldown_seconds,
                "window_ticks": self.retrain_window_ticks,
                "min_abs_return": self.min_retrain_abs_return,
                "min_spike_return": self.min_retrain_spike_return,
            },
            "auto_train": {
                "epochs": self.auto_train_epochs,
                "period": self.auto_train_period,
                "patience": self.auto_train_patience,
                "batch_size": self.auto_train_batch_size,
                "cold_start_epochs": self.cold_start_epochs,
                "cold_start_period": self.cold_start_period,
            },
            "persistence": {
                "enabled": self.persist_streams,
                "queue_size": self._persist_queue.qsize() if self.persist_streams else 0,
                "queue_maxsize": self.persist_queue_maxsize,
                "dropped_records": self._dropped_persist_records,
            },
            "redis_fanout": {
                "enabled": self.redis_enabled,
                "producer_enabled": self.redis_producer_enabled,
                "producer_active": self._is_producer_active(),
                "subscribe_enabled": self.redis_subscribe_enabled,
                "use_global_subscriptions": self.redis_use_global_subscriptions,
                "ready": self._redis_ready,
                "instance_id": self._instance_id,
                "channel_prefix": self.redis_channel_prefix,
                "leader_election": {
                    "enabled": self.redis_leader_election_enabled,
                    "eligible": self.redis_producer_enabled,
                    "is_leader": self._redis_is_leader,
                    "leader_instance_id": self._redis_current_leader_id,
                    "lock_key": self._redis_leader_key,
                    "heartbeat_seconds": self.redis_leader_heartbeat_seconds,
                    "lock_ttl_seconds": self.redis_leader_lock_ttl_seconds,
                    "jitter_ms": self.redis_leader_jitter_ms,
                    "last_changed_at": self._redis_leader_last_changed_at,
                    "acquired_count": self._redis_leader_acquired_count,
                    "lost_count": self._redis_leader_lost_count,
                    "heartbeat_failures": self._redis_leader_heartbeat_failures,
                },
                "subscriptions": {
                    "sync_seconds": self.redis_subs_sync_seconds,
                    "ttl_seconds": self.redis_subs_ttl_seconds,
                    "cache_ttl_seconds": self.redis_subs_cache_ttl_seconds,
                },
                "channels": {
                    "live": self._redis_live_channel,
                    "predict": self._redis_predict_channel,
                    "status": self._redis_status_channel,
                },
                "published_events": self._redis_publish_count,
                "received_events": self._redis_receive_count,
                "last_message_at": self._redis_last_message_at,
                "last_error": self._redis_last_error,
            },
            "live_cycles": self._live_cycles,
            "predict_cycles": self._predict_cycles,
            "last_live_cycle_at": self._last_live_cycle_at,
            "last_predict_cycle_at": self._last_predict_cycle_at,
            "active_training_jobs": active_training,
            "queued_training_jobs": queued_training,
            "tick_store_sizes": tick_store_sizes,
            "prediction_cache_tickers": sorted(self._latest_predictions.keys()),
            "logs": {
                "quotes_stream": self._quotes_log_path,
                "predictions_stream": self._predictions_log_path,
            },
        }

    def _append_jsonl_batch(self, path: str, payloads: List[Dict[str, Any]]):
        if not payloads:
            return
        lines = [json.dumps(p, separators=(",", ":"), default=str) + "\n" for p in payloads]
        with self._log_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.writelines(lines)

    def _enqueue_persist(self, path: str, payload: Dict[str, Any]):
        if not self.persist_streams:
            return
        try:
            self._persist_queue.put_nowait((path, payload))
        except asyncio.QueueFull:
            self._dropped_persist_records += 1

    async def _flush_persist_batch(self, buffered: List[Any]):
        if not buffered:
            return

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for path, payload in buffered:
            grouped[path].append(payload)

        loop = asyncio.get_running_loop()
        for path, payloads in grouped.items():
            await loop.run_in_executor(None, self._append_jsonl_batch, path, payloads)

    async def _run_persist_writer(self):
        flush_seconds = self.persist_flush_ms / 1000.0
        buffered: List[Any] = []

        try:
            while self._running or not self._persist_queue.empty():
                try:
                    item = await asyncio.wait_for(self._persist_queue.get(), timeout=flush_seconds)
                    buffered.append(item)
                except asyncio.TimeoutError:
                    pass

                if len(buffered) >= self.persist_batch_size or (buffered and self._persist_queue.empty()):
                    await self._flush_persist_batch(buffered)
                    buffered.clear()
        except asyncio.CancelledError:
            if buffered:
                await self._flush_persist_batch(buffered)
            raise

    async def _fetch_quote(self, ticker: str, market: str) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_running_loop()

        if self.live_service is not None:
            try:
                return await loop.run_in_executor(
                    None, lambda: self.live_service.get_realtime_quote(ticker=ticker, market=market)
                )
            except Exception as e:
                logger.debug(f"Live service quote failed for {ticker}: {e}")

        if self.fetcher is not None:
            try:
                return await loop.run_in_executor(
                    None, lambda: self.fetcher.get_live_price(ticker=ticker, market=market)
                )
            except Exception as e:
                logger.debug(f"Fetcher quote failed for {ticker}: {e}")

        return None

    def _train_blocking(
        self,
        trainer_obj,
        ticker: str,
        market: str,
        epochs: int,
        patience: int,
        period: str,
    ) -> Dict[str, Any]:
        if trainer_obj is None:
            return {"status": "skipped", "reason": "trainer_not_configured", "ticker": ticker}

        # In fallback mode, trainer may share the same model as inference; keep lock for safety.
        if trainer_obj is self.trainer:
            with self._model_lock:
                return trainer_obj.train(
                    ticker=ticker,
                    market=market,
                    epochs=epochs,
                    batch_size=self.auto_train_batch_size,
                    patience=patience,
                    period=period,
                )

        return trainer_obj.train(
            ticker=ticker,
            market=market,
            epochs=epochs,
            batch_size=self.auto_train_batch_size,
            patience=patience,
            period=period,
        )

    async def _schedule_training(self, ticker: str, market: str, reason: str):
        if self.trainer is None:
            return

        self._ensure_training_worker_trainer()

        ticker = str(ticker or "").upper().strip()
        market = str(market or "SP500").upper().strip()
        if not ticker:
            return

        if ticker in self._active_training_tickers:
            return

        existing = self._queued_training_jobs.get(ticker)
        if existing and self._reason_priority(existing.get("reason", "")) >= self._reason_priority(reason):
            return

        if reason == "cold_start":
            epochs = self.cold_start_epochs
            patience = self.cold_start_patience
            period = self.cold_start_period
        else:
            epochs = self.auto_train_epochs
            patience = self.auto_train_patience
            period = self.auto_train_period

        self._training_job_seq += 1
        job = {
            "job_id": self._training_job_seq,
            "ticker": ticker,
            "market": market,
            "reason": reason,
            "epochs": epochs,
            "patience": patience,
            "period": period,
            "enqueued_at": _now_iso(),
        }

        queued = await self._queue_training_job(job)
        if not queued:
            # If a retrain trigger could not be queued, allow future attempts quickly.
            if reason != "cold_start":
                self._last_retrain_trigger[ticker] = 0
            await self._emit_tunnel_status(
                ticker,
                status="training_queue_full",
                details={
                    "reason": reason,
                    "queue_size": self._training_queue.qsize(),
                    "queue_maxsize": self.training_queue_maxsize,
                },
            )
            return

        await self._emit_tunnel_status(
            ticker,
            status="training_queued",
            details={
                "reason": reason,
                "market": market,
                "epochs": epochs,
                "period": period,
                "patience": patience,
                "queue_size": self._training_queue.qsize(),
            },
        )

    def _should_retrain(self, ticker: str) -> bool:
        if self.trainer is None:
            return False

        if not self.trainer.is_model_trained(ticker):
            return False

        tick_count = len(self._tick_store.get(ticker, []))
        last_tick_baseline = self._last_retrain_tick_count.get(ticker, 0)
        if tick_count - last_tick_baseline < self.min_ticks_for_retrain:
            return False

        now = time.time()
        last_trigger = self._last_retrain_trigger.get(ticker, 0)
        if now - last_trigger < self.retrain_cooldown_seconds:
            return False

        recent_ticks = list(self._tick_store.get(ticker, []))[-self.retrain_window_ticks:]
        prices = [
            _safe_float(row.get("price"), 0.0)
            for row in recent_ticks
            if _safe_float(row.get("price"), 0.0) > 0
        ]

        if len(prices) >= 8:
            abs_returns = [
                abs((prices[i] - prices[i - 1]) / max(abs(prices[i - 1]), 1e-6))
                for i in range(1, len(prices))
            ]
            mean_abs_return = sum(abs_returns) / max(len(abs_returns), 1)
            spike_return = max(abs_returns) if abs_returns else 0.0

            if (
                mean_abs_return < self.min_retrain_abs_return
                and spike_return < self.min_retrain_spike_return
            ):
                return False

        self._last_retrain_trigger[ticker] = now
        return True

    def _select_cycle_configs(self, ticker_configs: List[Dict[str, Any]], loop_name: str) -> List[Dict[str, Any]]:
        if not ticker_configs:
            return []
        if len(ticker_configs) <= self.max_tickers_per_cycle:
            return ticker_configs

        if loop_name == "live":
            cursor = self._live_cycle_cursor % len(ticker_configs)
            rotated = ticker_configs[cursor:] + ticker_configs[:cursor]
            self._live_cycle_cursor = (cursor + self.max_tickers_per_cycle) % len(ticker_configs)
            return rotated[:self.max_tickers_per_cycle]

        cursor = self._predict_cycle_cursor % len(ticker_configs)
        rotated = ticker_configs[cursor:] + ticker_configs[:cursor]
        self._predict_cycle_cursor = (cursor + self.max_tickers_per_cycle) % len(ticker_configs)
        return rotated[:self.max_tickers_per_cycle]

    def _predict_blocking(self, ticker: str, market: str, capital: float, risk_tolerance: float) -> Dict[str, Any]:
        if self.predictor is None:
            return {"status": "skipped", "reason": "predictor_not_configured", "ticker": ticker}

        with self._model_lock:
            return self.predictor.predict(
                ticker=ticker,
                market=market,
                total_capital=capital,
                risk_tolerance=risk_tolerance,
            )

    async def _predict_async(self, ticker: str, market: str, capital: float, risk_tolerance: float) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._predict_blocking(ticker, market, capital, risk_tolerance),
        )

    async def _run_live_tunnel(self):
        while self._running:
            try:
                if self.redis_enabled and not self._is_producer_active():
                    await asyncio.sleep(max(1, self.live_interval_seconds))
                    continue

                local_configs = self.manager.get_ticker_configs()
                ticker_configs = await self._get_global_ticker_configs(local_configs)
                if not ticker_configs:
                    await asyncio.sleep(1)
                    continue

                cycle_configs = self._select_cycle_configs(ticker_configs, loop_name="live")
                semaphore = asyncio.Semaphore(self.max_live_parallelism)

                async def _process_live_cfg(cfg: Dict[str, Any]):
                    ticker = cfg["ticker"]
                    market = cfg.get("market", "SP500")
                    async with semaphore:
                        quote = await self._fetch_quote(ticker=ticker, market=market)
                    if not quote:
                        return

                    payload = {
                        "tunnel": "live_ui_tunnel",
                        "ticker": ticker,
                        "market": market,
                        "price": quote.get("price"),
                        "change": quote.get("change"),
                        "change_pct": quote.get("change_pct"),
                        "volume": quote.get("volume"),
                        "source": quote.get("source", "live"),
                        "timestamp": quote.get("timestamp", _now_iso()),
                    }

                    self._tick_store[ticker].append(payload)
                    await self._emit_live_tunnel(ticker, payload)
                    self._enqueue_persist(self._quotes_log_path, payload)

                results = await asyncio.gather(
                    *(_process_live_cfg(cfg) for cfg in cycle_configs),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, Exception):
                        logger.debug(f"Live tunnel worker error: {result}")

                self._live_cycles += 1
                self._last_live_cycle_at = _now_iso()
                await asyncio.sleep(self.live_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Live tunnel loop error: {e}")
                await asyncio.sleep(2)

    async def _run_predict_tunnel(self):
        while self._running:
            try:
                if self.redis_enabled and not self._is_producer_active():
                    await asyncio.sleep(max(1, self.predict_interval_seconds))
                    continue

                local_configs = self.manager.get_ticker_configs()
                ticker_configs = await self._get_global_ticker_configs(local_configs)
                if not ticker_configs:
                    await asyncio.sleep(2)
                    continue

                cycle_configs = self._select_cycle_configs(ticker_configs, loop_name="predict")
                predict_candidates: List[Dict[str, Any]] = []

                for cfg in cycle_configs:
                    ticker = cfg["ticker"]
                    market = cfg.get("market", "SP500")
                    capital = max(1000.0, _safe_float(cfg.get("capital"), 100000.0))
                    risk_tolerance = min(1.0, max(0.1, _safe_float(cfg.get("risk_tolerance"), 0.5)))
                    train_enabled = bool(cfg.get("enable_train_tunnel", False))

                    if not train_enabled:
                        continue

                    if self.trainer is not None and not self.trainer.is_model_trained(ticker):
                        await self._schedule_training(ticker=ticker, market=market, reason="cold_start")
                        continue

                    if self._should_retrain(ticker):
                        await self._schedule_training(ticker=ticker, market=market, reason="continuous_retrain")

                    if ticker in self._active_training_tickers:
                        await self._emit_tunnel_status(
                            ticker,
                            status="training_in_progress",
                            details={"market": market},
                        )
                        continue

                    if ticker in self._queued_training_jobs:
                        await self._emit_tunnel_status(
                            ticker,
                            status="training_queued",
                            details={
                                "market": market,
                                "queue_size": self._training_queue.qsize(),
                            },
                        )
                        continue

                    predict_candidates.append({
                        "ticker": ticker,
                        "market": market,
                        "capital": capital,
                        "risk_tolerance": risk_tolerance,
                    })

                predict_semaphore = asyncio.Semaphore(self.max_predict_parallelism)

                async def _predict_for_cfg(cfg: Dict[str, Any]):
                    ticker = cfg["ticker"]
                    market = cfg["market"]
                    capital = cfg["capital"]
                    risk_tolerance = cfg["risk_tolerance"]

                    async with predict_semaphore:
                        prediction = await self._predict_async(
                            ticker=ticker,
                            market=market,
                            capital=capital,
                            risk_tolerance=risk_tolerance,
                        )

                    if prediction.get("status") != "success":
                        await self._emit_tunnel_status(
                            ticker,
                            status="prediction_waiting",
                            details={"state": prediction.get("status", "unknown")},
                        )
                        return

                    payload = {
                        "tunnel": "model_train_predict_tunnel",
                        "ticker": ticker,
                        "market": market,
                        "timestamp": prediction.get("timestamp", _now_iso()),
                        "inference_time_ms": prediction.get("inference_time_ms"),
                        "live_price": prediction.get("live_price", {}),
                        "prediction": prediction.get("prediction", {}),
                        "signal": prediction.get("signal", {}),
                        "analysis": prediction.get("analysis", {}),
                    }

                    self._latest_predictions[ticker] = payload
                    await self._emit_prediction_tunnel(ticker, payload)
                    self._enqueue_persist(self._predictions_log_path, payload)

                predict_results = await asyncio.gather(
                    *(_predict_for_cfg(cfg) for cfg in predict_candidates),
                    return_exceptions=True,
                )
                for result in predict_results:
                    if isinstance(result, Exception):
                        logger.debug(f"Predict tunnel worker error: {result}")

                self._predict_cycles += 1
                self._last_predict_cycle_at = _now_iso()
                await asyncio.sleep(self.predict_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Predict tunnel loop error: {e}")
                await asyncio.sleep(3)


manager = ConnectionManager()
tunnel_coordinator = DualTunnelCoordinator(manager)


def get_tunnel_status() -> Dict[str, Any]:
    return tunnel_coordinator.status()


async def websocket_endpoint(
    websocket: WebSocket,
    fetcher=None,
    live_service=None,
    predictor=None,
    trainer=None,
):
    """
    WebSocket endpoint for two real-time tunnels.

    Client messages:
        {
          "action": "subscribe",
          "tickers": ["AAPL", "MSFT"],
          "market": "SP500",
          "capital": 30000,
          "risk_tolerance": 0.5,
          "enable_train_tunnel": false
        }
        {"action": "unsubscribe", "tickers": ["AAPL"]}
        {"action": "status"}

    Server messages:
        {"type": "live_tunnel",    "data": {ticker, price, change_pct, ...}}
        {"type": "predict_tunnel", "data": {ticker, prediction, signal, ...}}
        {"type": "tunnel_status",  "data": {ticker, status, details, ...}}
    """
    await manager.connect(websocket)

    tunnel_coordinator.configure(
        fetcher=fetcher,
        live_service=live_service,
        predictor=predictor,
        trainer=trainer,
    )
    await tunnel_coordinator.ensure_started()

    await manager.send_personal(websocket, {
        "type": "tunnel_ready",
        "timestamp": _now_iso(),
        "status": get_tunnel_status(),
    })

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action", "")

            if action == "subscribe":
                raw_tickers = msg.get("tickers", [])
                tickers = [raw_tickers] if isinstance(raw_tickers, str) else list(raw_tickers)
                market = str(msg.get("market", "SP500")).upper().strip()
                capital = _safe_float(msg.get("capital"), 100000.0)
                risk_tolerance = _safe_float(msg.get("risk_tolerance"), 0.5)
                enable_train_tunnel = bool(msg.get("enable_train_tunnel", False))

                subscribed = manager.subscribe(
                    websocket=websocket,
                    tickers=tickers,
                    market=market,
                    capital=capital,
                    risk_tolerance=risk_tolerance,
                    enable_train_tunnel=enable_train_tunnel,
                )

                await manager.send_personal(websocket, {
                    "type": "subscribed",
                    "tickers": subscribed,
                    "market": market,
                    "enable_train_tunnel": enable_train_tunnel,
                    "tunnels": {
                        "live_ui_tunnel": True,
                        "model_train_predict_tunnel": enable_train_tunnel,
                    },
                })

            elif action == "unsubscribe":
                raw_tickers = msg.get("tickers", [])
                tickers = [raw_tickers] if isinstance(raw_tickers, str) else list(raw_tickers)
                remaining = manager.unsubscribe(websocket, tickers)
                await manager.send_personal(websocket, {
                    "type": "unsubscribed",
                    "tickers": tickers,
                    "remaining": remaining,
                })

            elif action == "status":
                await manager.send_personal(websocket, {
                    "type": "status",
                    "timestamp": _now_iso(),
                    "status": get_tunnel_status(),
                })

            elif action == "ping":
                await manager.send_personal(websocket, {
                    "type": "pong",
                    "timestamp": _now_iso(),
                    "status": {
                        "connections": manager.connection_count,
                        "subscribed_tickers": manager.get_ticker_configs(),
                    },
                })

            else:
                await manager.send_personal(websocket, {
                    "type": "error",
                    "message": f"Unknown action: {action}",
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        if manager.connection_count == 0 and not tunnel_coordinator.keep_running_when_idle:
            await tunnel_coordinator.stop()
