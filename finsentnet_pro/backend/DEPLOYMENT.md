# FinSentNet Pro Backend Deployment Profiles

This document defines role intent per node class for distributed websocket deployments.

## Required Baseline

Set these on all distributed nodes:

- `WS_REDIS_ENABLED=true`
- `WS_REDIS_URL=redis://<host>:6379/0`
- `WS_REDIS_CHANNEL_PREFIX=finsent:ws:v1` (or your environment-specific prefix)
- `WS_REDIS_SUBSCRIBE_ENABLED=true`
- `WS_REDIS_USE_GLOBAL_SUBSCRIPTIONS=true`
- `WS_KEEP_RUNNING_WHEN_IDLE=true`

## Node Class Profiles

### 1) Leader-Candidate Node (Producer Eligible)

Use for compute-capable nodes that may run live fetch + predict + training work when elected leader.

- `WS_REDIS_PRODUCER_ENABLED=true`
- `WS_REDIS_LEADER_ELECTION_ENABLED=true`
- `WS_REDIS_LEADER_HEARTBEAT_SECONDS=2`
- `WS_REDIS_LEADER_LOCK_TTL_SECONDS=6`
- `WS_REDIS_LEADER_JITTER_MS=300`

Behavior:

- Exactly one instance becomes active producer at a time.
- If leader fails, another leader-candidate takes over after lock expiry and heartbeat cycle.

### 2) Follower-Only Node (No Producer Work)

Use for API edge nodes where role intent is subscription fanout/relay only.

- `WS_REDIS_PRODUCER_ENABLED=false`
- `WS_REDIS_LEADER_ELECTION_ENABLED=false`
- `WS_REDIS_SUBSCRIBE_ENABLED=true`
- `WS_REDIS_USE_GLOBAL_SUBSCRIPTIONS=true`

Behavior:

- Never runs producer loops (live fetch/predict/training scheduling).
- Receives distributed events from Redis and serves websocket clients.
- Safe to scale horizontally without accidental producer contention.

### 3) Single-Node / Local Mode

Use for local dev or non-distributed runs.

- `WS_REDIS_ENABLED=false`
- `WS_REDIS_PRODUCER_ENABLED=true`
- `WS_REDIS_LEADER_ELECTION_ENABLED=false`

Behavior:

- Producer always active on the local instance.
- No Redis dependency.

## Suggested Topology

- At least 2 leader-candidate nodes for high availability.
- Any number of follower-only edge nodes for websocket scale.
- Shared Redis for event fanout, leader lock, and subscription snapshots.

## Runtime Verification

Call `GET /api/live/tunnels/status` and inspect:

- `tunnels.redis_fanout.producer_enabled`
- `tunnels.redis_fanout.producer_active`
- `tunnels.redis_fanout.leader_election.is_leader`
- `tunnels.redis_fanout.leader_election.leader_instance_id`

Expected values:

- Leader-candidate winner: `producer_enabled=true`, `producer_active=true`, `is_leader=true`
- Leader-candidate follower: `producer_enabled=true`, `producer_active=false`, `is_leader=false`
- Follower-only node: `producer_enabled=false`, `producer_active=false`

## Example Environment Blocks

Leader-candidate:

```env
WS_REDIS_ENABLED=true
WS_REDIS_URL=redis://redis.internal:6379/0
WS_REDIS_CHANNEL_PREFIX=finsent:ws:prod
WS_REDIS_PRODUCER_ENABLED=true
WS_REDIS_LEADER_ELECTION_ENABLED=true
WS_REDIS_LEADER_HEARTBEAT_SECONDS=2
WS_REDIS_LEADER_LOCK_TTL_SECONDS=6
WS_REDIS_LEADER_JITTER_MS=300
WS_REDIS_SUBSCRIBE_ENABLED=true
WS_REDIS_USE_GLOBAL_SUBSCRIPTIONS=true
WS_KEEP_RUNNING_WHEN_IDLE=true
```

Follower-only:

```env
WS_REDIS_ENABLED=true
WS_REDIS_URL=redis://redis.internal:6379/0
WS_REDIS_CHANNEL_PREFIX=finsent:ws:prod
WS_REDIS_PRODUCER_ENABLED=false
WS_REDIS_LEADER_ELECTION_ENABLED=false
WS_REDIS_SUBSCRIBE_ENABLED=true
WS_REDIS_USE_GLOBAL_SUBSCRIPTIONS=true
WS_KEEP_RUNNING_WHEN_IDLE=true
```
