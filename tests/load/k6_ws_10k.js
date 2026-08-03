import ws from "k6/ws";
import { check } from "k6";
import exec from "k6/execution";
import { Counter, Rate, Trend } from "k6/metrics";

function envInt(name, fallback) {
  const raw = __ENV[name];
  if (raw === undefined) {
    return fallback;
  }

  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function envFloat(name, fallback) {
  const raw = __ENV[name];
  if (raw === undefined) {
    return fallback;
  }

  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function envBool(name, fallback) {
  const raw = __ENV[name];
  if (raw === undefined) {
    return fallback;
  }

  return ["1", "true", "yes", "on"].includes(String(raw).trim().toLowerCase());
}

function envCsv(name, fallbackCsv) {
  const raw = String(__ENV[name] || fallbackCsv);
  return raw
    .split(",")
    .map((value) => value.trim().toUpperCase())
    .filter((value) => value.length > 0);
}

function normalizeProfile(raw) {
  const value = String(raw || "transport_only").trim().toLowerCase();
  if (["mixed", "mixed_train_infer", "train", "training"].includes(value)) {
    return "mixed_train_infer";
  }
  return "transport_only";
}

const LOAD_PROFILE = normalizeProfile(__ENV.LOAD_PROFILE);
const PROFILE_IS_MIXED = LOAD_PROFILE === "mixed_train_infer";

const WS_URL = __ENV.WS_URL || "ws://127.0.0.1:8000/ws";
const MARKET = String(__ENV.MARKET || "SP500").toUpperCase();
const CAPITAL = Math.max(1000, envFloat("CAPITAL", 30000));
const RISK_TOLERANCE = Math.min(1, Math.max(0.1, envFloat("RISK_TOLERANCE", 0.5)));
const ENABLE_TRAIN_TUNNEL = envBool("ENABLE_TRAIN_TUNNEL", PROFILE_IS_MIXED);

const MAX_SOCKETS = Math.max(1, envInt("MAX_SOCKETS", 10000));
const STAGE_STEP_SECONDS = Math.max(10, envInt("STAGE_STEP_SECONDS", 120));
const HOLD_STAGE_SECONDS = Math.max(30, envInt("HOLD_STAGE_SECONDS", 600));
const RAMP_DOWN_SECONDS = Math.max(10, envInt("RAMP_DOWN_SECONDS", 120));
const STATUS_PROBE_RATIO = Math.min(1, Math.max(0.01, envFloat("STATUS_PROBE_RATIO", 0.1)));
const SOCKET_LIFETIME_SECONDS = Math.max(
  30,
  envInt(
    "SOCKET_LIFETIME_SECONDS",
    STAGE_STEP_SECONDS * 4 + HOLD_STAGE_SECONDS + RAMP_DOWN_SECONDS + 60,
  ),
);

const TICKERS = envCsv(
  "TICKERS",
  "AAPL,MSFT,GOOG,AMZN,META,NVDA,TSLA,JPM,V,UNH,MA,PG,COST,HD",
);

if (TICKERS.length === 0) {
  throw new Error("TICKERS cannot be empty");
}

const tier25 = Math.max(1, Math.floor(MAX_SOCKETS * 0.25));
const tier50 = Math.max(tier25, Math.floor(MAX_SOCKETS * 0.5));
const tier75 = Math.max(tier50, Math.floor(MAX_SOCKETS * 0.75));

const subscribeAckMs = new Trend("subscribe_ack_ms", true);
const statusRttMs = new Trend("status_rtt_ms", true);

const handshakeFailureRate = new Rate("handshake_failure_rate");
const appErrorRate = new Rate("app_error_rate");
const parseErrorRate = new Rate("parse_error_rate");
const unexpectedCloseRate = new Rate("unexpected_close_rate");
const readyMessageRate = new Rate("ready_message_rate");
const subscribedMessageRate = new Rate("subscribed_message_rate");
const statusMessageRate = new Rate("status_message_rate");
const sessionCompletedRate = new Rate("session_completed_rate");

const closeCodeTotal = new Counter("ws_close_code_total");

const TRANSPORT_ONLY_THRESHOLDS = {
  checks: ["rate>=0.995"],
  handshake_failure_rate: ["rate<=0.005"],
  session_completed_rate: ["rate>=0.985"],
  ready_message_rate: ["rate>=0.995"],
  subscribed_message_rate: ["rate>=0.995"],
  status_message_rate: ["rate>=0.98"],
  app_error_rate: ["rate<=0.01"],
  parse_error_rate: ["rate<=0.001"],
  unexpected_close_rate: ["rate<=0.01"],
  ws_connecting: ["p(95)<1200", "p(99)<2500"],
  subscribe_ack_ms: ["p(95)<400", "p(99)<900"],
  status_rtt_ms: ["p(95)<600", "p(99)<1200"],
};

const MIXED_TRAIN_INFER_THRESHOLDS = {
  checks: ["rate>=0.99"],
  handshake_failure_rate: ["rate<=0.01"],
  session_completed_rate: ["rate>=0.97"],
  ready_message_rate: ["rate>=0.99"],
  subscribed_message_rate: ["rate>=0.99"],
  status_message_rate: ["rate>=0.96"],
  app_error_rate: ["rate<=0.02"],
  parse_error_rate: ["rate<=0.001"],
  unexpected_close_rate: ["rate<=0.02"],
  ws_connecting: ["p(95)<1800", "p(99)<3200"],
  subscribe_ack_ms: ["p(95)<900", "p(99)<1800"],
  status_rtt_ms: ["p(95)<1200", "p(99)<2200"],
};

const ACTIVE_THRESHOLDS = PROFILE_IS_MIXED
  ? MIXED_TRAIN_INFER_THRESHOLDS
  : TRANSPORT_ONLY_THRESHOLDS;

export const options = {
  scenarios: {
    ws_10k: {
      executor: "ramping-vus",
      exec: "socketSession",
      startVUs: 0,
      gracefulRampDown: "30s",
      gracefulStop: "45s",
      stages: [
        { duration: `${STAGE_STEP_SECONDS}s`, target: tier25 },
        { duration: `${STAGE_STEP_SECONDS}s`, target: tier50 },
        { duration: `${STAGE_STEP_SECONDS}s`, target: tier75 },
        { duration: `${STAGE_STEP_SECONDS}s`, target: MAX_SOCKETS },
        { duration: `${HOLD_STAGE_SECONDS}s`, target: MAX_SOCKETS },
        { duration: `${RAMP_DOWN_SECONDS}s`, target: 0 },
      ],
      tags: {
        test_type: "websocket_10k",
        profile: LOAD_PROFILE,
      },
    },
  },

  thresholds: ACTIVE_THRESHOLDS,

  summaryTrendStats: ["min", "avg", "med", "p(90)", "p(95)", "p(99)", "max"],
  noConnectionReuse: true,
};

export function socketSession() {
  const ticker = TICKERS[(exec.vu.idInTest - 1) % TICKERS.length];
  const sendStatusProbe = Math.random() < STATUS_PROBE_RATIO;

  const subscribePayload = JSON.stringify({
    action: "subscribe",
    tickers: [ticker],
    market: MARKET,
    capital: CAPITAL,
    risk_tolerance: RISK_TOLERANCE,
    enable_train_tunnel: ENABLE_TRAIN_TUNNEL,
  });

  let subscribeSentAt = 0;
  let statusSentAt = 0;
  let readySeen = false;
  let subscribedSeen = false;
  let statusSeen = !sendStatusProbe;
  let appErrored = false;
  let parseErrored = false;
  let closeCode = 1000;

  const response = ws.connect(
    WS_URL,
    {
      tags: {
        scenario: "ws_10k",
        ticker,
        profile: LOAD_PROFILE,
      },
    },
    function (socket) {
      socket.on("open", function () {
        subscribeSentAt = Date.now();
        socket.send(subscribePayload);

        socket.setInterval(function () {
          socket.ping();
        }, 30000);
      });

      socket.on("message", function (rawMessage) {
        let message;

        try {
          message = JSON.parse(rawMessage);
        } catch (err) {
          parseErrored = true;
          return;
        }

        const messageType = String(message.type || "").toLowerCase();

        if (messageType === "tunnel_ready") {
          readySeen = true;
          return;
        }

        if (messageType === "subscribed") {
          if (!subscribedSeen && subscribeSentAt > 0) {
            subscribeAckMs.add(Date.now() - subscribeSentAt);
          }

          subscribedSeen = true;

          if (sendStatusProbe && statusSentAt === 0) {
            statusSentAt = Date.now();
            socket.send(JSON.stringify({ action: "status" }));
          }
          return;
        }

        if (messageType === "status") {
          if (sendStatusProbe && !statusSeen && statusSentAt > 0) {
            statusRttMs.add(Date.now() - statusSentAt);
          }
          statusSeen = true;
          return;
        }

        if (messageType === "error") {
          appErrored = true;
        }
      });

      socket.on("error", function () {
        appErrored = true;
      });

      socket.on("close", function (code) {
        closeCode = Number.isFinite(code) ? Number(code) : 1006;
        closeCodeTotal.add(1, { code: String(closeCode) });
      });

      socket.setTimeout(function () {
        socket.close();
      }, SOCKET_LIFETIME_SECONDS * 1000);
    },
  );

  const handshakeOk = check(response, {
    "websocket upgrade (101)": (r) => r && r.status === 101,
  });

  const closeExpected = closeCode === 1000 || closeCode === 1001;

  handshakeFailureRate.add(!handshakeOk);
  readyMessageRate.add(readySeen);
  subscribedMessageRate.add(subscribedSeen);
  if (sendStatusProbe) {
    statusMessageRate.add(statusSeen);
  }
  appErrorRate.add(appErrored);
  parseErrorRate.add(parseErrored);
  unexpectedCloseRate.add(!closeExpected);
  sessionCompletedRate.add(
    handshakeOk &&
      readySeen &&
      subscribedSeen &&
      statusSeen &&
      !appErrored &&
      !parseErrored &&
      closeExpected,
  );
}

export default function () {
  socketSession();
}
