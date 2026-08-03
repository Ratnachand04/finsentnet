/* ═══════════════════════════════════════════
   CONFIGURATION
   ═══════════════════════════════════════════ */
const API_BASE_STORAGE_KEY = 'finsent_api_base_v1';

function normalizeApiBase(value){
  const trimmed = String(value || '').trim().replace(/\/+$/, '');
  if(!trimmed) return '';
  return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
}

function resolveApiBase(){
  const queryApiBase = new URLSearchParams(window.location.search).get('api_base');
  if(queryApiBase){
    const fromQuery = normalizeApiBase(queryApiBase);
    if(fromQuery) return fromQuery;
  }

  try {
    const fromStorage = normalizeApiBase(localStorage.getItem(API_BASE_STORAGE_KEY));
    if(fromStorage) return fromStorage;
  } catch(e){
    // ignore storage access issues
  }

  const proto = window.location.protocol === 'https:' ? 'https:' : 'http:';
  const host = window.location.hostname || 'localhost';
  return `${proto}//${host}:8000/api`;
}

let API_BASE = resolveApiBase();
let apiBaseProbePromise = null;
const ANALYZE_TIMEOUT_AUTO_MS = 15000;
const ANALYZE_TIMEOUT_MANUAL_MS = 90000;
const APP_STATE_STORAGE_KEY = 'finsent_app_state_v3';

function getAnalyzeTimeoutMs(autoMode){
  if(autoMode) return ANALYZE_TIMEOUT_AUTO_MS;
  const tickerCount = Math.max(1, Array.isArray(state.selectedStocks) ? state.selectedStocks.length : 1);
  const dynamic = ANALYZE_TIMEOUT_MANUAL_MS + Math.max(0, tickerCount - 1) * 20000;
  return Math.min(dynamic, 300000);
}

/* ═══════════════════════════════════════════
   APPLICATION STATE
   ═══════════════════════════════════════════ */
const state = {
  capital: 30000,
  riskTolerance: 0.5,
  horizon: '1M',
  currency: 'USD',
  selectedMarkets: [],
  selectedStocks: [],
  analysisResults: null,   // full API / demo result
  currentStockIdx: 0,
  // chart references
  grafanaChart: null,
  chartResizeObserver: null,
  allocChart: null,
  // training + live state
  trainLossChart: null,
  liveRefreshTimer: null,
  livePriceTimer: null,
  liveChartTimer: null,
  liveTickerTimer: null,
  currentTimeframe: '1D',
  liveChartPollMs: 15000,
  liveChartData: null,
  activeChartTicker: null,
  chartRequestToken: 0,
  liveWs: null,
  liveWsConnected: false,
  liveWsReconnectTimer: null,
  liveWsReconnectDelayMs: 1500,
  liveTunnelSubscribedTicker: null,
  lastWsChartRefreshAt: 0,
  grafanaEmbedConfig: null,
  grafanaEmbedChecked: false,
  grafanaEmbedReady: false,
  lastGrafanaEmbedRefreshAt: 0,
  trainedTickers: {},  // {ticker: true} cache
  modelRegistry: null,
  autoAnalyzeTimer: null,
  autoAnalyzeAbortController: null,
  isAutoAnalyzing: false,
  analyzeInFlight: false,
  analyzeInFlightMode: null,
  analysisRequestCounter: 0,
  latestAnalysisRequestId: 0,
  lastAnalysisSignature: '',
  lastAnalysisAtMs: 0,
  analyzeBtnDefaultHtml: null,
  allowResetNavigation: false,
};

const GRAFANA_EMBED_STORAGE_KEY = 'finsent_grafana_embed_config';

const GRAFANA_THEME = {
  panelBg: '#0F172A',
  panelBgAlt: '#111B2F',
  grid: 'rgba(136, 170, 204, 0.14)',
  text: '#C7D0DB',
  muted: '#8FA2B8',
  up: '#22C55E',
  down: '#F2495C',
  accent: '#5794F2',
  warning: '#FFB357',
  volumeUp: 'rgba(34,197,94,0.30)',
  volumeDown: 'rgba(242,73,92,0.30)',
};

function toDateFromTime(v){
  if(v instanceof Date) return v;
  if(typeof v === 'number'){
    const ms = v > 1e12 ? v : v * 1000;
    return new Date(ms);
  }
  const numeric = Number(v);
  if(Number.isFinite(numeric) && numeric > 0){
    const ms = numeric > 1e12 ? numeric : numeric * 1000;
    return new Date(ms);
  }
  const d = new Date(v);
  if(Number.isNaN(d.getTime())) return new Date();
  return d;
}

function isIndicatorEnabled(id){
  const btn = document.querySelector(`.ind-btn[data-ind="${id}"]`);
  return !!btn && btn.classList.contains('active');
}

function computeEMAFromCandles(candles, period){
  if(!Array.isArray(candles) || candles.length < 2) return [];
  const alpha = 2 / (period + 1);
  let ema = Number(candles[0].close || 0);
  return candles.map((c, idx)=>{
    const close = Number(c.close || 0);
    ema = idx === 0 ? close : (close * alpha) + (ema * (1 - alpha));
    return { time: c.time, value: +ema.toFixed(4) };
  });
}

function computeBollingerFromCandles(candles, period = 20, stdMult = 2){
  if(!Array.isArray(candles) || candles.length < period) return { upper: [], lower: [] };
  const upper = [];
  const lower = [];

  for(let i = period - 1; i < candles.length; i++){
    const slice = candles.slice(i - period + 1, i + 1).map(c=>Number(c.close || 0));
    const mean = slice.reduce((s, v)=>s + v, 0) / period;
    const variance = slice.reduce((s, v)=>s + Math.pow(v - mean, 2), 0) / period;
    const std = Math.sqrt(variance);
    upper.push({ time: candles[i].time, value: +(mean + stdMult * std).toFixed(4) });
    lower.push({ time: candles[i].time, value: +(mean - stdMult * std).toFixed(4) });
  }

  return { upper, lower };
}

function parseBool(value, defaultValue=false){
  if(value===undefined || value===null) return defaultValue;
  if(typeof value==='boolean') return value;
  return ['1','true','yes','on'].includes(String(value).trim().toLowerCase());
}

function parseIntOrNull(value){
  if(value===undefined || value===null || value==='') return null;
  const n = Number.parseInt(String(value), 10);
  return Number.isFinite(n) ? n : null;
}

function toFiniteNumber(value, fallback=0){
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function getApiRoot(apiBase=API_BASE){
  return String(apiBase || '').replace(/\/api$/, '');
}

function persistApiBaseCandidate(apiBase){
  const normalized = normalizeApiBase(apiBase);
  if(!normalized) return;
  API_BASE = normalized;
  try {
    localStorage.setItem(API_BASE_STORAGE_KEY, normalized);
  } catch(e){
    // ignore storage access issues
  }
}

function buildApiBaseCandidates(){
  const proto = window.location.protocol === 'https:' ? 'https:' : 'http:';
  const host = window.location.hostname || 'localhost';
  const candidates = [
    API_BASE,
    `${window.location.origin}/api`,
    `${proto}//${host}:8000/api`,
    `${proto}//${host}:8001/api`,
    `${proto}//${host}:8080/api`,
    `${proto}//${host}:5000/api`,
    `${proto}//${host}:5001/api`,
  ];

  if(host === '127.0.0.1' || host === 'localhost'){
    candidates.push(`${proto}//127.0.0.1:8000/api`);
    candidates.push(`${proto}//localhost:8000/api`);
    candidates.push(`${proto}//127.0.0.1:8001/api`);
    candidates.push(`${proto}//localhost:8001/api`);
  }

  return [...new Set(candidates.map(normalizeApiBase).filter(Boolean))];
}

function healthLooksLikeFinsent(payload){
  if(!payload || typeof payload !== 'object') return false;
  const status = String(payload.status || '').toUpperCase();
  const model = String(payload.model || '').toUpperCase();
  if(status.includes('FINSENT') || model.includes('FINSENT')) return true;
  if(payload.compute && typeof payload.compute === 'object') return true;
  if(payload.modules && typeof payload.modules === 'object') return true;
  return false;
}

async function probeApiHealth(apiBase, timeoutMs=2200){
  const controller = new AbortController();
  const timeoutId = setTimeout(()=>{
    try { controller.abort(); } catch(e) { /* ignore */ }
  }, timeoutMs);

  try {
    const apiRoot = getApiRoot(apiBase);
    const res = await fetch(`${apiRoot}/api/health`, {
      method: 'GET',
      signal: controller.signal,
    });

    if(!res.ok){
      return { ok: false, status: res.status, apiBase };
    }

    let body = null;
    try {
      body = await res.json();
    } catch(e){
      body = null;
    }

    return {
      ok: healthLooksLikeFinsent(body),
      status: res.status,
      apiBase,
      body,
    };
  } catch(e){
    return { ok: false, status: 0, apiBase, error: e };
  } finally {
    clearTimeout(timeoutId);
  }
}

async function ensureApiBaseReady(force=false){
  if(apiBaseProbePromise && !force) return apiBaseProbePromise;

  apiBaseProbePromise = (async ()=>{
    const candidates = buildApiBaseCandidates();
    for(const candidate of candidates){
      const probe = await probeApiHealth(candidate, 1800);
      if(probe.ok){
        if(candidate !== API_BASE){
          console.log(`[FINSENT] API base switched to ${candidate}`);
        }
        persistApiBaseCandidate(candidate);
        return API_BASE;
      }
    }
    return API_BASE;
  })();

  try {
    return await apiBaseProbePromise;
  } finally {
    apiBaseProbePromise = null;
  }
}

function deriveApiBaseFromAnalyzeEndpoint(endpoint){
  try {
    const url = new URL(endpoint, window.location.href);
    let path = String(url.pathname || '').replace(/\/+$/, '');
    if(!path.endsWith('/analyze')) return '';
    path = path.slice(0, -('/analyze'.length));
    if(!path.endsWith('/api')) path = `${path}/api`;
    return normalizeApiBase(`${url.origin}${path}`);
  } catch(e){
    return '';
  }
}

function deriveApiBaseFromLiveSettingsEndpoint(endpoint){
  try {
    const url = new URL(endpoint, window.location.href);
    let path = String(url.pathname || '').replace(/\/+$/, '');
    const suffix = '/live/settings/api-keys';
    if(!path.endsWith(suffix)) return '';
    path = path.slice(0, -suffix.length);
    if(!path.endsWith('/api')) path = `${path}/api`;
    return normalizeApiBase(`${url.origin}${path}`);
  } catch(e){
    return '';
  }
}

function buildLiveSettingsEndpoints(){
  const apiRoot = getApiRoot(API_BASE);
  return [...new Set([
    `${API_BASE}/live/settings/api-keys`,
    `${apiRoot}/api/live/settings/api-keys`,
    `${apiRoot}/live/settings/api-keys`,
  ])];
}

async function fetchApiKeyStatusWithFallback(){
  await ensureApiBaseReady(true);

  const endpoints = buildLiveSettingsEndpoints();
  let lastErr = null;

  for(const endpoint of endpoints){
    let res;
    try {
      res = await fetch(endpoint, { method: 'GET' });
    } catch(e){
      lastErr = e;
      continue;
    }

    if(!res.ok){
      if(res.status === 404 || res.status === 405){
        lastErr = new Error(`HTTP ${res.status}`);
        continue;
      }
      lastErr = new Error(`HTTP ${res.status}`);
      continue;
    }

    let data = null;
    try {
      data = await res.json();
    } catch(e){
      lastErr = e;
      continue;
    }

    if(data && typeof data === 'object' && data.sources && typeof data.sources === 'object'){
      const detectedBase = deriveApiBaseFromLiveSettingsEndpoint(endpoint);
      if(detectedBase) persistApiBaseCandidate(detectedBase);
      return data;
    }

    lastErr = new Error('Malformed key status response');
  }

  throw lastErr || new Error('Unable to fetch key status');
}

async function saveApiKeysWithFallback(payload){
  await ensureApiBaseReady(true);

  const endpoints = buildLiveSettingsEndpoints();
  let lastErr = null;

  for(const endpoint of endpoints){
    let res;
    try {
      res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch(e){
      lastErr = e;
      continue;
    }

    let data = {};
    try { data = await res.json(); } catch(e) { data = {}; }

    if(res.ok && data.status === 'ok'){
      const detectedBase = deriveApiBaseFromLiveSettingsEndpoint(endpoint);
      if(detectedBase) persistApiBaseCandidate(detectedBase);
      return data;
    }

    if(res.status === 404 || res.status === 405){
      lastErr = new Error(`HTTP ${res.status}`);
      continue;
    }

    const detail = (data && data.detail) ? String(data.detail) : `HTTP ${res.status}`;
    lastErr = new Error(detail);
  }

  throw lastErr || new Error('Unable to save API keys');
}

async function requestAnalysisWithFallback(payload, signal){
  await ensureApiBaseReady();

  const apiRoot = getApiRoot(API_BASE);
  const endpoints = [
    `${API_BASE}/analyze`,
    `${apiRoot}/api/analyze`,
    `${apiRoot}/analyze`,
  ];

  const tried = new Set();
  let lastErr = null;

  for(const endpoint of endpoints){
    if(tried.has(endpoint)) continue;
    tried.add(endpoint);

    let res;
    try {
      res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal,
      });
    } catch(e){
      if(e && e.name === 'AbortError') throw e;
      lastErr = e;
      continue;
    }

    if(res.ok){
      const out = await res.json();
      const detectedBase = deriveApiBaseFromAnalyzeEndpoint(endpoint);
      if(detectedBase) persistApiBaseCandidate(detectedBase);
      return out;
    }

    if(res.status === 404 || res.status === 405){
      lastErr = new Error(`HTTP ${res.status}`);
      continue;
    }

    let detail = `HTTP ${res.status}`;
    try {
      const errBody = await res.json();
      if(errBody && errBody.detail !== undefined){
        detail += `: ${typeof errBody.detail === 'string' ? errBody.detail : JSON.stringify(errBody.detail)}`;
      }
    } catch(e){
      // ignore body parse issues
    }
    throw new Error(detail);
  }

  throw lastErr || new Error('HTTP 405');
}

function enrichSignalsWithAnalysisDetails(out){
  if(!out || !Array.isArray(out.signals) || !Array.isArray(out.stock_details)) return;

  const detailsByTicker = new Map();
  out.stock_details.forEach((detail)=>{
    const ticker = String(detail?.ticker || '').toUpperCase().trim();
    if(ticker) detailsByTicker.set(ticker, detail);
  });

  out.signals.forEach((sig)=>{
    const ticker = String(sig?.ticker || '').toUpperCase().trim();
    const detail = detailsByTicker.get(ticker);
    if(!detail) return;

    const quote = detail.live_quote || {};
    if(Number.isFinite(Number(quote.price))){
      sig._analysisLiveQuote = quote;
      if(!Number.isFinite(Number(sig.entry_price)) || Number(sig.entry_price) <= 0){
        sig.entry_price = Number(quote.price);
      }
    }

    const news = Array.isArray(detail.top_live_news) ? detail.top_live_news : [];
    if(news.length){
      sig._analysisLiveNews = news.slice(0, 10);
    }
  });
}

async function isBackendReachable(timeoutMs=2200){
  await ensureApiBaseReady();
  const probe = await probeApiHealth(API_BASE, timeoutMs);
  return !!probe.ok;
}

function normalizeGrafanaEmbedConfig(raw){
  if(!raw) return null;
  const cfg = raw.config || raw;
  const out = {
    enabled: parseBool(cfg.enabled, true),
    grafana_url: String(cfg.grafana_url || cfg.url || '').trim().replace(/\/$/, ''),
    dashboard_uid: String(cfg.dashboard_uid || cfg.uid || '').trim(),
    dashboard_slug: String(cfg.dashboard_slug || cfg.slug || 'finsent-live').trim(),
    org_id: parseIntOrNull(cfg.org_id) || 1,
    price_panel_id: parseIntOrNull(cfg.price_panel_id || cfg.panel_id),
    volume_panel_id: parseIntOrNull(cfg.volume_panel_id),
    theme: String(cfg.theme || 'dark').trim().toLowerCase(),
    refresh: String(cfg.refresh || '5s').trim(),
    ticker_var: String(cfg.ticker_var || 'ticker').trim(),
    market_var: String(cfg.market_var || 'market').trim(),
    timeframe_var: String(cfg.timeframe_var || 'timeframe').trim(),
  };

  // Guard against malformed URLs so embed mode can safely fall back.
  let parsedUrl = null;
  try {
    parsedUrl = new URL(out.grafana_url);
  } catch(e) {
    return null;
  }

  if(!['http:', 'https:'].includes(parsedUrl.protocol)) return null;
  out.grafana_url = `${parsedUrl.origin}${parsedUrl.pathname}`.replace(/\/$/, '');

  if(!out.enabled) return null;
  if(!out.grafana_url || !out.dashboard_uid || !Number.isFinite(out.price_panel_id)) return null;
  return out;
}

function getGrafanaRangeForTimeframe(tf){
  const map = {
    '1m': 'now-2h',
    '5m': 'now-12h',
    '15m': 'now-24h',
    '1H': 'now-7d',
    '4H': 'now-30d',
    '1D': 'now-180d',
  };
  return map[tf] || 'now-7d';
}

function buildGrafanaPanelUrl(config, sig, panelId, forceTimestamp){
  try {
    const url = new URL(`${config.grafana_url}/d-solo/${encodeURIComponent(config.dashboard_uid)}/${encodeURIComponent(config.dashboard_slug)}`);
    url.searchParams.set('orgId', String(config.org_id || 1));
    url.searchParams.set('panelId', String(panelId));
    url.searchParams.set('theme', config.theme || 'dark');
    url.searchParams.set('from', getGrafanaRangeForTimeframe(state.currentTimeframe));
    url.searchParams.set('to', 'now');
    url.searchParams.set('timezone', 'browser');
    url.searchParams.set('refresh', config.refresh || '5s');
    url.searchParams.set('kiosk', 'tv');
    const tickerVar = `var-${config.ticker_var || 'ticker'}`;
    const selected = Array.isArray(state.selectedStocks)
      ? [...new Set(state.selectedStocks.map(t=>String(t || '').toUpperCase().trim()).filter(Boolean))]
      : [];
    const tickerValues = selected.length ? selected : [String(sig.ticker || '').toUpperCase().trim()].filter(Boolean);
    url.searchParams.delete(tickerVar);
    tickerValues.forEach(t=>url.searchParams.append(tickerVar, t));
    url.searchParams.set(`var-${config.market_var || 'market'}`, String(state.selectedMarkets[0] || 'SP500'));
    url.searchParams.set(`var-${config.timeframe_var || 'timeframe'}`, String(state.currentTimeframe || '1D'));
    if(forceTimestamp) url.searchParams.set('_finsent_ts', String(Date.now()));
    return url.toString();
  } catch(e){
    return '';
  }
}

async function loadGrafanaEmbedConfig(){
  if(state.grafanaEmbedChecked) return state.grafanaEmbedConfig;
  state.grafanaEmbedChecked = true;

  let cfg = null;
  let backendAnswered = false;

  try {
    const res = await fetch(`${API_BASE}/live/grafana/embed-config`);
    if(res.ok){
      backendAnswered = true;
      const data = await res.json();
      if(data.ready){
        cfg = normalizeGrafanaEmbedConfig(data);
      } else {
        // Backend is authoritative for runtime embed readiness.
        // If backend says embed is not ready/reachable, force local Plotly path.
        try { localStorage.removeItem(GRAFANA_EMBED_STORAGE_KEY); } catch(e) { /* ignore */ }
        state.grafanaEmbedConfig = null;
        state.grafanaEmbedReady = false;
        return null;
      }
    }
  } catch(e){
    // backend unavailable -> force local Plotly mode
  }

  if(!backendAnswered){
    state.grafanaEmbedConfig = null;
    state.grafanaEmbedReady = false;
    return null;
  }

  state.grafanaEmbedConfig = cfg;
  state.grafanaEmbedReady = !!cfg;
  return state.grafanaEmbedConfig;
}

async function ensureGrafanaEmbedConfig(){
  if(state.grafanaEmbedChecked) return state.grafanaEmbedConfig;
  return loadGrafanaEmbedConfig();
}

function refreshGrafanaEmbeddedPanels(sig, force=false){
  if(!state.grafanaEmbedReady || !state.grafanaEmbedConfig) return false;
  const container = document.getElementById('chartContainer');
  if(!container) return false;

  const wrap = container.querySelector('.grafana-embed-wrap');
  if(!wrap) return false;

  const now = Date.now();
  if(!force && now - state.lastGrafanaEmbedRefreshAt < 1500) return true;
  state.lastGrafanaEmbedRefreshAt = now;

  const iframes = container.querySelectorAll('iframe[data-panel-id]');
  if(!iframes.length) return false;

  iframes.forEach(frame=>{
    const panelId = Number(frame.getAttribute('data-panel-id') || 0);
    if(!Number.isFinite(panelId) || panelId <= 0) return;
    const nextSrc = buildGrafanaPanelUrl(state.grafanaEmbedConfig, sig, panelId, true);
    if(nextSrc) frame.src = nextSrc;
  });
  return true;
}

async function renderGrafanaEmbeddedPanels(sig){
  const cfg = await ensureGrafanaEmbedConfig();
  if(!cfg) return false;

  const container = document.getElementById('chartContainer');
  if(!container) return false;

  const pricePanelId = Number(cfg.price_panel_id || 0);
  if(!Number.isFinite(pricePanelId) || pricePanelId <= 0) return false;

  const volPanelId = Number(cfg.volume_panel_id || 0);
  const hasVolumePanel = Number.isFinite(volPanelId) && volPanelId > 0;

  const priceSrc = buildGrafanaPanelUrl(cfg, sig, pricePanelId, true);
  if(!priceSrc) return false;
  const volumeSrc = hasVolumePanel ? buildGrafanaPanelUrl(cfg, sig, volPanelId, true) : '';
  const useVolumePanel = hasVolumePanel && !!volumeSrc;

  container.innerHTML = `
    <div class="grafana-embed-wrap ${useVolumePanel ? 'with-volume' : 'single'}">
      <iframe class="grafana-embed-panel grafana-embed-price" data-panel-id="${pricePanelId}" src="${priceSrc}" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
      ${useVolumePanel ? `<iframe class="grafana-embed-panel grafana-embed-volume" data-panel-id="${volPanelId}" src="${volumeSrc}" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>` : ''}
    </div>
  `;

  state.grafanaChart = container;
  state.activeChartTicker = sig.ticker;
  state.lastGrafanaEmbedRefreshAt = Date.now();
  return true;
}

function getTimeframeBucketSeconds(tf){
  const map = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '1H': 3600,
    '4H': 14400,
    '1D': 86400,
  };
  return map[tf] || 86400;
}

function toEpochSeconds(value){
  if(!value) return Math.floor(Date.now()/1000);
  if(typeof value === 'number') return value > 1e12 ? Math.floor(value/1000) : Math.floor(value);
  const d = new Date(value);
  if(Number.isNaN(d.getTime())) return Math.floor(Date.now()/1000);
  return Math.floor(d.getTime()/1000);
}

/* ═══════════════════════════════════════════
   DEMO STOCK DATABASE
   ═══════════════════════════════════════════ */
const STOCK_DB = {
  SP500: [
    {ticker:'AAPL',  name:'Apple Inc.',            price:182.52, change:2.31,  sector:'tech'},
    {ticker:'MSFT',  name:'Microsoft Corp',        price:415.20, change:1.12,  sector:'tech'},
    {ticker:'NVDA',  name:'NVIDIA Corporation',    price:875.30, change:4.05,  sector:'tech'},
    {ticker:'GOOGL', name:'Alphabet Inc.',         price:152.40, change:-0.54, sector:'tech'},
    {ticker:'AMZN',  name:'Amazon.com Inc.',       price:185.60, change:1.87,  sector:'tech'},
    {ticker:'META',  name:'Meta Platforms',        price:502.30, change:2.15,  sector:'tech'},
    {ticker:'TSLA',  name:'Tesla Inc.',            price:248.90, change:-1.23, sector:'tech'},
    {ticker:'JPM',   name:'JPMorgan Chase & Co',   price:196.80, change:0.92,  sector:'finance'},
    {ticker:'V',     name:'Visa Inc.',             price:282.40, change:0.67,  sector:'finance'},
    {ticker:'BAC',   name:'Bank of America',       price:38.20,  change:0.55,  sector:'finance'},
    {ticker:'GS',    name:'Goldman Sachs',         price:458.90, change:1.38,  sector:'finance'},
    {ticker:'JNJ',   name:'Johnson & Johnson',     price:152.30, change:-0.34, sector:'health'},
    {ticker:'UNH',   name:'UnitedHealth Group',    price:528.90, change:1.45,  sector:'health'},
    {ticker:'PFE',   name:'Pfizer Inc.',           price:26.80,  change:-0.92, sector:'health'},
    {ticker:'XOM',   name:'Exxon Mobil Corp',      price:104.20, change:0.88,  sector:'energy'},
    {ticker:'CVX',   name:'Chevron Corporation',   price:156.40, change:0.62,  sector:'energy'},
    {ticker:'PG',    name:'Procter & Gamble',      price:162.70, change:0.21,  sector:'consumer'},
    {ticker:'KO',    name:'Coca-Cola Co',          price:60.40,  change:0.33,  sector:'consumer'},
    {ticker:'WMT',   name:'Walmart Inc.',          price:168.20, change:0.76,  sector:'consumer'},
  ],
  NASDAQ: [
    {ticker:'AAPL',  name:'Apple Inc.',            price:182.52, change:2.31,  sector:'tech'},
    {ticker:'MSFT',  name:'Microsoft Corp',        price:415.20, change:1.12,  sector:'tech'},
    {ticker:'NVDA',  name:'NVIDIA Corporation',    price:875.30, change:4.05,  sector:'tech'},
    {ticker:'AMD',   name:'Advanced Micro Devices', price:178.60, change:3.22,  sector:'tech'},
    {ticker:'NFLX',  name:'Netflix Inc.',          price:628.40, change:1.89,  sector:'tech'},
    {ticker:'INTC',  name:'Intel Corporation',     price:30.80,  change:-1.15, sector:'tech'},
    {ticker:'AVGO',  name:'Broadcom Inc.',         price:1380.50,change:2.68,  sector:'tech'},
  ],
  NYSE: [
    {ticker:'JPM',   name:'JPMorgan Chase & Co',   price:196.80, change:0.92,  sector:'finance'},
    {ticker:'BAC',   name:'Bank of America',       price:38.20,  change:0.55,  sector:'finance'},
    {ticker:'WMT',   name:'Walmart Inc.',          price:168.20, change:0.76,  sector:'consumer'},
    {ticker:'DIS',   name:'Walt Disney Co.',       price:112.40, change:0.88,  sector:'consumer'},
  ],
  BSE: [
    {ticker:'RELIANCE.BO', name:'Reliance Industries', price:2840, change:0.82, sector:'energy'},
    {ticker:'TCS.BO',      name:'Tata Consultancy',    price:3920, change:1.15, sector:'tech'},
    {ticker:'HDFCBANK.BO', name:'HDFC Bank',           price:1650, change:0.45, sector:'finance'},
    {ticker:'INFY.BO',     name:'Infosys Ltd.',        price:1580, change:0.92, sector:'tech'},
  ],
  NSE: [
    {ticker:'RELIANCE.NS', name:'Reliance Industries', price:2840, change:0.82, sector:'energy'},
    {ticker:'TCS.NS',      name:'Tata Consultancy',    price:3920, change:1.15, sector:'tech'},
    {ticker:'HDFCBANK.NS', name:'HDFC Bank',           price:1650, change:0.45, sector:'finance'},
    {ticker:'INFY.NS',     name:'Infosys Ltd.',        price:1580, change:0.92, sector:'tech'},
    {ticker:'ICICIBANK.NS',name:'ICICI Bank',          price:1120, change:1.28, sector:'finance'},
    {ticker:'SBIN.NS',     name:'State Bank of India',  price:780,  change:0.55, sector:'finance'},
  ],
  COMMODITIES: [
    {ticker:'GC=F', name:'Gold Futures',  price:2340, change:-0.21, sector:'energy'},
    {ticker:'SI=F', name:'Silver Futures', price:27.80, change:1.05, sector:'energy'},
    {ticker:'CL=F', name:'Crude Oil WTI', price:78.40, change:-0.88, sector:'energy'},
    {ticker:'NG=F', name:'Natural Gas',   price:2.18,  change:2.30,  sector:'energy'},
  ],
  CRYPTO: [
    {ticker:'BTC-USD', name:'Bitcoin',   price:67420, change:3.42, sector:'tech'},
    {ticker:'ETH-USD', name:'Ethereum',  price:3580, change:2.18, sector:'tech'},
    {ticker:'SOL-USD', name:'Solana',    price:148.20, change:5.67, sector:'tech'},
    {ticker:'BNB-USD', name:'BNB',       price:598.40, change:1.32, sector:'tech'},
    {ticker:'XRP-USD', name:'XRP',       price:0.52, change:-0.88, sector:'tech'},
  ],
};

/* ═══════════════════════════════════════════
   UTILITY HELPERS
   ═══════════════════════════════════════════ */
function hashStr(s){let h=0;for(let i=0;i<s.length;i++){h=((h<<5)-h)+s.charCodeAt(i);h|=0;}return Math.abs(h);}
function mkRng(seed){return function(){seed=(seed*16807)%2147483647;return(seed-1)/2147483646;};}

function currSym(){
  return {USD:'$',INR:'₹',EUR:'€',GBP:'£'}[state.currency]||'$';
}
function fmt$(v){
  if(v==null) return currSym()+'0';
  return currSym()+Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
}

function getAllStocks(){
  let all = [];
  const markets = state.selectedMarkets.length > 0 ? state.selectedMarkets : ['SP500'];
  markets.forEach(m=>{ if(STOCK_DB[m]) all=all.concat(STOCK_DB[m]); });
  const seen = new Set();
  return all.filter(s=>{ if(seen.has(s.ticker)) return false; seen.add(s.ticker); return true; });
}

/* ═══════════════════════════════════════════
   ANIMATION 1 — PARTICLE NEURAL NETWORK
   ═══════════════════════════════════════════ */
(function(){
  const cv = document.getElementById('particleCanvas');
  const cx = cv.getContext('2d');
  const P = [];
  const N = 75;
  const maxDist = 140;

  function resize(){ cv.width=innerWidth; cv.height=innerHeight; }
  addEventListener('resize', resize); resize();

  for(let i=0;i<N;i++){
    P.push({
      x:Math.random()*cv.width, y:Math.random()*cv.height,
      vx:(Math.random()-0.5)*0.35, vy:(Math.random()-0.5)*0.35,
      r:1.2+Math.random()*1.8, ph:Math.random()*6.28,
    });
  }

  (function loop(){
    cx.clearRect(0,0,cv.width,cv.height);
    const t = performance.now()*0.001;
    // connections
    for(let i=0;i<P.length;i++){
      for(let j=i+1;j<P.length;j++){
        const dx=P[i].x-P[j].x, dy=P[i].y-P[j].y;
        const d=Math.sqrt(dx*dx+dy*dy);
        if(d<maxDist){
          cx.beginPath();
          cx.strokeStyle=`rgba(0,245,255,${(1-d/maxDist)*0.12})`;
          cx.lineWidth=0.5;
          cx.moveTo(P[i].x,P[i].y); cx.lineTo(P[j].x,P[j].y);
          cx.stroke();
        }
      }
    }
    // nodes
    P.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>cv.width)  p.vx*=-1;
      if(p.y<0||p.y>cv.height) p.vy*=-1;
      const glow = 0.35 + 0.3*Math.sin(t*2+p.ph);
      cx.beginPath();
      cx.arc(p.x,p.y,p.r,0,6.28);
      cx.fillStyle=`rgba(0,245,255,${glow})`;
      cx.fill();
    });
    requestAnimationFrame(loop);
  })();
})();

/* ═══════════════════════════════════════════
   ANIMATION 6 — TICKER STRIP
   ═══════════════════════════════════════════ */
function buildTickerFallback(){
  return [
    { s:'AAPL', p:'$182.52', c:'+2.31%', up:true },
    { s:'MSFT', p:'$415.20', c:'+1.12%', up:true },
    { s:'NVDA', p:'$875.30', c:'+4.05%', up:true },
    { s:'GOOGL', p:'$152.40', c:'-0.54%', up:false },
    { s:'BTC-USD', p:'$67,420', c:'+3.42%', up:true },
    { s:'CL=F', p:'$78.40', c:'-0.88%', up:false },
  ];
}

function renderTickerItems(items){
  const el = document.getElementById('tickerTrack');
  if(!el) return;
  let h = '';
  for(let r=0; r<2; r++) items.forEach(t=>{
    h += `<div class="ticker-item"><span class="symbol">${t.s}</span> <span class="price">${t.p}</span> <span class="${t.up?'up':'down'}">${t.c}</span></div>`;
  });
  el.innerHTML = h;
}

function normalizeMarketListForTicker(){
  const picked = state.selectedMarkets && state.selectedMarkets.length ? state.selectedMarkets : ['SP500','NASDAQ','NSE','CRYPTO','COMMODITIES'];
  return picked.join(',');
}

async function refreshTickerStrip(){
  try {
    const marketParam = encodeURIComponent(normalizeMarketListForTicker());
    const res = await fetch(`${API_BASE}/live/market-snapshot?markets=${marketParam}&per_market_limit=8`);
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const ratioItems = (data.markets || []).map(m=>{
      const upPct = ((m.advance_ratio || 0) * 100).toFixed(0);
      const dnPct = ((m.decline_ratio || 0) * 100).toFixed(0);
      const avg = Number(m.avg_change_pct || 0);
      return {
        s: `${m.market} U/D`,
        p: `${upPct}/${dnPct}`,
        c: `${avg>=0?'+':''}${avg.toFixed(2)}%`,
        up: avg >= 0,
      };
    });

    const quoteItems = (data.quotes || []).slice(0, 10).map(q=>{
      const pct = Number(q.change_pct || 0);
      const price = q.price == null ? '-' : Number(q.price).toLocaleString(undefined, { maximumFractionDigits: 2 });
      return {
        s: q.ticker,
        p: price,
        c: `${pct>=0?'+':''}${pct.toFixed(2)}%`,
        up: pct >= 0,
      };
    });

    const merged = ratioItems.concat(quoteItems);
    if(merged.length){
      renderTickerItems(merged);
      return;
    }
  } catch(e){
    // Fallback below keeps UI alive even if backend is down.
  }

  renderTickerItems(buildTickerFallback());
}

function startTickerStripLive(){
  if(state.liveTickerTimer){
    clearInterval(state.liveTickerTimer);
    state.liveTickerTimer = null;
  }
  refreshTickerStrip();
  state.liveTickerTimer = setInterval(refreshTickerStrip, 20000);
}

function getActiveScreenNumber(){
  const active = document.querySelector('.screen.active');
  if(!active || !active.id) return 1;
  const m = String(active.id).match(/^screen(\d+)$/i);
  return m ? Number(m[1]) : 1;
}

function lookupStockName(ticker){
  const target = String(ticker || '').toUpperCase().trim();
  if(!target) return '';

  for(const market of Object.keys(STOCK_DB || {})){
    const rows = STOCK_DB[market] || [];
    const hit = rows.find(s=>String(s.ticker || '').toUpperCase()===target);
    if(hit?.name) return String(hit.name);
  }

  return '';
}

function compactAnalysisForStorage(analysis){
  if(!analysis || !Array.isArray(analysis.signals) || !analysis.signals.length) return null;

  const keepFields = [
    'ticker','name','direction','confidence','predicted_return','predicted_downside',
    'entry_price','target_price','stop_loss','risk_reward','kelly_fraction',
    'quantity','capital_required','time_horizon','reasoning','regime',
    'sentiment_score','technical_score','fusion_confidence','_live',
    '_analysisLiveQuote','_analysisLiveNews',
  ];

  const compactSignals = analysis.signals.map((sig)=>{
    const out = {};
    keepFields.forEach((field)=>{
      if(sig && sig[field]!==undefined) out[field] = sig[field];
    });

    out.ticker = String(out.ticker || '').toUpperCase().trim();
    if(!out.name) out.name = lookupStockName(out.ticker);
    if(!Array.isArray(out.reasoning)) out.reasoning = [];
    if(!Number.isFinite(Number(out.predicted_downside))){
      out.predicted_downside = -Math.abs(toFiniteNumber(out.predicted_return, 0)) * 0.55;
    }
    if(!Number.isFinite(Number(out.fusion_confidence))){
      out.fusion_confidence = toFiniteNumber(out.confidence, 50) * 0.9;
    }
    out._live = !!out._live;
    return out;
  }).filter(sig=>!!sig.ticker);

  if(!compactSignals.length) return null;

  return {
    version: 1,
    status: String(analysis.status || 'success'),
    analysis_time_seconds: toFiniteNumber(analysis.analysis_time_seconds, 0),
    ticker_markets: analysis.ticker_markets || {},
    signals: compactSignals,
    portfolio: {
      allocation: analysis.portfolio?.allocation || {},
    },
    risk: analysis.risk || {},
    saved_at: Date.now(),
  };
}

function restoreAnalysisFromStorage(saved){
  if(!saved || !Array.isArray(saved.signals) || !saved.signals.length) return null;

  const signals = saved.signals.map((sig)=>{
    const ticker = String(sig?.ticker || '').toUpperCase().trim();
    if(!ticker) return null;

    const confidence = toFiniteNumber(sig.confidence, 50);
    const predictedReturn = toFiniteNumber(sig.predicted_return, 0);

    return {
      ticker,
      name: String(sig.name || lookupStockName(ticker) || ticker),
      direction: String(sig.direction || 'HOLD'),
      confidence,
      predicted_return: predictedReturn,
      predicted_downside: toFiniteNumber(sig.predicted_downside, -Math.abs(predictedReturn) * 0.55),
      entry_price: toFiniteNumber(sig.entry_price, 0),
      target_price: toFiniteNumber(sig.target_price, 0),
      stop_loss: toFiniteNumber(sig.stop_loss, 0),
      risk_reward: toFiniteNumber(sig.risk_reward, 1),
      kelly_fraction: toFiniteNumber(sig.kelly_fraction, 0),
      quantity: Math.max(0, Math.floor(toFiniteNumber(sig.quantity, 0))),
      capital_required: Math.max(0, toFiniteNumber(sig.capital_required, 0)),
      time_horizon: String(sig.time_horizon || '1-2 weeks'),
      reasoning: Array.isArray(sig.reasoning) ? sig.reasoning : [],
      regime: String(sig.regime || 'UNKNOWN'),
      sentiment_score: toFiniteNumber(sig.sentiment_score, 50),
      technical_score: toFiniteNumber(sig.technical_score, 50),
      fusion_confidence: toFiniteNumber(sig.fusion_confidence, confidence * 0.9),
      _live: !!sig._live,
      _analysisLiveQuote: sig._analysisLiveQuote || null,
      _analysisLiveNews: Array.isArray(sig._analysisLiveNews) ? sig._analysisLiveNews.slice(0, 10) : [],
    };
  }).filter(Boolean);

  if(!signals.length) return null;

  return {
    status: String(saved.status || 'success'),
    analysis_time_seconds: toFiniteNumber(saved.analysis_time_seconds, 0),
    ticker_markets: saved.ticker_markets || {},
    signals,
    portfolio: {
      allocation: saved.portfolio?.allocation || {},
    },
    risk: saved.risk || {},
  };
}

function persistAppState(){
  try {
    const payload = {
      selectedMarkets: Array.isArray(state.selectedMarkets) ? state.selectedMarkets : [],
      selectedStocks: Array.isArray(state.selectedStocks) ? state.selectedStocks : [],
      analysisSummary: compactAnalysisForStorage(state.analysisResults),
      currentStockIdx: Number(state.currentStockIdx || 0),
      currentTimeframe: String(state.currentTimeframe || '1D'),
      capital: Number(state.capital || 30000),
      riskTolerance: Number(state.riskTolerance || 0.5),
      horizon: String(state.horizon || '1M'),
      currency: String(state.currency || 'USD'),
      lastAnalysisAtMs: Number(state.lastAnalysisAtMs || 0),
      lastAnalysisSignature: String(state.lastAnalysisSignature || ''),
      screen: getActiveScreenNumber(),
    };
    localStorage.setItem(APP_STATE_STORAGE_KEY, JSON.stringify(payload));
  } catch(e){
    console.log('[FINSENT] State persistence skipped:', e?.message || e);
  }
}

function restoreAppState(){
  try {
    const raw = localStorage.getItem(APP_STATE_STORAGE_KEY);
    if(!raw) return;
    const saved = JSON.parse(raw);
    if(!saved || typeof saved !== 'object') return;

    state.selectedMarkets = Array.isArray(saved.selectedMarkets) ? saved.selectedMarkets : [];
    state.selectedStocks = Array.isArray(saved.selectedStocks) ? saved.selectedStocks : [];
    const compact = saved.analysisSummary || compactAnalysisForStorage(saved.analysisResults || null);
    state.analysisResults = restoreAnalysisFromStorage(compact);
    state.currentStockIdx = Number(saved.currentStockIdx || 0);
    state.currentTimeframe = String(saved.currentTimeframe || '1D');
    state.capital = Number(saved.capital || state.capital || 30000);
    state.riskTolerance = Number(saved.riskTolerance || state.riskTolerance || 0.5);
    state.horizon = String(saved.horizon || state.horizon || '1M');
    state.currency = String(saved.currency || state.currency || 'USD');
    state.lastAnalysisAtMs = Number(saved.lastAnalysisAtMs || 0);
    state.lastAnalysisSignature = String(saved.lastAnalysisSignature || '');

    renderSelectedStocks();
    const screen = Number(saved.screen || 1);
    if(screen === 4 && state.analysisResults?.signals?.length){
      goToScreen(4);
      if(state.currentStockIdx > 0 && state.currentStockIdx < state.analysisResults.signals.length){
        switchStock(state.currentStockIdx);
      }
    }
  } catch(e){
    console.log('[FINSENT] State restore skipped:', e?.message || e);
  }
}

function startNewAnalysis(){
  state.allowResetNavigation = true;
  state.analysisResults = null;
  persistAppState();
  goToScreen(1);
  state.allowResetNavigation = false;
}

/* ═══════════════════════════════════════════
   SCREEN NAVIGATION
   ═══════════════════════════════════════════ */
function goToScreen(n){
  const currentScreen = getActiveScreenNumber();
  if(
    n===1 &&
    currentScreen===4 &&
    state.analysisResults?.signals?.length &&
    !state.allowResetNavigation
  ){
    return;
  }

  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  const target = document.getElementById('screen'+n);
  if(target) target.classList.add('active');
  updateApiKeysFabVisibility(n);
  if(n===3) populateStockList();
  if(n===4){
    try {
      renderDashboard();
      startLiveUpdates();
    } catch(err) {
      console.log('[FINSENT] Dashboard render guard:', err?.message || err);
    }
  }
  if(n===5) renderPortfolioSummary();
  // Stop live updates when navigating away from screen 4
  if(n!==4) stopLiveUpdates({closeSocket:true});
  persistAppState();
  scrollTo({top:0,behavior:'smooth'});
}

function updateApiKeysFabVisibility(screenNumber){
  const fab = document.querySelector('.api-keys-fab');
  if(!fab) return;
  fab.style.display = screenNumber === 1 ? 'inline-flex' : 'none';
}

/* ═══════════════════════════════════════════
   SCREEN 1 — CAPITAL CONFIG
   ═══════════════════════════════════════════ */
function setCapital(val){
  state.capital=val;
  document.getElementById('capitalInput').value=val.toLocaleString();
  document.querySelectorAll('.quick-btn').forEach(b=>b.classList.remove('active'));
  if(event&&event.target) event.target.classList.add('active');
}

function updateCurrency(){
  state.currency=document.getElementById('currencySelect').value;
  document.getElementById('currSymbol').textContent=currSym();
}

function updateRiskLabel(){
  const v=+document.getElementById('riskSlider').value;
  state.riskTolerance=v/10;
  const labels=['Ultra Safe','Very Conservative','Conservative','Mod. Conservative','Moderate',
    'Moderate','Moderate Aggressive','Aggressive','Very Aggressive','Very Aggressive','Ultra Aggressive'];
  document.getElementById('riskLabel').textContent=labels[v];
}

function setHorizon(btn,val){
  state.horizon=val;
  document.querySelectorAll('.horizon-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}

/* ═══════════════════════════════════════════
   API KEY + MODEL SETTINGS
   ═══════════════════════════════════════════ */
function getSelectedProvider(){
  const base = document.getElementById('providerSelect')?.value || 'fmp';
  if(base !== 'custom') return base;
  const custom = (document.getElementById('customProviderInput')?.value || '').trim().toLowerCase();
  return custom || 'custom';
}

function toggleCustomProviderInput(){
  const sel = document.getElementById('providerSelect');
  const wrap = document.getElementById('customProviderWrap');
  if(!sel || !wrap) return;
  wrap.style.display = sel.value === 'custom' ? 'block' : 'none';
}

function setFmpCountText(count, suffix){
  const label = count + ' key' + (count!==1 ? 's' : '') + (suffix || '');
  document.querySelectorAll('[data-fmp-count]').forEach(el=>{ el.textContent = label; });
}

function getTotalConfiguredKeys(providers){
  return Object.values(providers || {}).reduce((sum, meta)=>{
    const n = Number(meta?.keys_count || 0);
    return sum + (Number.isFinite(n) && n > 0 ? n : 0);
  }, 0);
}

function collectLocalProviderKeys(){
  const out = {};
  try {
    for(let i=0; i<localStorage.length; i++){
      const key = localStorage.key(i);
      if(!key || !key.startsWith('provider_keys_')) continue;
      const provider = key.slice('provider_keys_'.length).trim().toLowerCase();
      if(!provider) continue;
      const raw = localStorage.getItem(key);
      if(!raw) continue;
      const parsed = JSON.parse(raw);
      const arr = Array.isArray(parsed)
        ? parsed.map(v=>String(v || '').trim()).filter(v=>v.length > 5)
        : [];
      if(arr.length) out[provider] = arr;
    }
  } catch(e){
    // ignore storage parse issues
  }
  return out;
}

function buildProviderPayloadFromMap(providerMap){
  const payload = { providers: {} };
  Object.entries(providerMap || {}).forEach(([provider, keys])=>{
    const p = String(provider || '').trim().toLowerCase();
    const arr = Array.isArray(keys) ? keys.map(v=>String(v || '').trim()).filter(v=>v.length > 5) : [];
    if(!p || !arr.length) return;

    payload.providers[p] = arr;
    if(p === 'fmp') payload.fmp_keys = arr;
    if(p === 'finnhub') payload.finnhub = arr[0];
    if(p === 'alpha_vantage') payload.alpha_vantage = arr[0];
    if(p === 'news_api') payload.news_api = arr[0];
  });
  return payload;
}

async function restoreProviderKeysFromLocal(){
  const providerMap = collectLocalProviderKeys();
  if(!Object.keys(providerMap).length) return null;
  const payload = buildProviderPayloadFromMap(providerMap);
  if(!Object.keys(payload.providers || {}).length) return null;
  return saveApiKeysWithFallback(payload);
}

function setFmpBudgetStats(total, used, source){
  const remaining = total!=null && used!=null ? Math.max(0, total - used) : null;
  const totalEl = document.getElementById('fmpBudgetTotal');
  const usedEl = document.getElementById('fmpBudgetUsed');
  const remEl = document.getElementById('fmpBudgetRemaining');
  const srcEl = document.getElementById('fmpSource');
  if(totalEl) totalEl.textContent = total!=null ? total.toLocaleString() : '—';
  if(usedEl) usedEl.textContent = used!=null ? used.toLocaleString() : '—';
  if(remEl) remEl.textContent = remaining!=null ? remaining.toLocaleString() : '—';
  if(srcEl) srcEl.textContent = source || '—';
}

async function saveFmpKeys(){
  const raw = document.getElementById('fmpKeysInput').value.trim();
  if(!raw){ document.getElementById('fmpStatus').textContent='⚠ No keys entered'; return; }
  const keys = raw.split(/[\n,]+/).map(k=>k.trim()).filter(k=>k.length>5);
  if(keys.length===0){ document.getElementById('fmpStatus').textContent='⚠ Invalid keys'; return; }

  const provider = getSelectedProvider();
  if(provider === 'custom'){
    document.getElementById('fmpStatus').textContent='⚠ Enter a custom provider id';
    return;
  }

  document.getElementById('fmpStatus').textContent='Saving...';
  const payload = { providers: { [provider]: keys } };
  if(provider === 'fmp') payload.fmp_keys = keys;
  if(provider === 'finnhub') payload.finnhub = keys[0];
  if(provider === 'alpha_vantage') payload.alpha_vantage = keys[0];
  if(provider === 'news_api') payload.news_api = keys[0];

  try {
    const d = await saveApiKeysWithFallback(payload);

    const providers = d.sources?.providers || {};
    const totalCnt = getTotalConfiguredKeys(providers);
    const cnt = totalCnt || (providers.fmp?.keys_count ?? d.sources?.fmp_keys_count ?? keys.length);
    setFmpCountText(cnt);
    document.getElementById('fmpStatus').textContent=`✅ ${keys.length} key(s) saved for ${provider}`;
    document.getElementById('fmpKeysInput').value='';
    try { localStorage.setItem(`provider_keys_${provider}`, JSON.stringify(keys)); } catch(e) { /* ignore */ }
    await loadFmpKeyStatus();
  } catch(e){
    document.getElementById('fmpStatus').textContent='⚠ Server offline — provider keys saved locally';
    localStorage.setItem(`provider_keys_${provider}`,JSON.stringify(keys));
    if(provider === 'fmp'){
      setFmpCountText(keys.length, ' (local)');
      setFmpBudgetStats(keys.length * 250, 0, 'Local');
    }
  }
}

function renderProviderStatus(providers){
  const container = document.getElementById('providerStatus');
  if(!container) return;
  const entries = Object.entries(providers || {});
  if(entries.length===0){
    container.innerHTML = '<div class="api-provider-item"><div class="name">No providers</div><div class="val">0 configured</div></div>';
    return;
  }
  container.innerHTML = entries
    .sort((a,b)=>a[0].localeCompare(b[0]))
    .map(([name,meta])=>{
      const n = Number(meta.keys_count || 0);
      return `<div class="api-provider-item"><div class="name">${name}</div><div class="val">${n} key(s)</div></div>`;
    })
    .join('');
}

async function loadFmpKeyStatus(opts={}){
  const allowRestore = opts.allowRestore !== false;
  try {
    const d = await fetchApiKeyStatusWithFallback();
    const src=d.sources||{};
    const providers = src.providers || {};
    renderProviderStatus(providers);
    const totalConfigured = getTotalConfiguredKeys(providers);

    if(totalConfigured<=0 && allowRestore){
      try {
        const restored = await restoreProviderKeysFromLocal();
        if(restored && restored.status === 'ok'){
          await loadFmpKeyStatus({ allowRestore: false });
          return;
        }
      } catch(e){
        // continue with server-reported zero state
      }
    }

    if(totalConfigured > 0){
      setFmpCountText(totalConfigured);
    }
    if(src.fmp){
      const cnt=src.fmp_keys_count||0;
      if(totalConfigured<=0) setFmpCountText(cnt);
      if(src.fmp_status){
        setFmpBudgetStats(src.fmp_status.total_budget, src.fmp_status.total_used, 'Server');
      } else {
        setFmpBudgetStats(src.fmp_daily_budget || cnt * 250, null, 'Server');
      }
      if(src.fmp_status && src.fmp_status.keys){
        let html='<div style="margin-top:6px;">';
        src.fmp_status.keys.forEach(k=>{
          const pct=Math.round((k.calls_used/(k.calls_used+k.calls_remaining))*100)||0;
          const col=k.exhausted?'#ff4444':'var(--green)';
          html+=`<div style="margin:3px 0;display:flex;align-items:center;gap:8px;">
            <span style="color:var(--cyan);">Key #${k.key_index}</span>
            <div style="flex:1;height:4px;background:#1a1a3a;border-radius:2px;overflow:hidden;">
              <div style="width:${pct}%;height:100%;background:${col};"></div>
            </div>
            <span>${k.calls_used}/${k.calls_used+k.calls_remaining}</span>
          </div>`;
        });
        html+=`<div style="margin-top:6px;color:var(--gold);">Total: ${src.fmp_status.total_used}/${src.fmp_status.total_budget} calls used today</div>`;
        html+='</div>';
        document.getElementById('fmpKeyStatus').innerHTML=html;
      } else {
        document.getElementById('fmpKeyStatus').innerHTML='<div style="margin-top:6px;">No key usage telemetry from server.</div>';
      }
    } else {
      if(totalConfigured<=0) setFmpCountText(0);
      setFmpBudgetStats(null, null, 'Server');
      document.getElementById('fmpKeyStatus').innerHTML='<div style="margin-top:6px;">No server keys configured.</div>';
    }
  } catch(_e){
    const localRaw = localStorage.getItem('provider_keys_fmp');
    const localKeys = localRaw ? JSON.parse(localRaw) : [];
    setFmpCountText(localKeys.length, localKeys.length ? ' (local)' : '');
    setFmpBudgetStats(localKeys.length * 250, 0, 'Local');
    renderProviderStatus({ fmp: { keys_count: localKeys.length } });
    document.getElementById('fmpKeyStatus').innerHTML=
      localKeys.length ? '<div style="margin-top:6px;">Using locally saved keys. Server offline.</div>' :
      '<div style="margin-top:6px;">Server offline and no local keys found.</div>';
  }
}

function renderModelStructureMeta(profile){
  const el = document.getElementById('modelStructureMeta');
  if(!el || !profile){
    if(el) el.innerHTML = '';
    return;
  }
  const rows = [
    ['Profile', profile.display_name || profile.profile_id],
    ['Version', profile.version || '—'],
    ['Model Class', profile.model_class || '—'],
    ['Price Window', profile.price_window || '—'],
    ['Price Features', profile.price_features || '—'],
    ['Text Tokens', profile.text_tokens || '—'],
    ['Heads', (profile.output_heads || []).join(', ') || '—'],
    ['Checkpoint', profile.checkpoint_pattern || '—'],
  ];
  el.innerHTML = rows.map(([k,v])=>
    `<div class="model-meta-item"><div class="label">${k}</div><div class="value">${v}</div></div>`
  ).join('');
}

function loadModelProfiles(){
  const select = document.getElementById('modelProfileSelect');
  const status = document.getElementById('modelProfileStatus');
  if(!select || !status) return;

  fetch(`${API_BASE.replace('/api','')}/api/system/model-profiles`)
    .then(r=>r.json())
    .then(d=>{
      const mr = d.model_registry || {};
      const profiles = mr.profiles || [];
      const active = mr.active_profile_id;
      state.modelRegistry = mr;
      select.innerHTML = profiles.map(p=>
        `<option value="${p.profile_id}" ${p.profile_id===active?'selected':''}>${p.display_name} (${p.profile_id})</option>`
      ).join('');
      status.textContent = active ? `Active profile: ${active}` : 'Profiles loaded';
      renderModelStructureMeta(mr.active_profile || profiles[0]);
    })
    .catch(()=>{
      status.textContent = 'Model profile API unavailable';
    });
}

function saveModelProfile(){
  const select = document.getElementById('modelProfileSelect');
  const status = document.getElementById('modelProfileStatus');
  if(!select || !status) return;
  const profileId = select.value;
  if(!profileId){
    status.textContent = 'Select a profile first';
    return;
  }
  status.textContent = 'Updating profile...';
  fetch(`${API_BASE.replace('/api','')}/api/system/model-profiles/active`, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ profile_id: profileId }),
  }).then(r=>r.json()).then(d=>{
    if(d.status === 'ok'){
      status.textContent = `✅ Active profile set: ${profileId}`;
      const mr = d.model_registry || {};
      state.modelRegistry = mr;
      renderModelStructureMeta(mr.active_profile || null);
    } else {
      status.textContent = '⚠ Failed to update profile';
    }
  }).catch(()=>{
    status.textContent = '⚠ Failed to update profile';
  });
}

// Load FMP status on page load
document.addEventListener('DOMContentLoaded',()=>{
  const providerSelect = document.getElementById('providerSelect');
  if(providerSelect) providerSelect.addEventListener('change', toggleCustomProviderInput);
  const analyzeBtn = document.getElementById('analyzeBtn');
  if(analyzeBtn) state.analyzeBtnDefaultHtml = analyzeBtn.innerHTML;
  updateApiKeysFabVisibility(1);
  toggleCustomProviderInput();
  // Resolve the correct API backend before first key/status loads.
  ensureApiBaseReady().then(()=>{
    setTimeout(loadGrafanaEmbedConfig, 250);
    setTimeout(loadFmpKeyStatus, 450);
    setTimeout(loadModelProfiles, 650);
    setTimeout(startTickerStripLive, 350);
  }).catch(()=>{
    setTimeout(loadGrafanaEmbedConfig, 450);
    setTimeout(loadFmpKeyStatus, 1000);
    setTimeout(loadModelProfiles, 1200);
    setTimeout(startTickerStripLive, 600);
  });
  setTimeout(restoreAppState, 250);
});

/* ═══════════════════════════════════════════
   SCREEN 2 — MARKET SELECTOR
   ═══════════════════════════════════════════ */
function toggleMarket(el,market){
  el.classList.toggle('selected');
  const idx=state.selectedMarkets.indexOf(market);
  if(idx>=0) state.selectedMarkets.splice(idx,1);
  else state.selectedMarkets.push(market);
  refreshTickerStrip();
  scheduleAutoAnalysis();
}

/* ═══════════════════════════════════════════
   SCREEN 3 — STOCK SEARCH + SELECT
   ═══════════════════════════════════════════ */
function populateStockList(){
  const stocks=getAllStocks();
  // trending tags
  document.getElementById('trendingTags').innerHTML=stocks.slice(0,8).map(s=>{
    const cls=s.change>=0?'up':'down';
    return `<div class="trending-tag" onclick="addStock('${s.ticker}')"><span class="sym">${s.ticker}</span> <span class="${cls}">${s.change>=0?'+':''}${s.change}%</span></div>`;
  }).join('');
  renderStockList(stocks);
}

function renderStockList(stocks){
  document.getElementById('stockList').innerHTML=stocks.map(s=>{
    const cls=s.change>=0?'up':'down';
    const sign=s.change>=0?'▲ +':'▼ ';
    const sel=state.selectedStocks.includes(s.ticker);
    return `<div class="stock-row" ondblclick="addStock('${s.ticker}')">
      <div class="info"><span class="ticker">${s.ticker}</span><span class="company">${s.name}</span></div>
      <span class="price ${cls}">${fmt$(s.price)} <small>${sign}${Math.abs(s.change)}%</small></span>
      <button class="add-btn ${sel?'added':''}" onclick="event.stopPropagation();${sel?`removeStock('${s.ticker}')`:`addStock('${s.ticker}')`}">${sel?'✓ ADDED':'+ ADD'}</button>
    </div>`;
  }).join('');
}

function filterStocks(){
  const q=document.getElementById('stockSearch').value.toUpperCase();
  renderStockList(getAllStocks().filter(s=>s.ticker.includes(q)||s.name.toUpperCase().includes(q)));
}

function filterSector(btn,sector){
  document.querySelectorAll('.sector-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  let stocks=getAllStocks();
  if(sector!=='all') stocks=stocks.filter(s=>s.sector===sector);
  renderStockList(stocks);
}

function addStock(ticker){
  if(!state.selectedStocks.includes(ticker)) state.selectedStocks.push(ticker);
  renderSelectedStocks(); renderStockList(getAllStocks());
  scheduleAutoAnalysis();
}

function removeStock(ticker){
  state.selectedStocks=state.selectedStocks.filter(t=>t!==ticker);
  renderSelectedStocks(); renderStockList(getAllStocks());
  scheduleAutoAnalysis();
}

function renderSelectedStocks(){
  document.getElementById('selectedStocks').innerHTML=state.selectedStocks.map(t=>
    `<div class="selected-tag">${t}<span class="remove" onclick="removeStock('${t}')">✕</span></div>`
  ).join('');
  // Enable/disable analyze button
  const btn=document.getElementById('analyzeBtn');
  if(state.selectedStocks.length===0 || state.isAutoAnalyzing) btn.classList.add('disabled');
  else btn.classList.remove('disabled');
}

function aiSuggest(){
  const stocks=getAllStocks().sort((a,b)=>b.change-a.change).slice(0,4);
  stocks.forEach(s=>addStock(s.ticker));
}

function findMarketForTicker(ticker){
  if(!ticker) return null;
  const target = String(ticker).toUpperCase();
  const markets = state.selectedMarkets.length ? state.selectedMarkets : Object.keys(STOCK_DB || {});
  for(const market of markets){
    const rows = STOCK_DB?.[market] || [];
    if(rows.some(s=>String(s.ticker || '').toUpperCase()===target)) return market;
  }
  for(const market of Object.keys(STOCK_DB || {})){
    const rows = STOCK_DB?.[market] || [];
    if(rows.some(s=>String(s.ticker || '').toUpperCase()===target)) return market;
  }
  return null;
}

function resolveTickerMarket(ticker){
  const t = String(ticker || '').toUpperCase().trim();
  if(!t) return state.selectedMarkets[0]||'SP500';

  const mapped = state.analysisResults?.ticker_markets?.[t];
  if(mapped) return String(mapped).toUpperCase();

  const fromSelection = findMarketForTicker(t);
  if(fromSelection) return fromSelection;

  return state.selectedMarkets[0]||'SP500';
}

/* ═══════════════════════════════════════════
   ANALYSIS PIPELINE (Auto + Manual)
   ═══════════════════════════════════════════ */
function getAnalysisPayload(){
  state.capital=parseInt(document.getElementById('capitalInput').value.replace(/[^0-9]/g,''))||30000;
  const tickerMarkets = {};
  state.selectedStocks.forEach(t=>{
    const market = findMarketForTicker(t);
    if(market) tickerMarkets[t] = market;
  });

  return {
    tickers:[...state.selectedStocks],
    market:state.selectedMarkets[0]||'SP500',
    markets:[...state.selectedMarkets],
    ticker_markets:tickerMarkets,
    investment_amount:state.capital,
    risk_tolerance:state.riskTolerance,
    currency:state.currency,
    horizon:state.horizon,
    force_refresh:false,
  };
}

function getAnalysisSignature(payload){
  const tickers=[...(payload.tickers||[])].map(t=>String(t).toUpperCase()).sort();
  const markets=[...(payload.markets||[])]
    .map(m=>String(m).toUpperCase().trim())
    .filter(Boolean)
    .sort();
  const tickerMarkets = Object.fromEntries(
    Object.entries(payload.ticker_markets || {})
      .map(([k,v])=>[String(k).toUpperCase().trim(), String(v).toUpperCase().trim()])
      .filter(([k,v])=>!!k && !!v)
      .sort((a,b)=>a[0].localeCompare(b[0]))
  );

  return JSON.stringify({
    tickers,
    market:String(payload.market||'').toUpperCase(),
    markets,
    ticker_markets:tickerMarkets,
    capital:Number(payload.investment_amount||0),
    risk:Number(payload.risk_tolerance||0),
    currency:String(payload.currency||'').toUpperCase(),
    horizon:String(payload.horizon||'').toUpperCase(),
  });
}

function setAnalyzeButtonState({busy=false,text=''}={}){
  const btn=document.getElementById('analyzeBtn');
  if(!btn) return;

  if(busy){
    state.isAutoAnalyzing=true;
    btn.classList.add('disabled');
    btn.innerHTML=text||'⚡ ANALYZING...';
    return;
  }

  state.isAutoAnalyzing=false;
  btn.innerHTML=state.analyzeBtnDefaultHtml||'⚡ TRAIN &amp; ANALYZE →';
  if(state.selectedStocks.length===0) btn.classList.add('disabled');
  else btn.classList.remove('disabled');
}

function scheduleAutoAnalysis(){
  if(state.analyzeInFlight) return;

  if(state.autoAnalyzeTimer){
    clearTimeout(state.autoAnalyzeTimer);
    state.autoAnalyzeTimer=null;
  }

  if(!state.selectedStocks.length) return;

  state.autoAnalyzeTimer=setTimeout(()=>{
    runAnalysis({auto:true,navigate:false});
  }, 650);
}

async function runAnalysis(opts={}){
  const auto=!!opts.auto;
  const navigate=opts.navigate!==false;
  const timeoutMs = getAnalyzeTimeoutMs(auto);

  if(state.analyzeInFlight){
    if(auto) return;
    if(state.analyzeInFlightMode==='manual') return;
  }

  if(!state.selectedStocks.length){
    if(!auto) alert('Please select at least one stock.');
    return;
  }

  const payload=getAnalysisPayload();
  payload.force_refresh = !auto;
  const signature=getAnalysisSignature(payload);

  if(auto){
    const autoWarmAgeMs = Date.now() - Number(state.lastAnalysisAtMs || 0);
    if(state.lastAnalysisSignature===signature && autoWarmAgeMs<7000){
      return;
    }
  }

  if(state.autoAnalyzeAbortController){
    try { state.autoAnalyzeAbortController.abort(); } catch(e) { /* ignore */ }
    state.autoAnalyzeAbortController=null;
  }

  if(!auto && state.autoAnalyzeTimer){
    clearTimeout(state.autoAnalyzeTimer);
    state.autoAnalyzeTimer=null;
  }

  const controller=new AbortController();
  state.autoAnalyzeAbortController=controller;
  state.analyzeInFlight=true;
  state.analyzeInFlightMode=auto ? 'auto' : 'manual';
  let analyzeTimedOut=false;
  const timeoutId=setTimeout(()=>{
    analyzeTimedOut=true;
    try { controller.abort(); } catch(e) { /* ignore */ }
  }, timeoutMs);
  const requestId=++state.analysisRequestCounter;
  state.latestAnalysisRequestId=requestId;

  if(auto){
    setAnalyzeButtonState({busy:true,text:'⚡ ANALYZING...'});
  } else {
    setAnalyzeButtonState({busy:true,text:'⚡ ANALYZING...'});
    showLoading();
  }

  try {
    const out = await requestAnalysisWithFallback(payload, controller.signal);
    enrichSignalsWithAnalysisDetails(out);
    if(requestId!==state.latestAnalysisRequestId) return;

    state.analysisResults=out;
    state.lastAnalysisSignature=signature;
    state.lastAnalysisAtMs=Date.now();
    persistAppState();

    if(!auto && navigate){
      await sleep(150);
      goToScreen(4);
    }

    // Keep UI snappy: enrich live details in background after primary analysis is ready.
    enrichWithLivePredictions().then(()=>{
      const screen4 = document.getElementById('screen4');
      if(screen4 && screen4.classList.contains('active')){
        renderDashboard();
      }
    }).catch(()=>{});
  } catch(e) {
    const errMsg = String(e?.message || 'Unknown error');
    const abortLike = !!(e && (e.name==='AbortError' || e.code==='ABORT_ERR'));
    const timeoutLike = analyzeTimedOut || /analyze request timeout|timed out|timeout/i.test(errMsg);

    if(abortLike || timeoutLike){
      if(!auto){
        if(state.analysisResults?.signals?.length){
          if(navigate) goToScreen(4);
        } else {
          if(navigate) goToScreen(3);
          const sec = Math.max(1, Math.round(timeoutMs / 1000));
          alert(
            `Analysis timed out after ${sec}s.\n\n` +
            'Try reducing selected tickers or retrying once data providers are warm.'
          );
        }
      }
      return;
    }

    console.log('[FINSENT] Analysis failed:', errMsg);

    if(!auto){
      const networkFailure = (e?.name === 'TypeError') || /failed to fetch/i.test(errMsg);
      const endpointMismatch = /HTTP\s+(404|405)/i.test(errMsg);
      if(networkFailure || endpointMismatch){
        const backendUp = await isBackendReachable();
        if(!backendUp){
          if(navigate) goToScreen(3);
          alert(
            `Analysis failed: FINSENT backend was not found at ${API_BASE}.\n\n` +
            'Either the backend is not running, or a different API service is using that port (Docker commonly binds 8000).\n\n' +
            'Start backend on a free port:\n' +
            'python -m uvicorn finsentnet_pro.backend.api.main:app --host 127.0.0.1 --port 8001\n\n' +
            'If needed, open frontend with explicit API base:\n' +
            '.../index.html?api_base=http://127.0.0.1:8001/api'
          );
          return;
        }
      }

      if(state.analysisResults?.signals?.length){
        if(navigate) goToScreen(4);
      } else {
        if(navigate) goToScreen(3);
        alert(`Analysis failed: ${errMsg}`);
      }
    }
  } finally {
    clearTimeout(timeoutId);
    if(!auto) hideLoading();
    setAnalyzeButtonState({busy:false});
    state.analyzeInFlight=false;
    state.analyzeInFlightMode=null;

    if(state.autoAnalyzeAbortController===controller){
      state.autoAnalyzeAbortController=null;
    }
  }
}

async function trainModel(ticker, market){
  showTrainingOverlay(ticker);
  try {
    // Start training
    await fetch(`${API_BASE}/train/start`,{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        ticker:ticker, market:market,
        epochs:50, period:'5y',
      }),
    });

    // Poll for progress
    let done=false;
    const pollStart=Date.now();
    const maxPollMs=30*60*1000;
    let emptyProgressCount=0;
    while(!done){
      if(Date.now()-pollStart>maxPollMs){
        console.warn(`Training poll timeout for ${ticker}`);
        break;
      }
      await sleep(1500);
      try {
        const res=await fetch(`${API_BASE}/train/status/${ticker}`);
        const data=await res.json();
        const progress=data.progress;
        if(!progress){
          emptyProgressCount++;
          if(emptyProgressCount>=40){
            console.warn(`No training progress updates for ${ticker}`);
            break;
          }
          continue;
        }

        emptyProgressCount=0;

        updateTrainingOverlay(progress);

        if(progress.status==='completed'){
          state.trainedTickers[ticker]=true;
          done=true;
        } else if(progress.status==='failed'){
          console.warn(`Training failed for ${ticker}: ${progress.message}`);
          done=true;
        }
      } catch(e){ /* keep polling */ }
    }
  } catch(e) {
    console.log(`[FINSENT] Training unavailable for ${ticker}:`, e.message);
  }
  hideTrainingOverlay();
}

async function enrichWithLivePredictions(){
  if(!state.analysisResults||!state.analysisResults.signals) return;
  const tasks = state.analysisResults.signals.map(async (sig)=>{
    if(!state.trainedTickers[sig.ticker]) return;
    const market = resolveTickerMarket(sig.ticker);
    try {
      const res=await fetch(`${API_BASE}/live/predict/${sig.ticker}?market=${market}&capital=${state.capital}&risk_tolerance=${state.riskTolerance}`);
      if(!res.ok) return;
      const pred=await res.json();
      if(pred.status!=='success') return;
      // Merge live prediction data into the signal
      sig.direction=pred.prediction.direction;
      sig.confidence=pred.prediction.confidence;
      sig.predicted_return=pred.prediction.predicted_return;
      sig.entry_price=pred.signal.entry_price;
      sig.target_price=pred.signal.target_price;
      sig.stop_loss=pred.signal.stop_loss;
      sig.risk_reward=pred.signal.risk_reward;
      sig.capital_required=pred.signal.capital_required;
      sig.quantity=pred.signal.quantity;
      sig.time_horizon=pred.signal.time_horizon;
      sig.sentiment_score=pred.analysis.sentiment_score;
      sig.technical_score=pred.analysis.technical_score;
      sig.regime=pred.analysis.regime;
      sig.reasoning=pred.analysis.reasoning;
      sig._live=true;
    } catch(e){ /* use existing signal */ }
  });

  await Promise.allSettled(tasks);
}

/* ═══════════════════════════════════════════
   TRAINING OVERLAY UI
   ═══════════════════════════════════════════ */
function showTrainingOverlay(ticker){
  document.getElementById('trainingOverlay').classList.add('active');
  document.getElementById('trainTicker').textContent=ticker;
  document.getElementById('trainStatus').textContent='Initializing neural network…';
  document.getElementById('trainPct').textContent='0%';
  document.getElementById('trainProgressFill').style.width='0%';
  document.getElementById('trainEpoch').textContent='—';
  document.getElementById('trainLoss').textContent='—';
  document.getElementById('trainValLoss').textContent='—';
  document.getElementById('trainAcc').textContent='—';
  // Reset loss chart
  const chartEl=document.getElementById('trainLossChart');
  if(chartEl && typeof Plotly!=='undefined') Plotly.purge(chartEl);
  state.trainLossChart=null;
}

function updateTrainingOverlay(p){
  document.getElementById('trainTicker').textContent=p.ticker||'—';
  document.getElementById('trainStatus').textContent=p.message||p.status;
  const pct=p.progress_pct||0;
  document.getElementById('trainPct').textContent=pct.toFixed(0)+'%';
  document.getElementById('trainProgressFill').style.width=pct+'%';
  document.getElementById('trainEpoch').textContent=
    p.current_epoch?`${p.current_epoch}/${p.total_epochs}`:'—';
  document.getElementById('trainLoss').textContent=
    p.train_loss?p.train_loss.toFixed(4):'—';
  document.getElementById('trainValLoss').textContent=
    p.val_loss?p.val_loss.toFixed(4):'—';
  document.getElementById('trainAcc').textContent=
    p.val_accuracy?p.val_accuracy.toFixed(1)+'%':'—';

  // Draw mini loss chart
  if(p.history&&p.history.train_loss&&p.history.train_loss.length>1){
    drawTrainLossChart(p.history);
  }
}

function drawTrainLossChart(history){
  const chartEl=document.getElementById('trainLossChart');
  if(!chartEl||typeof Plotly==='undefined') return;

  const labels=(history.train_loss||[]).map((_,i)=>i+1);
  const traces=[
    {
      x:labels,
      y:history.train_loss||[],
      type:'scatter',
      mode:'lines',
      name:'Train',
      line:{color:'#00F5FF', width:2},
      hovertemplate:'Epoch %{x}<br>Train %{y:.5f}<extra></extra>',
    },
    {
      x:labels,
      y:history.val_loss||[],
      type:'scatter',
      mode:'lines',
      name:'Val',
      line:{color:'#FFD700', width:2},
      hovertemplate:'Epoch %{x}<br>Val %{y:.5f}<extra></extra>',
    },
  ];

  const layout={
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,245,255,0.02)',
    margin:{l:26,r:8,t:16,b:20},
    showlegend:true,
    legend:{orientation:'h',x:0,y:1.18,font:{size:9,color:'#8FA2B8'}},
    xaxis:{
      showgrid:false,
      showticklabels:false,
      zeroline:false,
      color:'#667788',
    },
    yaxis:{
      showgrid:true,
      gridcolor:'rgba(0,245,255,0.06)',
      zeroline:false,
      color:'#667788',
      tickfont:{size:8},
    },
    font:{family:'JetBrains Mono',color:'#8FA2B8'},
  };

  Plotly.react(chartEl, traces, layout, {
    responsive:true,
    displayModeBar:false,
    staticPlot:false,
  });
  state.trainLossChart=chartEl;
}

function hideTrainingOverlay(){
  document.getElementById('trainingOverlay').classList.remove('active');
  const chartEl=document.getElementById('trainLossChart');
  if(chartEl&&typeof Plotly!=='undefined') Plotly.purge(chartEl);
  state.trainLossChart=null;
}

function getTunnelWsUrl(){
  try {
    const httpUrl = new URL(API_BASE, window.location.href);
    const proto = httpUrl.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${httpUrl.host}/ws`;
  } catch(e){
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}/ws`;
  }
}

function sendTunnelMessage(payload){
  if(!state.liveWs || state.liveWs.readyState!==WebSocket.OPEN) return false;
  try {
    state.liveWs.send(JSON.stringify(payload));
    return true;
  } catch(e){
    return false;
  }
}

function closeTunnelSocket(){
  if(state.liveWsReconnectTimer){
    clearTimeout(state.liveWsReconnectTimer);
    state.liveWsReconnectTimer=null;
  }
  if(state.liveWs){
    try { state.liveWs.close(); } catch(e) { /* ignore */ }
  }
  state.liveWs=null;
  state.liveWsConnected=false;
  state.liveWsReconnectDelayMs=1500;
  state.liveTunnelSubscribedTicker=null;
}

function connectTunnelSocket(){
  if(state.liveWs && (state.liveWs.readyState===WebSocket.OPEN || state.liveWs.readyState===WebSocket.CONNECTING)){
    return;
  }

  const wsUrl=getTunnelWsUrl();
  try {
    const ws = new WebSocket(wsUrl);
    state.liveWs=ws;

    ws.onopen = ()=>{
      state.liveWsConnected=true;
      state.liveWsReconnectDelayMs=1500;
      sendTunnelMessage({action:'status'});
      subscribeCurrentTickerToTunnels();
    };

    ws.onmessage = (event)=>{
      handleTunnelSocketMessage(event.data);
    };

    ws.onclose = ()=>{
      state.liveWsConnected=false;
      state.liveTunnelSubscribedTicker=null;
      if(document.getElementById('screen4')?.classList.contains('active')){
        if(state.liveWsReconnectTimer) clearTimeout(state.liveWsReconnectTimer);
        const reconnectDelay = Math.min(
          Math.max(1500, Number(state.liveWsReconnectDelayMs || 1500)),
          15000
        );
        state.liveWsReconnectTimer=setTimeout(()=>{
          connectTunnelSocket();
        }, reconnectDelay);
        state.liveWsReconnectDelayMs = Math.min(Math.floor(reconnectDelay * 1.6), 15000);
      }
    };

    ws.onerror = ()=>{
      state.liveWsConnected=false;
    };
  } catch(e){
    console.log('[FINSENT] Tunnel websocket connect failed:', e.message);
  }
}

function subscribeCurrentTickerToTunnels(){
  const data=state.analysisResults;
  if(!data||!data.signals||!data.signals.length) return;
  if(!state.liveWsConnected) return;

  const sig=data.signals[state.currentStockIdx];
  if(!sig) return;

  const ticker=String(sig.ticker||'').toUpperCase();
  if(!ticker) return;

  if(state.liveTunnelSubscribedTicker && state.liveTunnelSubscribedTicker!==ticker){
    sendTunnelMessage({
      action:'unsubscribe',
      tickers:[state.liveTunnelSubscribedTicker],
    });
  }

  sendTunnelMessage({
    action:'subscribe',
    tickers:[ticker],
    market:resolveTickerMarket(ticker),
    capital:state.capital,
    risk_tolerance:state.riskTolerance,
    enable_train_tunnel:false,
  });

  state.liveTunnelSubscribedTicker=ticker;
}

function mergePredictTunnelIntoState(tunnelData){
  const data=state.analysisResults;
  if(!data||!Array.isArray(data.signals)) return;

  const ticker=String(tunnelData?.ticker||'').toUpperCase();
  if(!ticker) return;

  const idx=data.signals.findIndex(s=>String(s.ticker||'').toUpperCase()===ticker);
  if(idx<0) return;

  const sig=data.signals[idx];
  const p=tunnelData.prediction||{};
  const s=tunnelData.signal||{};
  const a=tunnelData.analysis||{};

  if(typeof p.direction==='string' && p.direction) sig.direction=p.direction;
  if(Number.isFinite(Number(p.confidence))) sig.confidence=Number(p.confidence);
  if(Number.isFinite(Number(p.predicted_return))) sig.predicted_return=Number(p.predicted_return);

  if(Number.isFinite(Number(s.entry_price))) sig.entry_price=Number(s.entry_price);
  if(Number.isFinite(Number(s.target_price))) sig.target_price=Number(s.target_price);
  if(Number.isFinite(Number(s.stop_loss))) sig.stop_loss=Number(s.stop_loss);
  if(Number.isFinite(Number(s.risk_reward))) sig.risk_reward=Number(s.risk_reward);
  if(Number.isFinite(Number(s.capital_required))) sig.capital_required=Number(s.capital_required);
  if(Number.isFinite(Number(s.quantity))) sig.quantity=Number(s.quantity);
  if(typeof s.time_horizon==='string' && s.time_horizon) sig.time_horizon=s.time_horizon;

  if(Number.isFinite(Number(a.sentiment_score))) sig.sentiment_score=Number(a.sentiment_score);
  if(Number.isFinite(Number(a.technical_score))) sig.technical_score=Number(a.technical_score);
  if(typeof a.regime==='string' && a.regime) sig.regime=a.regime;
  if(Array.isArray(a.reasoning) && a.reasoning.length) sig.reasoning=a.reasoning;

  sig._live=true;
  state.trainedTickers[ticker]=true;

  if(idx===state.currentStockIdx){
    renderStockView(idx);
  }
}

function applyLiveTickToChart(sig, quote){
  if(!sig || !quote) return;

  if(state.grafanaEmbedReady){
    const refreshed = refreshGrafanaEmbeddedPanels(sig, false);
    if(!refreshed){
      Promise.resolve(renderGrafanaEmbeddedPanels(sig)).catch((e)=>{
        console.log('[FINSENT] Embed chart refresh skipped:', e.message);
      });
    }
    return;
  }

  if(state.activeChartTicker!==sig.ticker) return;
  if(!state.liveChartData || !Array.isArray(state.liveChartData.candles)) return;

  const price = Number(quote.price);
  if(!Number.isFinite(price)) return;

  const candles = state.liveChartData.candles;
  if(!candles.length) return;

  const tfSec = getTimeframeBucketSeconds(state.currentTimeframe || '1D');
  const tickEpoch = toEpochSeconds(quote.timestamp || Date.now());
  const bucketEpoch = Math.floor(tickEpoch / tfSec) * tfSec;

  const last = candles[candles.length - 1];
  if(bucketEpoch === last.time){
    last.high = Math.max(Number(last.high || price), price);
    last.low = Math.min(Number(last.low || price), price);
    last.close = price;
    if(Number.isFinite(Number(quote.volume))){
      last.volume = Math.max(Number(last.volume || 0), Number(quote.volume));
    }
  } else if(bucketEpoch > last.time){
    candles.push({
      time: bucketEpoch,
      open: Number(last.close || price),
      high: price,
      low: price,
      close: price,
      volume: Number.isFinite(Number(quote.volume)) ? Number(quote.volume) : 0,
    });
    if(candles.length > 1600) candles.shift();
  } else {
    return;
  }

  if(Array.isArray(state.liveChartData.volume)){
    const vol = state.liveChartData.volume;
    const activeCandle = candles[candles.length - 1];
    const volColor = activeCandle.close >= activeCandle.open ? GRAFANA_THEME.volumeUp : GRAFANA_THEME.volumeDown;
    const volValue = Number.isFinite(Number(quote.volume)) ? Number(quote.volume) : Number(activeCandle.volume || 0);

    if(vol.length && Number(vol[vol.length - 1].time) === Number(activeCandle.time)){
      vol[vol.length - 1].value = volValue;
      vol[vol.length - 1].color = volColor;
    } else {
      vol.push({ time: activeCandle.time, value: volValue, color: volColor });
      if(vol.length > 1600) vol.shift();
    }
  }

  const now = Date.now();
  if(now - state.lastWsChartRefreshAt < 1200) return;
  state.lastWsChartRefreshAt = now;
  Promise.resolve(applyGrafanaChart(sig, state.liveChartData)).catch((e)=>{
    console.log('[FINSENT] Tick chart update skipped:', e.message);
  });
}

function handleTunnelSocketMessage(raw){
  let msg=null;
  try { msg=JSON.parse(raw); } catch(e) { return; }
  if(!msg||!msg.type) return;

  if(msg.type==='live_tunnel' || msg.type==='price_update'){
    const quote=msg.data||{};
    const ticker=String(quote.ticker||'').toUpperCase();
    if(!ticker) return;

    const data=state.analysisResults;
    if(!data||!Array.isArray(data.signals)) return;
    const idx=data.signals.findIndex(s=>String(s.ticker||'').toUpperCase()===ticker);
    if(idx<0) return;

    const sig=data.signals[idx];
    if(Number.isFinite(Number(quote.price))) sig.entry_price=Number(quote.price);
    sig._live=true;

    if(idx===state.currentStockIdx && Number.isFinite(Number(quote.price))){
      document.getElementById('dashPrice').textContent=fmt$(quote.price);
      const changePct=Number(quote.change_pct||0);
      const up=changePct>=0;
      document.getElementById('dashChange').innerHTML=
        `<span style="color:${up?'var(--signal-buy)':'var(--signal-sell)'}">${up?'▲':'▼'} ${up?'+':''}${changePct.toFixed(2)}% (${quote.source||'tunnel'})</span>`;

      const liveEl=document.getElementById('liveIndicator');
      if(liveEl) liveEl.style.display='inline-flex';

      applyLiveTickToChart(sig, quote);
    }
    return;
  }

  if(msg.type==='predict_tunnel' || msg.type==='signal_alert'){
    mergePredictTunnelIntoState(msg.data||{});
    return;
  }
}

/* ═══════════════════════════════════════════
   LIVE DATA UPDATES
   ═══════════════════════════════════════════ */
function startLiveUpdates(){
  // Stop any existing timers
  stopLiveUpdates({closeSocket:false});

  // Ensure embed config is loaded before chart rendering path decides mode
  ensureGrafanaEmbedConfig();

  // Show live indicator
  const liveEl=document.getElementById('liveIndicator');
  if(liveEl) liveEl.style.display='inline-flex';

  // Price polling every 30s
  state.livePriceTimer=setInterval(()=>{
    if(!state.liveWsConnected) updateLivePrices();
  }, 30000);

  const pollByTf={ '1m':12000, '5m':15000, '15m':18000, '1H':25000, '4H':30000, '1D':45000 };
  state.liveChartPollMs=pollByTf[state.currentTimeframe]||15000;

  // Chart polling for live candles
  state.liveChartTimer=setInterval(()=>{
    const staleWs = !state.liveWsConnected || (Date.now()-state.lastWsChartRefreshAt > state.liveChartPollMs * 1.25);
    if(staleWs) updateLiveChartFromBackend();
  }, state.liveChartPollMs);

  // Initial refresh
  updateLivePrices();
  updateLiveChartFromBackend();

  // Dual tunnel websocket stream
  connectTunnelSocket();
  subscribeCurrentTickerToTunnels();

  // Fetch news for current stock
  fetchLiveNews();
}

function stopLiveUpdates(opts={}){
  const closeSocket=!!opts.closeSocket;
  if(state.livePriceTimer){ clearInterval(state.livePriceTimer); state.livePriceTimer=null; }
  if(state.liveRefreshTimer){ clearInterval(state.liveRefreshTimer); state.liveRefreshTimer=null; }
  if(state.liveChartTimer){ clearInterval(state.liveChartTimer); state.liveChartTimer=null; }
  if(closeSocket) closeTunnelSocket();
}

async function updateLivePrices(){
  const data=state.analysisResults;
  if(!data||!data.signals) return;
  const sig=data.signals[state.currentStockIdx];
  if(!sig) return;
  const market=resolveTickerMarket(sig.ticker);

  try {
    await ensureApiBaseReady();
    const res=await fetch(`${API_BASE}/live/quote/${sig.ticker}?market=${market}`);
    if(!res.ok) return;
    const quote=await res.json();
    if(quote.price){
      sig.entry_price=Number(quote.price);
      sig._live=true;
      document.getElementById('dashPrice').textContent=fmt$(quote.price);
      const up=quote.change_pct>=0;
      document.getElementById('dashChange').innerHTML=
        `<span style="color:${up?'var(--signal-buy)':'var(--signal-sell)'}">${up?'▲':'▼'} ${up?'+':''}${quote.change_pct}% (${quote.source||'live'})</span>`;
    }
  } catch(e){ /* silent */ }
}

async function refreshPrediction(){
  const data=state.analysisResults;
  if(!data||!data.signals) return;
  const sig=data.signals[state.currentStockIdx];
  if(!sig) return;
  const market=resolveTickerMarket(sig.ticker);

  if(!state.trainedTickers[sig.ticker]){
    console.log('Model not trained for', sig.ticker, '— showing demo data');
    return;
  }

  try {
    await ensureApiBaseReady();
    const res=await fetch(`${API_BASE}/live/predict/${sig.ticker}?market=${market}&capital=${state.capital}&risk_tolerance=${state.riskTolerance}`);
    if(!res.ok) return;
    const pred=await res.json();
    if(pred.status!=='success') return;

    sig.direction=pred.prediction.direction;
    sig.confidence=pred.prediction.confidence;
    sig.predicted_return=pred.prediction.predicted_return;
    sig.entry_price=pred.signal.entry_price;
    sig.target_price=pred.signal.target_price;
    sig.stop_loss=pred.signal.stop_loss;
    sig.risk_reward=pred.signal.risk_reward;
    sig._live=true;

    renderStockView(state.currentStockIdx);
  } catch(e){ console.log('Refresh failed:', e.message); }
}

async function fetchLiveNews(){
  const data=state.analysisResults;
  if(!data||!data.signals) return;
  const sig=data.signals[state.currentStockIdx];
  if(!sig) return;
  const market=resolveTickerMarket(sig.ticker);
  const panel=document.getElementById('newsPanel');
  const list=document.getElementById('newsList');
  if(panel) panel.style.display='block';

  const renderArticles=(articles)=>{
    if(!list) return;
    if(Array.isArray(articles)&&articles.length){
      list.innerHTML=articles.slice(0,6).map(a=>
        `<div class="news-item">
          <div class="news-title">${a.title||'Untitled'}</div>
          <div class="news-meta">${a.source||''}${a.published_at?' · '+new Date(a.published_at).toLocaleString():''}</div>
        </div>`
      ).join('');
      return;
    }
    list.innerHTML='<div class="news-item"><div class="news-title">No recent headlines found for this ticker.</div><div class="news-meta">Try another market or ticker.</div></div>';
  };

  const fallbackNews=Array.isArray(sig._analysisLiveNews) ? sig._analysisLiveNews : [];
  const fallbackFromDetails=(Array.isArray(data.stock_details)
    ? data.stock_details.find(d=>String(d?.ticker||'').toUpperCase()===String(sig.ticker||'').toUpperCase())
    : null);
  const fallbackArticles=fallbackNews.length
    ? fallbackNews
    : (Array.isArray(fallbackFromDetails?.top_live_news) ? fallbackFromDetails.top_live_news : []);

  try {
    await ensureApiBaseReady();
    const res=await fetch(`${API_BASE}/live/news/${sig.ticker}?market=${market}`);
    if(!res.ok){
      if(fallbackArticles.length){
        sig._analysisLiveNews = fallbackArticles.slice(0, 10);
        renderArticles(fallbackArticles);
      } else if(list){
        list.innerHTML='<div class="news-item"><div class="news-title">Live news unavailable right now.</div><div class="news-meta">Please retry in a few seconds.</div></div>';
      }
      return;
    }
    const newsData=await res.json();
    if(Array.isArray(newsData.articles)&&newsData.articles.length){
      sig._analysisLiveNews = newsData.articles.slice(0, 10);
      renderArticles(newsData.articles);
      return;
    }
    renderArticles(fallbackArticles);
  } catch(e){
    if(fallbackArticles.length){
      sig._analysisLiveNews = fallbackArticles.slice(0, 10);
      renderArticles(fallbackArticles);
      return;
    }
    if(list) list.innerHTML='<div class="news-item"><div class="news-title">News fetch failed.</div><div class="news-meta">Check API keys or network connectivity.</div></div>';
  }
}

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

/* ═══════════════════════════════════════════
   DEMO SIGNAL GENERATOR
   ═══════════════════════════════════════════ */
function buildDemoResult(){
  const signals = state.selectedStocks.map(t=>demoSignal(t));
  const deployed = signals.reduce((s,sig)=>s+sig.capital_required,0);
  return {
    status:'success',
    signals,
    portfolio:{
      allocation:{
        total_deployed:+deployed.toFixed(2),
        cash_remaining:+(state.capital-deployed).toFixed(2),
        utilization:+((deployed/state.capital)*100).toFixed(1),
      }
    },
    risk:{
      sharpe_ratio:  +(1.1+Math.random()*1.8).toFixed(2),
      sortino_ratio: +(1.4+Math.random()*1.6).toFixed(2),
      calmar_ratio:  +(1.2+Math.random()*1.5).toFixed(2),
      max_drawdown:  +(-3-Math.random()*9).toFixed(1),
      var_95:        +(2+Math.random()*5).toFixed(1),
      annualized_return:    +(8+Math.random()*18).toFixed(1),
      annualized_volatility:+(7+Math.random()*10).toFixed(1),
      win_rate: +(52+Math.random()*18).toFixed(1),
    },
  };
}

function demoSignal(ticker){
  const rng=mkRng(hashStr(ticker)+7);
  const pUp = 0.28 + rng()*0.58;
  const dirs=['STRONG SELL','SELL','HOLD','BUY','STRONG BUY'];
  const di = pUp<0.22?0 : pUp<0.38?1 : pUp<0.52?2 : pUp<0.68?3 : 4;
  const db=getAllStocks();
  const info=db.find(s=>s.ticker===ticker)||{price:100,name:ticker};
  const price=info.price;
  const predRet=(rng()-0.3)*0.16;
  const atr=price*0.022;
  const isBuy=di>=3;
  const target=isBuy ? price*(1+Math.abs(predRet)) : price*(1-Math.abs(predRet));
  const stop=isBuy ? price-atr*2 : price+atr*2;
  const rr=Math.abs(target-price)/Math.max(Math.abs(price-stop),0.01);
  const kelly=Math.max(0,Math.min(0.25,(rr*pUp-(1-pUp))/rr)) * (0.5+state.riskTolerance);
  const qty=Math.max(1,Math.floor(state.capital*kelly/price));
  const sent=35+rng()*55;
  const tech=35+rng()*55;
  const regimes=['BULL — Low Vol','BULL — Normal','TRANSITIONAL','BEAR — High Vol','VOLATILE'];
  const regime=regimes[Math.floor(rng()*regimes.length)];

  return {
    ticker, name:info.name,
    direction: dirs[di],
    confidence: +(pUp*100).toFixed(1),
    entry_price: +price.toFixed(2),
    target_price: +target.toFixed(2),
    stop_loss: +stop.toFixed(2),
    risk_reward: +rr.toFixed(2),
    kelly_fraction: +kelly.toFixed(4),
    quantity: qty,
    capital_required: +(qty*price).toFixed(2),
    predicted_return: +(predRet*100).toFixed(2),
    predicted_downside: +(-Math.abs(predRet)*55).toFixed(2),
    time_horizon: rr>2?'3-6 weeks':rr>1?'1-2 weeks':'2-5 days',
    regime,
    sentiment_score: +sent.toFixed(1),
    technical_score: +tech.toFixed(1),
    fusion_confidence: +(pUp*90+5).toFixed(1),
    reasoning:[
      `FINSENT fusion confidence: ${(pUp*100).toFixed(1)}% — ${dirs[di]}`,
      `Predicted return: ${predRet>0?'+':''}${(predRet*100).toFixed(2)}% over ${rr>2?'3-6 weeks':'1-2 weeks'}`,
      `Sentiment analysis (FinBERT): ${sent.toFixed(0)}/100 — ${sent>60?'Positive':'Neutral'} news flow`,
      `Technical score: ${tech.toFixed(0)}/100 — RSI(${(30+rng()*40).toFixed(0)}), ${pUp>0.5?'bullish':'bearish'} MACD crossover`,
      `Regime detection: ${regime}`,
      `Kelly-optimal position: ${kelly.toFixed(2)}% of capital → ${qty} shares`,
      `Risk/Reward: 1:${rr.toFixed(1)} — ${rr>2?'Favorable':'Moderate'} expected payoff`,
    ],
  };
}

/* ═══════════════════════════════════════════
   ANIMATION 12 — LOADING SCAN
   ═══════════════════════════════════════════ */
let loadingInterval = null;
function showLoading(){
  const ov=document.getElementById('loadingOverlay');
  ov.classList.add('active');
  const steps=[...document.querySelectorAll('#loadingSteps li')];
  steps.forEach(s=>s.classList.remove('active','done'));
  let i=0;
  loadingInterval=setInterval(()=>{
    if(i>0 && steps[i-1]) { steps[i-1].classList.remove('active'); steps[i-1].classList.add('done'); }
    if(i<steps.length) steps[i].classList.add('active');
    i++;
    if(i>steps.length) clearInterval(loadingInterval);
  }, 350);
}
function hideLoading(){
  clearInterval(loadingInterval);
  document.getElementById('loadingOverlay').classList.remove('active');
}

/* ═══════════════════════════════════════════
   SCREEN 4 — DASHBOARD RENDERER
   ═══════════════════════════════════════════ */
function renderDashboard(){
  const data=state.analysisResults;
  if(!data||!data.signals||!data.signals.length) return;

  // Stock tabs
  document.getElementById('stockTabs').innerHTML=data.signals.map((s,i)=>
    `<button class="stock-tab ${i===0?'active':''}" onclick="switchStock(${i})">${s.ticker}</button>`
  ).join('');

  state.currentStockIdx=0;
  renderStockView(0);
}

function switchStock(idx){
  state.currentStockIdx=idx;
  document.querySelectorAll('.stock-tab').forEach((t,i)=>t.classList.toggle('active',i===idx));
  renderStockView(idx);
  fetchLiveNews();
  updateLivePrices();
  subscribeCurrentTickerToTunnels();
}

function renderStockView(idx){
  const sig=state.analysisResults.signals[idx];
  if(!sig) return;

  /* ── Header ── */
  document.getElementById('dashTicker').textContent=sig.ticker;
  document.getElementById('dashName').textContent=sig.name||'';
  document.getElementById('dashPrice').textContent=fmt$(sig.entry_price);
  const up=sig.predicted_return>=0;
  document.getElementById('dashChange').innerHTML=
    `<span style="color:${up?'var(--signal-buy)':'var(--signal-sell)'}">${up?'▲':'▼'} ${up?'+':''}${sig.predicted_return}% predicted${sig._live?' (LIVE)':''}</span>`;

  /* ── Live indicator ── */
  const liveEl=document.getElementById('liveIndicator');
  if(liveEl) liveEl.style.display=sig._live?'inline-flex':'none';

  /* ── Signal Badge ── */
  const badge=document.getElementById('signalBadge');
  const cls=sig.direction.includes('BUY')?'buy':sig.direction.includes('SELL')?'sell':'hold';
  badge.className='signal-badge '+cls;
  const emoji=cls==='buy'?'🟢':cls==='sell'?'🔴':'🟡';
  document.getElementById('signalDir').textContent=emoji+' '+sig.direction;
  animateCount('signalConf',0,sig.confidence,'%',1200);

  /* ── Trade Levels ── */
  document.getElementById('lvlEntry').textContent=fmt$(sig.entry_price);
  document.getElementById('lvlTarget').textContent=fmt$(sig.target_price);
  document.getElementById('lvlStop').textContent=fmt$(sig.stop_loss);
  document.getElementById('lvlRR').textContent='1 : '+sig.risk_reward.toFixed(1);
  document.getElementById('lvlHorizon').textContent=sig.time_horizon;
  document.getElementById('lvlDeploy').textContent=
    `${sig.quantity} × ${fmt$(sig.entry_price)} = ${fmt$(sig.capital_required)}`;

  /* ── Score Bars (animated) ── */
  requestAnimationFrame(()=>{
    animateBar('barSentiment','valSentiment',sig.sentiment_score);
    animateBar('barTechnical','valTechnical',sig.technical_score);
    animateBar('barFusion','valFusion',sig.fusion_confidence||sig.confidence*0.9);
  });

  /* ── Regime ── */
  const regEl=document.getElementById('regimeBadge');
  regEl.textContent=sig.regime;
  regEl.className='regime-badge '+(
    sig.regime.includes('BULL')?'bull':
    sig.regime.includes('BEAR')?'bear':
    sig.regime.includes('VOLATILE')?'volatile':'transitional'
  );

  /* ── Predictions ── */
  const predR=document.getElementById('predReturn');
  predR.textContent=(sig.predicted_return>=0?'+':'')+sig.predicted_return+'%';
  predR.style.color=sig.predicted_return>=0?'var(--signal-buy)':'var(--signal-sell)';
  document.getElementById('predDownside').textContent=sig.predicted_downside+'%';
  document.getElementById('predDownside').style.color='var(--signal-sell)';

  /* ── Reasoning ── */
  document.getElementById('reasoningList').innerHTML=
    (sig.reasoning||[]).map(r=>`<li>${r}</li>`).join('');

  /* ── Allocation Chart ── */
  renderAllocChart();

  /* ── Risk Meters ── */
  renderRiskMeters();

  /* ── Grafana-style Live Chart ── */
  renderGrafanaChart(sig);
}

/* ═══════════════════════════════════════════
   ANIMATION 9 — COUNTER ANIMATION
   ═══════════════════════════════════════════ */
function animateCount(elId,from,to,suffix,dur){
  const el=document.getElementById(elId);
  if(!el) return;
  const start=performance.now();
  (function tick(now){
    const t=Math.min((now-start)/dur,1);
    const ease=1-Math.pow(1-t,3);
    el.textContent=(from+(to-from)*ease).toFixed(1)+suffix;
    if(t<1) requestAnimationFrame(tick);
  })(start);
}

function animateBar(barId,valId,value){
  const bar=document.getElementById(barId);
  if(bar) bar.style.width=value+'%';
  animateCount(valId,0,value,'/100',1600);
}

/* ═══════════════════════════════════════════
   ANIMATION 4 — CUSTOM GRAFANA CHART DRAW (Live Data)
   ═══════════════════════════════════════════ */
function normalizeLineSeries(series){
  if(!Array.isArray(series)) return [];
  return series
    .map(p=>({
      time: Number(p?.time ?? 0),
      value: Number(p?.value ?? p?.close ?? 0),
    }))
    .filter(p=>Number.isFinite(p.time) && Number.isFinite(p.value));
}

async function fetchChartPayload(sig){
  let candles=null;
  let overlays={};
  let volumeData=null;
  let source='demo';
  let liveQuote=null;
  let liveHeadline='';
  const market=resolveTickerMarket(sig.ticker);
  const tf=state.currentTimeframe||'1D';
  const analysisNews = Array.isArray(sig?._analysisLiveNews) ? sig._analysisLiveNews : [];
  if(analysisNews.length && analysisNews[0]?.title){
    liveHeadline = String(analysisNews[0].title);
  }

  // Pull latest quote/news in parallel so chart reflects live market context.
  try {
    await ensureApiBaseReady();
    const [quoteRes, newsRes] = await Promise.allSettled([
      fetch(`${API_BASE}/live/quote/${encodeURIComponent(sig.ticker)}?market=${encodeURIComponent(market)}`),
      fetch(`${API_BASE}/live/news/${encodeURIComponent(sig.ticker)}?market=${encodeURIComponent(market)}`),
    ]);

    if(quoteRes.status==='fulfilled' && quoteRes.value?.ok){
      const q = await quoteRes.value.json();
      if(q && Number.isFinite(Number(q.price))){
        liveQuote = q;
      }
    }

    if(newsRes.status==='fulfilled' && newsRes.value?.ok){
      const n = await newsRes.value.json();
      const first = (n?.articles || [])[0];
      if(first?.title) liveHeadline = String(first.title);
    }
  } catch(e){
    console.log('[FINSENT] quote/news overlay unavailable:', e.message);
  }

  try {
    if(tf==='1D'){
      const res=await fetch(`${API_BASE}/live/daily/${encodeURIComponent(sig.ticker)}?market=${encodeURIComponent(market)}&period=6mo`);
      if(res.ok){
        const data=await res.json();
        if(Array.isArray(data.candles)&&data.candles.length>5){
          candles=data.candles;
          overlays=data.overlays||{};
          volumeData=data.volume||null;
          source=(data.status||'live').toLowerCase();
        }
      }
    } else {
      const intervalMap={'1m':'1m','5m':'5m','15m':'15m','1H':'1h','4H':'1h'};
      const periodMap={'1m':'1d','5m':'5d','15m':'5d','1H':'5d','4H':'5d'};
      const interval=intervalMap[tf]||'5m';
      const period=periodMap[tf]||'5d';
      const res=await fetch(`${API_BASE}/live/candles/${encodeURIComponent(sig.ticker)}?market=${encodeURIComponent(market)}&interval=${interval}&period=${period}`);
      if(res.ok){
        const data=await res.json();
        if(Array.isArray(data.candles)&&data.candles.length>5){
          candles=data.candles;
          source=(data.status||'live').toLowerCase();
        }
      }
    }
  } catch(e){
    console.log('[FINSENT] Candle API fallback:', e.message);
  }

  if(!Array.isArray(candles)||candles.length<5){
    const intervalSeconds={'1m':60,'5m':300,'15m':900,'1H':3600,'4H':14400,'1D':86400}[tf]||86400;
    candles=genCandles(sig.entry_price,120,intervalSeconds);
    source='synthetic';
  }

  const normalizedCandles=candles
    .map(c=>({
      time:Number(c?.time ?? 0),
      open:Number(c?.open ?? 0),
      high:Number(c?.high ?? 0),
      low:Number(c?.low ?? 0),
      close:Number(c?.close ?? 0),
      volume:Number(c?.volume ?? 0),
    }))
    .filter(c=>
      Number.isFinite(c.time) && Number.isFinite(c.open) && Number.isFinite(c.high) &&
      Number.isFinite(c.low) && Number.isFinite(c.close)
    )
    .sort((a,b)=>a.time-b.time);

  const normalizedVolume=(Array.isArray(volumeData)&&volumeData.length
    ? volumeData.map(v=>({
      time:Number(v?.time ?? 0),
      value:Number(v?.value ?? 0),
      color:v?.color,
    }))
    : normalizedCandles.map(c=>({
      time:c.time,
      value:c.volume||0,
      color:c.close>=c.open?GRAFANA_THEME.volumeUp:GRAFANA_THEME.volumeDown,
    }))
  ).filter(v=>Number.isFinite(v.time) && Number.isFinite(v.value));

  return {
    ticker:sig.ticker,
    market,
    timeframe:tf,
    source,
    liveQuote,
    liveHeadline,
    candles:normalizedCandles,
    overlays:overlays||{},
    volume:normalizedVolume,
  };
}

async function applyGrafanaChart(sig, payload){
  const container=document.getElementById('chartContainer');
  if(!container||typeof Plotly==='undefined') return;

  const candles=payload.candles||[];
  if(!candles.length){
    container.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#8FA2B8;">No candle data available</div>';
    return;
  }

  const showVolume=isIndicatorEnabled('vol');
  const showBB=isIndicatorEnabled('bb');
  const showEMA=isIndicatorEnabled('ema');

  const traces=[];
  const xCandles=candles.map(c=>toDateFromTime(c.time));

  traces.push({
    type:'candlestick',
    x:xCandles,
    open:candles.map(c=>c.open),
    high:candles.map(c=>c.high),
    low:candles.map(c=>c.low),
    close:candles.map(c=>c.close),
    name:`${sig.ticker} OHLC`,
    yaxis:'y',
    increasing:{line:{color:GRAFANA_THEME.up,width:1}, fillcolor:'rgba(34,197,94,0.35)'},
    decreasing:{line:{color:GRAFANA_THEME.down,width:1}, fillcolor:'rgba(242,73,92,0.35)'},
    whiskerwidth:0.8,
    hoverlabel:{font:{family:'JetBrains Mono'}},
  });

  if(showEMA){
    let ema = normalizeLineSeries(payload.overlays?.ema_20);
    if(!ema.length) ema=computeEMAFromCandles(candles,20);
    if(ema.length){
      traces.push({
        type:'scatter',
        mode:'lines',
        x:ema.map(p=>toDateFromTime(p.time)),
        y:ema.map(p=>p.value),
        yaxis:'y',
        name:'EMA 20',
        line:{color:GRAFANA_THEME.warning,width:1.4},
        hovertemplate:'EMA 20 %{y:.2f}<extra></extra>',
      });
    }

    const ema50 = normalizeLineSeries(payload.overlays?.ema_50);
    if(ema50.length){
      traces.push({
        type:'scatter',
        mode:'lines',
        x:ema50.map(p=>toDateFromTime(p.time)),
        y:ema50.map(p=>p.value),
        yaxis:'y',
        name:'EMA 50',
        line:{color:'rgba(255,196,0,0.75)',width:1.2},
        hovertemplate:'EMA 50 %{y:.2f}<extra></extra>',
      });
    }
  }

  const sma50 = normalizeLineSeries(payload.overlays?.sma_50);
  if(sma50.length){
    traces.push({
      type:'scatter',
      mode:'lines',
      x:sma50.map(p=>toDateFromTime(p.time)),
      y:sma50.map(p=>p.value),
      yaxis:'y',
      name:'SMA 50',
      line:{color:'rgba(115,191,105,0.8)',width:1.05},
      hovertemplate:'SMA 50 %{y:.2f}<extra></extra>',
    });
  }

  const sma200 = normalizeLineSeries(payload.overlays?.sma_200);
  if(sma200.length){
    traces.push({
      type:'scatter',
      mode:'lines',
      x:sma200.map(p=>toDateFromTime(p.time)),
      y:sma200.map(p=>p.value),
      yaxis:'y',
      name:'SMA 200',
      line:{color:'rgba(248,113,113,0.8)',width:1.05},
      hovertemplate:'SMA 200 %{y:.2f}<extra></extra>',
    });
  }

  if(showBB){
    let bbUpper=normalizeLineSeries(payload.overlays?.bb_upper);
    let bbLower=normalizeLineSeries(payload.overlays?.bb_lower);
    if(!bbUpper.length||!bbLower.length){
      const bb=computeBollingerFromCandles(candles,20,2);
      bbUpper=bb.upper;
      bbLower=bb.lower;
    }
    if(bbUpper.length&&bbLower.length){
      traces.push({
        type:'scatter',
        mode:'lines',
        x:bbUpper.map(p=>toDateFromTime(p.time)),
        y:bbUpper.map(p=>p.value),
        yaxis:'y',
        name:'BB Upper',
        line:{color:'rgba(87,148,242,0.55)',width:1,dash:'dot'},
        hovertemplate:'BB Upper %{y:.2f}<extra></extra>',
      });
      traces.push({
        type:'scatter',
        mode:'lines',
        x:bbLower.map(p=>toDateFromTime(p.time)),
        y:bbLower.map(p=>p.value),
        yaxis:'y',
        name:'BB Lower',
        line:{color:'rgba(87,148,242,0.55)',width:1,dash:'dot'},
        hovertemplate:'BB Lower %{y:.2f}<extra></extra>',
      });
    }
  }

  if(showVolume){
    traces.push({
      type:'bar',
      x:payload.volume.map(v=>toDateFromTime(v.time)),
      y:payload.volume.map(v=>v.value),
      yaxis:'y2',
      name:'Volume',
      marker:{
        color:payload.volume.map(v=>v.color||'rgba(90,140,200,0.35)'),
      },
      opacity:0.8,
      hovertemplate:'Volume %{y:,.0f}<extra></extra>',
    });
  }

  const annotations=[];
  const shapes=[];
  const levelDefs=[
    {value:sig.entry_price, label:'Entry', color:GRAFANA_THEME.accent},
    {value:sig.target_price, label:'Target', color:GRAFANA_THEME.up},
    {value:sig.stop_loss, label:'Stop', color:GRAFANA_THEME.down},
  ];

  levelDefs.forEach(level=>{
    const v=Number(level.value);
    if(!Number.isFinite(v)||v<=0) return;
    shapes.push({
      type:'line',
      xref:'paper', x0:0, x1:1,
      yref:'y', y0:v, y1:v,
      line:{color:level.color,width:1,dash:'dash'},
    });
    annotations.push({
      xref:'paper', x:1.005,
      yref:'y', y:v,
      text:`${level.label} ${v.toFixed(2)}`,
      showarrow:false,
      xanchor:'left',
      font:{family:'JetBrains Mono',size:10,color:level.color},
      bgcolor:'rgba(15,23,42,0.65)',
      bordercolor:level.color,
      borderwidth:1,
      borderpad:2,
    });
  });

  if(sig._live&&sig.direction){
    const last=candles[candles.length-1];
    if(last){
      const isBuy=sig.direction.includes('BUY');
      const isSell=sig.direction.includes('SELL');
      if(isBuy||isSell){
        const color=isBuy?GRAFANA_THEME.up:GRAFANA_THEME.down;
        annotations.push({
          x:toDateFromTime(last.time),
          y:last.close,
          xref:'x',
          yref:'y',
          text:isBuy?'BUY':'SELL',
          showarrow:true,
          arrowhead:2,
          arrowsize:1,
          arrowwidth:1.2,
          ax:0,
          ay:isBuy?34:-34,
          arrowcolor:color,
          font:{family:'JetBrains Mono',size:10,color:'#E5EDF5'},
          bgcolor:color,
          bordercolor:color,
        });
      }
    }
  }

  const layout={
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:GRAFANA_THEME.panelBg,
    dragmode:'pan',
    hovermode:'x unified',
    margin:{l:54,r:44,t:38,b:34},
    showlegend:true,
    legend:{
      orientation:'h',
      x:0,
      y:1.14,
      bgcolor:'rgba(0,0,0,0)',
      font:{family:'JetBrains Mono',size:10,color:GRAFANA_THEME.muted},
    },
    title:{
      text:`${sig.ticker} • ${payload.timeframe} • ${String(payload.source||'live').toUpperCase()} DATA`,
      x:0.01,
      font:{family:'JetBrains Mono',size:12,color:GRAFANA_THEME.muted},
    },
    xaxis:{
      showgrid:true,
      gridcolor:GRAFANA_THEME.grid,
      showline:false,
      zeroline:false,
      color:GRAFANA_THEME.muted,
      rangeslider:{visible:false},
      tickfont:{family:'JetBrains Mono',size:10,color:GRAFANA_THEME.muted},
      tickformat:payload.timeframe==='1D' ? '%d %b %Y' : '%d %b\n%H:%M',
    },
    yaxis:{
      domain:showVolume ? [0.28,1] : [0,1],
      showgrid:true,
      gridcolor:GRAFANA_THEME.grid,
      zeroline:false,
      color:GRAFANA_THEME.text,
      tickfont:{family:'JetBrains Mono',size:10,color:GRAFANA_THEME.text},
    },
    yaxis2:{
      domain:[0,0.2],
      showgrid:true,
      gridcolor:GRAFANA_THEME.grid,
      zeroline:false,
      color:GRAFANA_THEME.muted,
      visible:showVolume,
      tickfont:{family:'JetBrains Mono',size:9,color:GRAFANA_THEME.muted},
    },
    font:{family:'JetBrains Mono',color:GRAFANA_THEME.text},
    annotations,
    shapes,
    uirevision:`${sig.ticker}-${payload.timeframe}`,
  };

  if(payload.liveQuote && Number.isFinite(Number(payload.liveQuote.price))){
    annotations.push({
      xref:'paper', yref:'paper', x:0.01, y:1.11,
      text:`LIVE ${payload.liveQuote.source||'quote'} ${Number(payload.liveQuote.price).toFixed(2)} (${Number(payload.liveQuote.change_pct||0).toFixed(2)}%)`,
      showarrow:false,
      font:{family:'JetBrains Mono',size:10,color:GRAFANA_THEME.text},
      bgcolor:'rgba(15,23,42,0.55)',
      bordercolor:GRAFANA_THEME.grid,
      borderwidth:1,
      borderpad:3,
      xanchor:'left',
    });
  }

  if(payload.liveHeadline){
    annotations.push({
      xref:'paper', yref:'paper', x:0.99, y:1.11,
      text:`NEWS: ${String(payload.liveHeadline).slice(0, 96)}`,
      showarrow:false,
      font:{family:'JetBrains Mono',size:9,color:GRAFANA_THEME.muted},
      bgcolor:'rgba(15,23,42,0.35)',
      bordercolor:GRAFANA_THEME.grid,
      borderwidth:1,
      borderpad:3,
      xanchor:'right',
      align:'right',
    });
  }

  await Plotly.react(container, traces, layout, {
    responsive:true,
    displaylogo:false,
    modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d','toImage'],
    scrollZoom:true,
  });
}

async function renderGrafanaChart(sig){
  const container=document.getElementById('chartContainer');
  if(!container) return;

  const embedded = await renderGrafanaEmbeddedPanels(sig);
  if(embedded){
    state.liveChartData = null;
    if(state.chartResizeObserver){
      state.chartResizeObserver.disconnect();
      state.chartResizeObserver = null;
    }
    return;
  }

  if(typeof Plotly==='undefined'){
    container.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#8FA2B8;">Loading Grafana chart runtime…</div>';
    return;
  }

  const token=++state.chartRequestToken;
  const payload=await fetchChartPayload(sig);
  if(token!==state.chartRequestToken) return;
  state.liveChartData=payload;
  state.activeChartTicker=sig.ticker;
  await applyGrafanaChart(sig,payload);

  if(state.chartResizeObserver){
    state.chartResizeObserver.disconnect();
  }
  state.chartResizeObserver=new ResizeObserver(()=>{
    if(typeof Plotly==='undefined') return;
    Plotly.Plots.resize(container);
  });
  state.chartResizeObserver.observe(container);
  state.grafanaChart=container;
}

async function updateLiveChartFromBackend(){
  const data=state.analysisResults;
  if(!data||!data.signals||!data.signals.length) return;
  const screen=document.getElementById('screen4');
  if(!screen||!screen.classList.contains('active')) return;
  const sig=data.signals[state.currentStockIdx];
  if(!sig) return;

  const cfg = await ensureGrafanaEmbedConfig();
  if(cfg){
    const refreshed = refreshGrafanaEmbeddedPanels(sig, true);
    if(refreshed) return;

    const embedded = await renderGrafanaEmbeddedPanels(sig);
    if(embedded) return;
  }

  try {
    const token=++state.chartRequestToken;
    const payload=await fetchChartPayload(sig);
    if(token!==state.chartRequestToken) return;
    state.liveChartData=payload;
    state.activeChartTicker=sig.ticker;
    await applyGrafanaChart(sig,payload);
  } catch(e){
    console.log('[FINSENT] Live chart refresh skipped:', e.message);
  }
}

function genCandles(base,n,intervalSeconds=86400){
  const out=[];
  const start=Math.max(1,Number(base)||100);
  let p=start*0.92;
  const now=Math.floor(Date.now()/1000);
  for(let i=n;i>0;i--){
    const chg=(Math.random()-0.48)*p*0.026;
    const o=p;
    p+=chg;
    const c=p;
    const h=Math.max(o,c)+Math.random()*p*0.012;
    const l=Math.min(o,c)-Math.random()*p*0.012;
    out.push({
      time:now-i*intervalSeconds,
      open:+o.toFixed(2),
      high:+h.toFixed(2),
      low:+l.toFixed(2),
      close:+c.toFixed(2),
      volume:Math.floor(4e6+Math.random()*50e6),
    });
  }
  return out;
}

function setTimeframe(btn,tf){
  document.querySelectorAll('.tf-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  state.currentTimeframe=tf;
  if(document.getElementById('screen4')?.classList.contains('active')){
    startLiveUpdates();
  }
  const data=state.analysisResults;
  if(data&&data.signals&&data.signals[state.currentStockIdx]){
    renderGrafanaChart(data.signals[state.currentStockIdx]);
  }
}

function toggleIndicator(btn){
  btn.classList.toggle('active');
  const data=state.analysisResults;
  if(data&&data.signals&&data.signals[state.currentStockIdx]){
    renderGrafanaChart(data.signals[state.currentStockIdx]);
  }
}

/* ═══════════════════════════════════════════
   ALLOCATION DONUT (Custom Grafana)
   ═══════════════════════════════════════════ */
function renderAllocChart(){
  const data=state.analysisResults;
  if(!data) return;
  const sigs=data.signals;
  const colors=['#5794F2','#FFB357','#B877D9','#73BF69','#F2495C','#33A2B8','#C15C17','#A352CC'];
  const deployed=sigs.reduce((s,sig)=>s+sig.capital_required,0);
  const cash=Math.max(0,state.capital-deployed);

  const labels=sigs.map(s=>s.ticker).concat(['CASH']);
  const values=sigs.map(s=>s.capital_required).concat([cash]);
  const bg=sigs.map((_,i)=>colors[i%colors.length]).concat(['#2F3C4F']);
  const hoverValues=values.map(v=>fmt$(v));
  const chartEl=document.getElementById('allocChart');
  if(!chartEl||typeof Plotly==='undefined') return;

  const traces=[{
    type:'pie',
    labels,
    values,
    hole:0.68,
    sort:false,
    direction:'clockwise',
    marker:{
      colors:bg,
      line:{color:'#0B1220',width:1},
    },
    customdata:hoverValues,
    textinfo:'none',
    hovertemplate:'%{label}<br>%{customdata} (%{percent})<extra></extra>',
  }];

  const layout={
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    margin:{l:4,r:4,t:4,b:4},
    showlegend:false,
    annotations:[{
      x:0.5,
      y:0.5,
      xref:'paper',
      yref:'paper',
      text:'ALLOC',
      showarrow:false,
      font:{family:'Orbitron',size:11,color:'#8FA2B8'},
    }],
  };

  Plotly.react(chartEl, traces, layout, {
    responsive:true,
    displayModeBar:false,
  });
  state.allocChart=chartEl;

  document.getElementById('allocList').innerHTML=labels.map((l,i)=>{
    const pct=((values[i]/state.capital)*100).toFixed(1);
    return `<div class="alloc-row"><span><span class="dot" style="background:${bg[i]}"></span>${l}</span><span style="color:${bg[i]}">${pct}%</span></div>`;
  }).join('');
}

/* ═══════════════════════════════════════════
   RISK METERS (ANIMATION 8 — GAUGE SWEEP)
   ═══════════════════════════════════════════ */
function renderRiskMeters(){
  const risk=state.analysisResults.risk||{};
  const meters=[
    {label:'Sharpe',  val:risk.sharpe_ratio||0,         max:4,  color:'var(--cyan)'},
    {label:'Sortino', val:risk.sortino_ratio||0,        max:4,  color:'var(--purple)'},
    {label:'Max DD',  val:Math.abs(risk.max_drawdown||0),max:25, color:'var(--signal-sell)'},
    {label:'VaR 95%', val:risk.var_95||0,               max:15, color:'var(--gold)'},
    {label:'Calmar',  val:risk.calmar_ratio||0,         max:4,  color:'var(--signal-buy)'},
    {label:'Win Rate',val:risk.win_rate||0,             max:100,color:'var(--orange)'},
  ];
  document.getElementById('riskMeters').innerHTML=meters.map(m=>{
    const pct=Math.min((m.val/m.max)*100,100);
    return `<div class="risk-meter">
      <div class="label">${m.label}</div>
      <div class="gauge-mini"><div class="gauge-fill" style="width:0%;background:${m.color};" data-w="${pct}"></div></div>
      <div class="val" style="color:${m.color}">${typeof m.val==='number'?m.val.toFixed(2):m.val}</div>
    </div>`;
  }).join('');
  // Animate gauge fills
  requestAnimationFrame(()=>{
    document.querySelectorAll('.gauge-fill[data-w]').forEach(el=>{
      el.style.width=el.dataset.w+'%';
    });
  });
}

/* ═══════════════════════════════════════════
   SCREEN 5 — PORTFOLIO SUMMARY
   ═══════════════════════════════════════════ */
function renderPortfolioSummary(){
  const data=state.analysisResults;
  if(!data) return;
  const alloc=data.portfolio?.allocation||{};
  const deployed=alloc.total_deployed||data.signals.reduce((s,sig)=>s+sig.capital_required,0);

  document.getElementById('portCapital').textContent=fmt$(state.capital);
  document.getElementById('portDeployed').textContent=fmt$(deployed);
  document.getElementById('portCash').textContent=fmt$(Math.max(0,state.capital-deployed));

  // Table rows
  document.getElementById('portTableBody').innerHTML=data.signals.map(s=>{
    const dc=s.direction.includes('BUY')?'var(--signal-buy)':s.direction.includes('SELL')?'var(--signal-sell)':'var(--signal-hold)';
    const w=((s.capital_required/state.capital)*100).toFixed(1);
    return `<tr>
      <td style="color:var(--cyan);font-weight:600;">${s.ticker}</td>
      <td style="color:${dc};font-weight:600;">${s.direction}</td>
      <td>${s.confidence}%</td>
      <td>${s.quantity}</td>
      <td>${fmt$(s.entry_price)}</td>
      <td style="color:var(--signal-buy)">${fmt$(s.target_price)}</td>
      <td style="color:var(--signal-sell)">${fmt$(s.stop_loss)}</td>
      <td>${fmt$(s.capital_required)}</td>
      <td><span style="color:var(--gold)">${w}%</span></td>
    </tr>`;
  }).join('');

  // Risk summary cards
  const risk=data.risk||{};
  const metrics=[
    {label:'EXPECTED RETURN',      val:(risk.annualized_return||0)+'%', color:'var(--signal-buy)'},
    {label:'SHARPE RATIO',         val:risk.sharpe_ratio||'—',         color:'var(--cyan)'},
    {label:'SORTINO RATIO',        val:risk.sortino_ratio||'—',        color:'var(--purple)'},
    {label:'VOLATILITY',           val:'±'+(risk.annualized_volatility||0)+'%', color:'var(--gold)'},
    {label:'MAX DRAWDOWN',         val:(risk.max_drawdown||0)+'%',     color:'var(--signal-sell)'},
    {label:'VALUE-AT-RISK (95%)',   val:fmt$(state.capital*(risk.var_95||0)/100), color:'var(--orange)'},
  ];
  document.getElementById('riskSummary').innerHTML=metrics.map(m=>
    `<div class="risk-card"><div class="metric-label">${m.label}</div><div class="metric-value" style="color:${m.color}">${m.val}</div></div>`
  ).join('');
}

/* ═══════════════════════════════════════════
   EXPORT REPORT
   ═══════════════════════════════════════════ */
function exportReport(){
  const data=state.analysisResults;
  if(!data) return;

  const line='═'.repeat(55);
  let txt=`${line}\n  FINSENT NET PRO — PORTFOLIO INTELLIGENCE REPORT\n${line}\n\n`;
  txt+=`Capital:        ${fmt$(state.capital)}\n`;
  txt+=`Risk Tolerance: ${(state.riskTolerance*100).toFixed(0)}%\n`;
  txt+=`Horizon:        ${state.horizon}\n`;
  txt+=`Markets:        ${state.selectedMarkets.join(', ')||'SP500'}\n`;
  txt+=`Generated:      ${new Date().toISOString()}\n\n`;

  data.signals.forEach(s=>{
    txt+=`${'─'.repeat(45)}\n`;
    txt+=`  ${s.ticker} — ${s.name||''}\n`;
    txt+=`${'─'.repeat(45)}\n`;
    txt+=`  Signal:     ${s.direction} (${s.confidence}% confidence)\n`;
    txt+=`  Entry:      ${fmt$(s.entry_price)}\n`;
    txt+=`  Target:     ${fmt$(s.target_price)}\n`;
    txt+=`  Stop Loss:  ${fmt$(s.stop_loss)}\n`;
    txt+=`  R/R Ratio:  1:${s.risk_reward}\n`;
    txt+=`  Quantity:   ${s.quantity} shares\n`;
    txt+=`  Deploy:     ${fmt$(s.capital_required)}\n`;
    txt+=`  Horizon:    ${s.time_horizon}\n`;
    txt+=`  Regime:     ${s.regime}\n\n`;
    txt+=`  Reasoning:\n`;
    (s.reasoning||[]).forEach(r=>txt+=`    • ${r}\n`);
    txt+='\n';
  });

  const risk=data.risk||{};
  txt+=`${line}\n  PORTFOLIO RISK METRICS\n${line}\n`;
  txt+=`  Sharpe Ratio:     ${risk.sharpe_ratio}\n`;
  txt+=`  Sortino Ratio:    ${risk.sortino_ratio}\n`;
  txt+=`  Max Drawdown:     ${risk.max_drawdown}%\n`;
  txt+=`  VaR (95%):        ${risk.var_95}%\n`;
  txt+=`  Exp. Return:      ${risk.annualized_return}%\n`;
  txt+=`  Volatility:       ${risk.annualized_volatility}%\n\n`;
  txt+=`${'═'.repeat(55)}\n`;
  txt+=`  Generated by FINSENT NET PRO — AI-Powered Quantitative Trading Intelligence\n`;

  const blob=new Blob([txt],{type:'text/plain'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=`FINSENT_Report_${new Date().toISOString().slice(0,10)}.txt`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ═══════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', ()=>{
  updateRiskLabel();
  updateCurrency();
  ensureApiBaseReady().catch(()=>{});
});
