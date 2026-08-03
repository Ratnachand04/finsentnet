# FINSENT Live Grafana Graphs (All Selected Tickers)

This setup ensures Grafana always plots live graphs for the currently selected companies/stocks from the analyze flow.

## How it works

1. Every analyze call updates the backend-selected ticker set.
2. Prometheus scrapes `http://localhost:8000/metrics` every 5 seconds.
3. During each scrape, backend fetches fresh live quotes for selected tickers using configured API keys.
4. Grafana dashboard reads Prometheus metrics and draws live charts for all selected tickers.

## Start services

From `finsentnet_pro/observability`:

```powershell
docker compose -f docker-compose.grafana.yml up -d
```

## Access

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

Default Grafana credentials:

- user: `admin`
- password: `admin`

## Required backend runtime

Run backend on port 8000 and keep your live API keys configured (`FMP_API_KEYS`, `FINNHUB_API_KEY`, etc.).

Suggested env:

```powershell
$env:FINSENT_DEVICE='cuda'
$env:GRAFANA_EMBED_ENABLED='1'
$env:GRAFANA_URL='http://localhost:3000'
$env:GRAFANA_DASHBOARD_UID='finsent-live'
$env:GRAFANA_DASHBOARD_SLUG='finsent-live-tickers'
$env:GRAFANA_PANEL_PRICE_ID='1'
$env:GRAFANA_PANEL_VOLUME_ID='3'
```

## Metrics exposed

- `finsent_live_price{ticker,market,source}`
- `finsent_live_change_pct{ticker,market,source}`
- `finsent_live_volume{ticker,market,source}`
- `finsent_live_last_update_epoch{ticker,market,source}`
- `finsent_live_scrape_success{ticker,market}`

## Notes

- Selected tickers are sourced from `/api/analyze` requests.
- Dashboard ticker variable is dynamic: `label_values(finsent_live_price, ticker)`.
- Frontend Grafana embeds now pass all selected tickers, so charts represent the full selected basket.
