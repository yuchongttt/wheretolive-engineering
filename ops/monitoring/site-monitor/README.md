# site-monitor — external blackbox site health checker

Runs from **Hetzner** (or any external host outside the home network),
hits `https://wheretolive.xyz` via public DNS + cloudflared, and reports
results to the Mac admin endpoint `/api/admin/site-health/ingest`.

## What it catches

For each URL in `checks.py:URLS`:

| check_type | what fails it |
|---|---|
| `http` | non-200 status, navigation timeout |
| `meta` | missing title/description/canonical/og:title/og:image |
| `selector` | main content selector not visible after load |
| `console` | any `console.error` during page load |
| `network` | any 4xx/5xx XHR/fetch/img (excl. favicon) — `warn` only |
| `auth-gate` | (login-required URLs only) login wall doesn't render |
| `lighthouse` | (public URLs only) any of perf/seo/a11y/best-practices < 50 |

Plus 2 site-level checks: `/robots.txt` and `/sitemap.xml` HTTP 200.

## Deployment

```bash
# On Hetzner (one-time setup)
sudo apt-get install -y python3.12-venv
sudo mkdir -p /opt/wtl-monitor
sudo python3 -m venv /opt/wtl-monitor/.venv
sudo /opt/wtl-monitor/.venv/bin/pip install playwright httpx
sudo /opt/wtl-monitor/.venv/bin/playwright install --with-deps chromium
sudo npm install -g lighthouse

# Code (rsync from the repo; MONITOR_HOST = the external box)
rsync -av ops/monitoring/site-monitor/ "root@$MONITOR_HOST:/opt/wtl-monitor/"

# Env file
sudo tee /etc/wtl-monitor.env > /dev/null <<EOF
INGEST_URL=https://wheretolive.xyz/api/admin/site-health/ingest
SITE_HEALTH_INGEST_TOKEN=<set-me: same value as the web app's SITE_HEALTH_INGEST_TOKEN>
EOF
sudo chmod 600 /etc/wtl-monitor.env

# Install systemd units
sudo cp /opt/wtl-monitor/wtl-monitor.service /etc/systemd/system/
sudo cp /opt/wtl-monitor/wtl-monitor.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wtl-monitor.timer

# Trigger a one-off run to verify
sudo systemctl start wtl-monitor.service
sudo journalctl -u wtl-monitor.service -f
```

## Local dev / debug

```bash
INGEST_URL=https://wheretolive.xyz/api/admin/site-health/ingest \
SITE_HEALTH_INGEST_TOKEN=<set-me> \
/opt/wtl-monitor/.venv/bin/python /opt/wtl-monitor/monitor.py
```

## Wall time budget

Per run (from `checks.py` / `monitor.py`):
- 7 public URLs × Lighthouse (90 s timeout each, 180 s for `/`, 6 s pause between runs)
- 10 URLs × Playwright (30 s navigation timeout + up to 8 s network-idle wait)
- 2 static fetches + push

systemd `RuntimeMaxSec=1500` (25 min) is the hard cap.

Frequency: `OnCalendar=*-*-* *:05:00` — every hour at :05.
