#!/usr/bin/env python3
"""External site health monitor for wheretolive.xyz.

Runs from Hetzner (an external host outside the home network), hits
the public DNS, and reports to the Mac admin endpoint. Catches issues that
internal monitoring can't see — cloudflared tunnel drops, DNS failures,
SSR errors, missing SEO meta, JS console errors, Lighthouse regressions.

For each URL in checks.URLS the monitor emits these check rows:
    http        HTTP status + final URL + total load time
    meta        title / description / canonical / og:* presence
    selector    critical content selector visible after load
    console     JS console.error count
    network     4xx/5xx network requests (XHR/fetch/img)
    auth-gate   (auth=True only) verifies login wall renders
    lighthouse  (auth=False only) full Lighthouse audit, 4 score categories

Plus site-level: /robots.txt + /sitemap.xml HTTP checks.

Push: POST to INGEST_URL with x-ingest-token header. If push fails, the
payload is queued in queue.db; the next successful run flushes the queue.

Env (set via systemd EnvironmentFile=/etc/wtl-monitor.env):
    INGEST_URL                  full URL of the Mac ingest endpoint
    SITE_HEALTH_INGEST_TOKEN    shared secret matching Mac .env
"""

from __future__ import annotations

import glob
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

from checks import BASE_URL, STATIC_PATHS, URLS

# Benign console-error substrings (lowercased) ignored by the `console` check —
# 3rd-party / expected-anonymous noise, not site bugs. See on_console().
CONSOLE_IGNORE = (
    "%c%d",                       # Cloudflare Turnstile styled debug log
    "font-size:0;color:transparent",
    "challenges.cloudflare.com",  # Turnstile assets
    "turnstile",
    "cloudflareinsights",         # CF analytics beacon
    "/cdn-cgi/rum",
    "a status of 401",            # anon getUser 401 (monitor is logged-out)
)

# The Cloudflare RUM beacon is injected unconditionally in
# web/src/app/layout.tsx, so every navigation this monitor makes lands in
# Cloudflare Web Analytics as a real "visit". Measured 2026-09-03: a large share of the
# recorded visits were this monitor, and the "poor LCP" band was the
# Lighthouse leg alone — Lighthouse scores the very
# same pages 99-100/100, so that LCP is a late paint candidate from the audit
# lifecycle, not anything a user experiences. Suppress the reporting so the RUM
# panel shows real users. (It cannot be excluded in CF's dashboard: RUM is
# client-side, so CF's bot management never sees it, and its only filters are
# browser/OS/country — filtering the Lighthouse mobile UA would drop real
# mobile users too.)
#
# We intercept ONLY the payload endpoint, not the beacon script. Verified
# 2026-09-03: beacon.min.js loads from static.cloudflareinsights.com and then
# POSTs the measurement FIRST-PARTY to wheretolive.xyz/cdn-cgi/rum, so killing
# the POST is sufficient. Two rejected alternatives, both measured:
#   - route.abort() on the script: works, but a blocked <script src> logs
#     "Failed to load resource: net::ERR_FAILED" and the console check below
#     turned all 10 URLs red (2 phantom errors each).
#   - fulfilling the script with an empty 200: trips its SRI `integrity`
#     attribute — "Failed to find a valid digest".
# Leaving the script alone also keeps the audit honest: Lighthouse measures the
# page with the beacon present, exactly as a real visitor gets it.
#
# If Cloudflare ever moves the endpoint this silently stops working and the
# monitor starts polluting RUM again. To re-check: load a page under Playwright
# and assert nothing matching /cdn-cgi/ reaches `requestfinished`.
RUM_PAYLOAD_RE = re.compile(r"/cdn-cgi/rum")
RUM_PAYLOAD_GLOB = "*/cdn-cgi/rum*"


def suppress_rum_payload(route) -> None:
    """Answer the RUM upload with an empty 204 so nothing reaches Cloudflare."""
    route.fulfill(status=204, body="")

# Benign 3rd-party request URLs whose 4xx is EXPECTED (not a site fault),
# skipped by the `network` check. Cloudflare Turnstile's challenge-platform
# handshake on the auth pages returns 401 as part of its normal anonymous
# flow — it was warning every single hourly run, painting the heat strip
# amber ("all yellow") even though the site was fine.
NETWORK_IGNORE = (
    "challenges.cloudflare.com",
    "/cdn-cgi/challenge-platform/",
    "cloudflareinsights",
    "/cdn-cgi/rum",
)


# Lighthouse uses chrome-launcher to spawn Chrome. On Hetzner we don't
# install system Chrome — we use Playwright's bundled chromium. Resolve
# its path once at startup and feed it to Lighthouse via CHROME_PATH.
def _resolve_chrome_path() -> str | None:
    candidates = sorted(
        glob.glob(os.path.expanduser(
            "~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome"
        ))
    )
    return candidates[-1] if candidates else None


CHROME_PATH = _resolve_chrome_path()


ROOT = Path("/opt/wtl-monitor")
LOGS = ROOT / "logs"
LIGHTHOUSE_REPORTS = ROOT / "lighthouse-reports"
SCREENSHOTS = ROOT / "screenshots"
QUEUE_DB = ROOT / "queue.db"

NAV_TIMEOUT_MS = 30_000
NETWORK_IDLE_MS = 8_000
LIGHTHOUSE_TIMEOUT_S = 90
# Homepage carries a large hero video + heavy hero animations (Reveal /
# ScrollProgress / SVG skyline) — on Hetzner's 2-vCPU box chrome's perf
# profiler often can't finish in 90s. Allow more headroom for this URL only.
LIGHTHOUSE_TIMEOUT_HEAVY_S = 180
LIGHTHOUSE_HEAVY_PATHS = {"/"}
# Sleep between Lighthouse runs so the previous chromium has time to fully
# release its CPU / memory before the next one starts. Without this, eight
# back-to-back runs saturate the 2-vCPU host and inflate every Lighthouse
# perf score's TBT (we saw /check fall to perf=67 from a true ~100).
LIGHTHOUSE_INTER_RUN_SLEEP_S = 6
SELECTOR_VISIBLE_MS = 5_000
ARTIFACT_RETENTION_S = 86_400  # 24h

HOSTNAME = socket.gethostname()
ORIGIN = f"hetzner-{HOSTNAME}"


# --------------------------------------------------------------------------- #
# Push queue (handles transient Mac unreachability)                           #
# --------------------------------------------------------------------------- #

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_queue_schema() -> None:
    conn = sqlite3.connect(str(QUEUE_DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_pushes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            payload_json TEXT NOT NULL,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def push_payload(payload: dict) -> bool:
    """POST to Mac. Returns True on 200, False otherwise."""
    ingest_url = os.environ["INGEST_URL"]
    token = os.environ["SITE_HEALTH_INGEST_TOKEN"]
    try:
        r = httpx.post(
            ingest_url,
            json=payload,
            headers={"x-ingest-token": token},
            timeout=30.0,
        )
        if r.status_code == 200:
            return True
        print(f"[push] {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[push] exception: {e}", file=sys.stderr)
        return False


def queue_payload(payload: dict) -> None:
    conn = sqlite3.connect(str(QUEUE_DB))
    conn.execute(
        "INSERT INTO pending_pushes(payload_json) VALUES (?)",
        (json.dumps(payload),),
    )
    conn.commit()
    conn.close()


def flush_queue() -> int:
    conn = sqlite3.connect(str(QUEUE_DB))
    rows = conn.execute(
        "SELECT id, payload_json FROM pending_pushes ORDER BY id"
    ).fetchall()
    flushed = 0
    for row_id, payload_json in rows:
        if push_payload(json.loads(payload_json)):
            conn.execute("DELETE FROM pending_pushes WHERE id = ?", (row_id,))
            conn.commit()
            flushed += 1
        else:
            # Stop on first failure — endpoint is still down, retry next run.
            break
    conn.close()
    return flushed


# --------------------------------------------------------------------------- #
# Static checks (robots.txt + sitemap.xml)                                    #
# --------------------------------------------------------------------------- #

def check_static(path: str) -> dict:
    start = time.monotonic()
    try:
        r = httpx.get(BASE_URL + path, timeout=15.0, follow_redirects=True)
        duration_ms = int((time.monotonic() - start) * 1000)
        ok = r.status_code == 200 and len(r.text) > 0
        return {
            "url": path,
            "check_type": "static-resource",
            "status": "pass" if ok else "fail",
            "details": {
                "http_status": r.status_code,
                "size_bytes": len(r.content),
                "first_chars": r.text[:200],
            },
            "duration_ms": duration_ms,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "url": path,
            "check_type": "static-resource",
            "status": "fail",
            "details": {"error": str(e)},
            "duration_ms": int((time.monotonic() - start) * 1000),
        }


# --------------------------------------------------------------------------- #
# Playwright per-URL check (returns multiple rows: http/meta/selector/...)    #
# --------------------------------------------------------------------------- #

def safe_filename(path: str) -> str:
    name = path.strip("/").replace("/", "_").replace("?", "_").replace("=", "_")
    return name or "home"


def check_url_via_playwright(browser, entry: dict) -> list[dict]:
    path = entry["path"]
    full_url = BASE_URL + path
    main_selector = entry["main_selector"]
    is_auth_gated = entry["auth"]

    console_errors: list[dict] = []
    bad_requests: list[dict] = []

    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) "
            "wtl-monitor/1.0 (+https://wheretolive.xyz)"
        ),
        viewport={"width": 1280, "height": 800},
    )
    # Keep our own synthetic traffic out of Cloudflare RUM — see RUM_PAYLOAD_RE.
    context.route(RUM_PAYLOAD_RE, suppress_rum_payload)
    page = context.new_page()

    def on_console(msg):
        try:
            if msg.type == "error":
                text = msg.text or ""
                # Ignore benign 3rd-party / expected-anonymous console noise — these
                # are NOT site bugs and were turning the whole heatmap red:
                #  - Cloudflare Turnstile's styled debug logs on the auth pages
                #    ("%c%d font-size:0;color:transparent") = the captcha working.
                #  - The monitor is intentionally logged-OUT, so Supabase's
                #    getUser → 401 on /auth/v1/user ("Failed to load resource:
                #    …status of 401") is expected, not a fault (a real 4xx still
                #    surfaces via the separate `network` check).
                #  - Cloudflare insights/RUM beacon.
                lo = text.lower()
                if any(p in lo for p in CONSOLE_IGNORE):
                    return
                console_errors.append({"text": text[:300]})
        except Exception:
            pass

    def on_response(resp):
        try:
            if resp.status >= 400 and not resp.url.endswith("favicon.ico"):
                lo = resp.url.lower()
                if any(p in lo for p in NETWORK_IGNORE):
                    return
                bad_requests.append({"url": resp.url[:300], "status": resp.status})
        except Exception:
            pass

    page.on("console", on_console)
    page.on("response", on_response)

    rows: list[dict] = []
    start = time.monotonic()
    http_status = None
    final_url = None

    network_idle = True
    dcl_ms = None
    try:
        resp = page.goto(full_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        http_status = resp.status if resp else None
        final_url = page.url
        dcl_ms = int((time.monotonic() - start) * 1000)
        try:
            page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_MS)
        except Exception:
            # networkidle can time out on pages with SSE/WS or third-party
            # heartbeats (e.g. /auth/login's Turnstile widget polls
            # challenges.cloudflare.com forever) — that's fine, we already
            # have DOMContentLoaded.
            network_idle = False
    except Exception as e:
        rows.append({
            "url": path,
            "check_type": "http",
            "status": "fail",
            "details": {"error": str(e)[:300], "phase": "navigation"},
            "duration_ms": int((time.monotonic() - start) * 1000),
        })
        context.close()
        return rows

    # Pages that never reach networkidle would otherwise report the full
    # NETWORK_IDLE_MS timeout as latency (we saw /auth/login pinned at a
    # constant ~8.2s, 5-7x every other page, purely from Turnstile
    # heartbeats). Fall back to DOMContentLoaded — closer to what users
    # actually feel — and flag it in details.
    settled_ms = int((time.monotonic() - start) * 1000)
    load_ms = settled_ms if network_idle else (dcl_ms or settled_ms)

    # http
    http_ok = http_status == entry["expected_status"]
    http_row = {
        "url": path,
        "check_type": "http",
        "status": "pass" if http_ok else "fail",
        "details": {
            "http_status": http_status,
            "final_url": final_url,
            "load_ms": load_ms,
            "network_idle": network_idle,
        },
        "duration_ms": load_ms,
    }
    rows.append(http_row)

    # meta
    try:
        meta = page.evaluate(
            """() => ({
                title: document.title || null,
                description: document.querySelector('meta[name=description]')?.content || null,
                canonical: document.querySelector('link[rel=canonical]')?.href || null,
                og_title: document.querySelector('meta[property="og:title"]')?.content || null,
                og_image: document.querySelector('meta[property="og:image"]')?.content || null,
                og_description: document.querySelector('meta[property="og:description"]')?.content || null,
                robots: document.querySelector('meta[name=robots]')?.content || null,
                viewport: document.querySelector('meta[name=viewport]')?.content || null,
            })"""
        )
    except Exception as e:  # noqa: BLE001
        meta = {"error": str(e)[:200]}

    required = ["title", "description", "canonical", "og_title", "og_image"]
    missing = [k for k in required if not meta.get(k)]
    meta_status = "pass" if not missing else ("warn" if len(missing) <= 2 else "fail")
    rows.append({
        "url": path,
        "check_type": "meta",
        "status": meta_status,
        "details": {**meta, "missing": missing},
        "duration_ms": 5,
    })

    # selector
    selector_present = False
    try:
        selector_present = page.locator(main_selector).first.is_visible(
            timeout=SELECTOR_VISIBLE_MS
        )
    except Exception:
        selector_present = False
    rows.append({
        "url": path,
        "check_type": "selector",
        "status": "pass" if selector_present else "fail",
        "details": {"selector": main_selector, "found": selector_present},
        "duration_ms": 5,
    })

    # auth-gate
    if is_auth_gated:
        try:
            body_text = (page.inner_text("body") or "")[:2000]
        except Exception:
            body_text = ""
        is_loginwall = any(
            kw in body_text.lower() for kw in ["sign in", "log in", "login", "登录"]
        )
        rows.append({
            "url": path,
            "check_type": "auth-gate",
            "status": "pass" if is_loginwall else "fail",
            "details": {
                "shows_loginwall": is_loginwall,
                "body_excerpt": body_text[:300],
            },
            "duration_ms": 5,
        })

    # console
    rows.append({
        "url": path,
        "check_type": "console",
        "status": "pass" if not console_errors else "fail",
        "details": {"errors": console_errors[:20], "count": len(console_errors)},
        "duration_ms": 5,
    })

    # network — warn (not fail), since some 4xx are legit (e.g., probes)
    rows.append({
        "url": path,
        "check_type": "network",
        "status": "pass" if not bad_requests else "warn",
        "details": {"bad_requests": bad_requests[:20], "count": len(bad_requests)},
        "duration_ms": 5,
    })

    # screenshot on fail
    if any(r["status"] == "fail" for r in rows):
        png_name = f"{safe_filename(path)}_{int(time.time())}.png"
        png_path = SCREENSHOTS / png_name
        try:
            page.screenshot(path=str(png_path), full_page=True)
            for r in rows:
                if r["check_type"] == "http":
                    r["details"]["screenshot"] = png_name
                    break
        except Exception:
            pass

    context.close()
    return rows


# --------------------------------------------------------------------------- #
# Lighthouse per-URL                                                          #
# --------------------------------------------------------------------------- #

def run_lighthouse(url: str, label: str) -> dict:
    out_json = LIGHTHOUSE_REPORTS / f"{label}_{int(time.time())}.json"
    start = time.monotonic()
    env = os.environ.copy()
    if CHROME_PATH:
        env["CHROME_PATH"] = CHROME_PATH
    path = urlparse(url).path or "/"
    timeout_s = (
        LIGHTHOUSE_TIMEOUT_HEAVY_S if path in LIGHTHOUSE_HEAVY_PATHS
        else LIGHTHOUSE_TIMEOUT_S
    )
    try:
        proc = subprocess.run(
            [
                "lighthouse",
                url,
                "--output=json",
                f"--output-path={out_json}",
                "--quiet",
                "--chrome-flags=--headless --no-sandbox --disable-gpu",
                # Identify ourselves: modern Lighthouse's emulated mobile UA no
                # longer contains "Chrome-Lighthouse", so the site's analytics
                # bot filter stopped matching and every hourly audit registered
                # as a real mobile visitor. Keep the
                # mobile device emulation but pin a UA the pageview endpoint's
                # BOT_PATTERNS (wtl-monitor) matches.
                "--emulated-user-agent=Mozilla/5.0 (Linux; Android 11; moto g power (2022)) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36 Chrome-Lighthouse wtl-monitor/1.0",
                "--only-categories=performance,accessibility,best-practices,seo",
                "--throttling-method=provided",
                # Keep audits out of Cloudflare RUM — see RUM_PAYLOAD_RE.
                f"--blocked-url-patterns={RUM_PAYLOAD_GLOB}",
                "--max-wait-for-load=45000",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    # NOTE: a Lighthouse run that times out, exits non-zero ("Runtime error …
    # unable to reliably load the page"), or emits an unparseable report is a
    # MEASUREMENT flake, not a site defect — Chrome/LH couldn't measure, which
    # happens intermittently on the Hetzner box. Report these as `warn` (still
    # recorded + visible) rather than `fail`, so a one-off LH hiccup doesn't
    # colour the heat strip. A real score regression (worst < 50 below) keeps
    # its `fail`.
    except subprocess.TimeoutExpired:
        return {
            "url": path,
            "check_type": "lighthouse",
            "status": "warn",
            "details": {"error": f"timeout after {timeout_s}s", "flaky": True},
            "duration_ms": timeout_s * 1000,
        }

    duration_ms = int((time.monotonic() - start) * 1000)
    if proc.returncode != 0 or not out_json.exists():
        return {
            "url": urlparse(url).path or "/",
            "check_type": "lighthouse",
            "status": "warn",
            "details": {"error": (proc.stderr or proc.stdout)[-500:], "flaky": True},
            "duration_ms": duration_ms,
        }

    try:
        data = json.loads(out_json.read_text())
    except Exception as e:  # noqa: BLE001
        return {
            "url": urlparse(url).path or "/",
            "check_type": "lighthouse",
            "status": "warn",
            "details": {"error": f"failed to parse report: {e}", "flaky": True},
            "duration_ms": duration_ms,
        }

    categories = data.get("categories", {}) or {}
    scores: dict[str, int] = {}
    for k, v in categories.items():
        s = v.get("score") if isinstance(v, dict) else None
        scores[k] = int(round((s or 0) * 100))

    # Some paths are intentionally noindex per web/src/app/robots.ts
    # (e.g. /auth/*) — Lighthouse can't tell "deliberately excluded" from
    # "broken SEO" and always penalises with seo ≈ 50. Exclude SEO from the
    # threshold check on these paths unless it drops below 30 (real defect).
    path = urlparse(url).path or "/"
    NOINDEX_PATHS = ("/auth/",)
    is_noindex = any(path.startswith(p) for p in NOINDEX_PATHS)

    effective_scores = dict(scores)
    if is_noindex and effective_scores.get("seo", 100) >= 30:
        effective_scores.pop("seo", None)

    worst = min(effective_scores.values()) if effective_scores else 0
    if worst >= 80:
        status = "pass"
    elif worst >= 50:
        status = "warn"
    else:
        status = "fail"

    return {
        "url": path,
        "check_type": "lighthouse",
        "status": status,
        "details": {
            "scores": scores,  # keep original SEO score visible in admin UI
            "report": out_json.name,
            **({"seo_excluded": True} if is_noindex else {}),
        },
        "duration_ms": duration_ms,
    }


# --------------------------------------------------------------------------- #
# Cleanup                                                                     #
# --------------------------------------------------------------------------- #

def cleanup_old_artifacts() -> None:
    cutoff = time.time() - ARTIFACT_RETENTION_S
    for d in (SCREENSHOTS, LIGHTHOUSE_REPORTS):
        if not d.exists():
            continue
        for f in d.iterdir():
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> int:
    started_at = utc_now_iso()
    t0 = time.monotonic()

    LOGS.mkdir(parents=True, exist_ok=True)
    LIGHTHOUSE_REPORTS.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    ensure_queue_schema()

    if "INGEST_URL" not in os.environ or "SITE_HEALTH_INGEST_TOKEN" not in os.environ:
        print("ERROR: INGEST_URL and SITE_HEALTH_INGEST_TOKEN must be set", file=sys.stderr)
        return 1

    checks: list[dict] = []

    # 1) Static endpoints
    for p in STATIC_PATHS:
        print(f"[static] {p}", flush=True)
        checks.append(check_static(p))

    # 2) Playwright (browser kept alive, contexts disposable)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu"],
        )
        for entry in URLS:
            print(f"[playwright] {entry['path']}", flush=True)
            try:
                checks.extend(check_url_via_playwright(browser, entry))
            except Exception as e:  # noqa: BLE001
                checks.append({
                    "url": entry["path"],
                    "check_type": "http",
                    "status": "fail",
                    "details": {"error": f"runner crashed: {e}"[:300]},
                    "duration_ms": 0,
                })
        browser.close()

    # 3) Lighthouse — public pages only (login walls skew scores). Sleep
    #    between runs so the previous chromium fully releases CPU/RAM;
    #    on a 2-vCPU Hetzner box, back-to-back Lighthouse runs saturated
    #    CPU and inflated TBT for every run after the first.
    public_entries = [e for e in URLS if not e["auth"]]
    for i, entry in enumerate(public_entries):
        url = BASE_URL + entry["path"]
        print(f"[lighthouse] {url}", flush=True)
        checks.append(run_lighthouse(url, entry["label"]))
        if i < len(public_entries) - 1:
            time.sleep(LIGHTHOUSE_INTER_RUN_SLEEP_S)

    completed_at = utc_now_iso()
    duration_s = int(time.monotonic() - t0)

    payload = {
        "origin": ORIGIN,
        "started_at": started_at,
        "completed_at": completed_at,
        "notes": f"duration={duration_s}s checks={len(checks)}",
        "checks": checks,
    }

    cleanup_old_artifacts()

    pushed = push_payload(payload)
    if not pushed:
        print("[main] push failed — queuing payload for next run", file=sys.stderr)
        queue_payload(payload)
    else:
        flushed = flush_queue()
        if flushed:
            print(f"[main] flushed {flushed} backlog payloads", flush=True)

    print(
        f"[main] done in {duration_s}s; {len(checks)} checks; "
        f"push={'ok' if pushed else 'queued'}",
        flush=True,
    )
    # Exit clean even when push queued — systemd shouldn't flap on transient
    # network blips; the queue handles it.
    return 0


if __name__ == "__main__":
    sys.exit(main())
