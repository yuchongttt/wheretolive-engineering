"""URL list + per-URL expectations for the external site health monitor.

Hetzner hits these via public DNS (wheretolive.xyz) and reports back to
the Mac admin endpoint. See monitor.py for the runner.
"""

BASE_URL = "https://wheretolive.xyz"

# Each URL gets these check types:
#   - http              status code + load time (Playwright navigation)
#   - meta              SEO meta presence (title, description, canonical, og:*)
#   - selector          main content selector visible after load
#   - console           JS console.error count during load
#   - network           4xx/5xx XHR/fetch from the page
#   - lighthouse        full Lighthouse audit (public pages only)
#   - auth-gate         (auth-gated routes only) verifies login wall renders
URLS = [
    # Public pages — full audit.
    {
        "path": "/",
        "label": "homepage",
        "auth": False,
        "main_selector": "main, h1",
        "expected_status": 200,
    },
    {
        "path": "/evaluate?postcode=E14",
        "label": "evaluate-page",
        "auth": False,
        "main_selector": "main",
        "expected_status": 200,
    },
    {
        "path": "/check",
        "label": "check-form",
        "auth": False,
        "main_selector": "input, form",
        "expected_status": 200,
    },
    {
        "path": "/compare-input",
        "label": "compare-input",
        "auth": False,
        "main_selector": "main",
        "expected_status": 200,
    },
    {
        "path": "/property-compare",
        "label": "property-compare",
        "auth": False,
        "main_selector": "main",
        "expected_status": 200,
    },
    {
        "path": "/image-search",
        "label": "image-search",
        "auth": False,
        "main_selector": "main",
        "expected_status": 200,
    },
    {
        "path": "/auth/login",
        "label": "login",
        "auth": False,
        "main_selector": "form, input[type=email]",
        "expected_status": 200,
    },
    # Auth-gated — verify login wall, skip Lighthouse (wall page perf is misleading).
    {
        "path": "/map",
        "label": "map-loginwall",
        "auth": True,
        "main_selector": "main, body",
        "expected_status": 200,
    },
    {
        "path": "/chat",
        "label": "chat-loginwall",
        "auth": True,
        "main_selector": "main, body",
        "expected_status": 200,
    },
    {
        "path": "/value-property",
        "label": "value-property-loginwall",
        "auth": True,
        "main_selector": "main, body",
        "expected_status": 200,
    },
]

# Static endpoints (no rendering, just HTTP GET).
STATIC_PATHS = ["/robots.txt", "/sitemap.xml"]
