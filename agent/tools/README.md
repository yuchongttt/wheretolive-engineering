# agent/tools — the chat assistant's tool layer

The site's property assistant (bilingual EN/ZH) answers questions about London
areas, prices and listings. The LLM never touches the database directly: it
calls small, deterministic Python tools that run read-only SQL against a local
SQLite database and return text written for a model to read — numbers plus the
caveats needed to quote them correctly.

This directory is an extract: 10 of the 27 tool servers registered in
production at snapshot time, the two ways they are served, and their offline
tests. The harness that evaluates the agent lives in [`../evals/`](../evals/).

## How it works

```
               chat route (web app, not in this extract)
          ┌───────────────────────┴─────────────────────────┐
  default backend: Claude (Sonnet)           CHAT_BACKEND=qwen: Qwen 27B on vLLM
  = an MCP client                            = OpenAI-style tool calling
          │                                                  │
  mcp.json         one stdio server per tool   openai_tool_schemas.json  (tools=[…])
  mcp.single.json  mcp_all.py: every tool,     tool_runner.py <tool> '<json args>'
                   one process, one handshake    (imports the handler directly)
          └───────────────────────┬─────────────────────────┘
                                  ▼
      <tool>.py   list_tools() → name + JSON Schema
                  call_tool(name, args) → [TextContent]  (some append a "--- structured ---" JSON block)
                                  │  sqlite3.connect("file:…?mode=ro", uri=True)
                                  ▼
      $WTL_DATA_DIR/  evaluations.db · area_intel.db · london_outcodes.json
```

Every tool module has the same shape — a module-level `server = Server(...)`,
an async `list_tools` returning one `types.Tool`, and an async
`call_tool(name, arguments)` — so the same code can run as its own stdio MCP
server, inside the aggregate `mcp_all.py`, or be imported by `tool_runner.py`.

| File | Role |
|---|---|
| `mcp_all.py` | Aggregated MCP server; its module list is parsed from `mcp.json` (no second registry). |
| `tool_runner.py` | CLI invoker for the Qwen backend: `TOOL_MAP` name → handler, one JSON line out. |
| `openai_tool_schemas.json` | The same tools as OpenAI function definitions (filtered to this extract). |
| `area_scope`, `geo_radius`, `coverage_note`, `lr_category`, `council_tax`, `flood_area` | Shared helpers, each owning one trap (outcode vs sector parsing, mixed postcode storage, coverage, sale categories, borough rates, flood shares). |
| `radar_vocab.py` + `.json` | Shared condition vocabulary → parameterised SQL compiler, also used by saved searches ("radars") and listing search; the tools here use its coverage set and property-type families. |
| `pipeline/` | Helpers shared with the site's batch jobs: `market_baselines` (distorted-month detection, seasonality), `ptype_kind`, `outdoor_space`, garden helpers. |

## The tools

| Tool | Answers | Guard visible in the code |
|---|---|---|
| `get_postcode_scores` | 5-dimension livability score | An outcode/sector returns median **and range** over its unit postcodes; a legend defines each dimension (Price = asset quality, not value for money). |
| `compare_postcodes` | 2–4 areas side by side, flood share, council tax | Sold median vs asking mean labelled as different markets; straddling boroughs hedged. |
| `get_area_overview` | One-call area summary incl. rent/yield | Counts name their scope; rent figures carry their snapshot date. |
| `screen_areas`, `screen_by_commute` | Screeners over precomputed scores / sector→hub commutes | Commutes stated as sector-centroid approximations. |
| `get_price_trend` | Repeat-sales CAGR, unit → London | <8 pairs widens to sector/outcode and says so; CAGR definition ships with every number; stale-table check. |
| `get_sold_nearby` | Recent sales; "did buyers from period X gain?" | Narrowest adequate rung (~800 m → sector → outcode); honesty tiers name what the headline covers; <3 pairs → "insufficient data". |
| `get_comparables` | Land Registry comparables | No £/m² or bedroom claims (LR has neither); category-B sales excluded but counted. |
| `get_market_risk` | Share of listings asking below the owner's purchase price | n<10 → refuses to quote a rate; cohort window stated; asking ≠ achieved. |
| `compute_buying_costs` | Stamp duty, max affordable price, monthly cost | Deterministic arithmetic, not LLM arithmetic; rates carry an as-of date and escalate to "verify" after 200 days. |

## Design decisions (visible in the code and tests here)

- **stdout is the protocol.** Logs go to stderr; one stray `print` corrupts the
  JSON-RPC stream — and once aggregated, every tool with it.
  `test_mcp_stdout_purity.py` checks three ways: static scan, import-time
  subprocess, and a live handshake validating every stdout line.
- **Aggregation without shared fate** (`mcp_all.py`). Its docstring (2026-07-13)
  names 24 interpreter cold starts per turn as one of the largest parts of a
  p50 time-to-first-byte of ≈22 s. The aggregate imports each module in
  isolation (a broken module loses only its own tools) and dispatches calls via
  `asyncio.to_thread`: the handlers are async-signed but block on sqlite, so
  awaiting them on the loop would serialise the model's parallel calls.
- **Test the real entrypoint.** The first aggregate ran `asyncio.run()` inside
  a running loop; the per-module `try/except` swallowed the error and the server
  exposed **0 tools** from 2026-07-13 to 2026-08-18 while the agent kept
  answering fluently. `_load_all()` now refuses to run inside a loop, and
  `test_mcp_all_entrypoint.py` spawns the process and counts tools over stdio.
- **Pin the decorator, not the function.** `@server.call_tool()` registers the
  function directly below it; a helper inserted there made one tool fail 100%
  for 18 days with its unit tests green. `test_mcp_handler_registration.py`
  AST-checks every server module.
- **Read-only, trust at the edge.** Every connection a tool opens is
  `file:…?mode=ro`; the tools do no auth — the chat route is the trust boundary.
  A tool is only usable if registered (`mcp.json`), allowlisted in the route and
  described in both zh and en prompts; missing any one fails silently
  (`test_chat_tool_registry_sync.py` covers the part that lives here).
- **No coverage ≠ no data.** Outside the 20 covered postal areas, tools say "we
  can't see this area", not "no sales" (`coverage_note.py`); a full postcode
  with no trace anywhere gets a loud alert before area aggregates can make it
  look real (`geo_radius.unit_postcode_known`).
- **Thin samples are reported, not quoted**: flood shares need ≥20 checked
  points, by-type medians ≥5 sales, CAGR ≥8 pairs, cohorts ≥3 pairs,
  market-risk rates n≥10.
- **Pointers in first-hop output.** Where a better tool exists the output names
  it (e.g. `get_market_risk` → `fetch_listing` for the seller's purchase price).
- **Bilingual.** Descriptions and aliases deliberately contain Chinese trigger
  phrases (`'SW11 值得买吗'`, `全伦敦`, `站`) and `radar_vocab.json` has Chinese
  UI labels; these are functional and stay untranslated.

## Running locally

Python 3.10+, the `mcp` Python SDK 1.x and `pytest` (`pip install -r requirements.txt`). The servers use the 1.x low-level `Server` decorator API.
The tools read `$WTL_DATA_DIR` (default `./data` here), built by the site's data
pipeline (not in this extract; the test fixtures show the columns each tool reads):

- `evaluations.db` — `postcode_scores` + `dim_*` scores; listings
  (`rm_sales_overview` with a generated `postcode_norm`, `rm_delisted_properties`,
  `rm_nearest_stations`, `price_history`); HM Land Registry `lr_transactions` and
  its derivatives (`v_resale_outcomes`, `geo_cagr`, `semantic_meta`,
  `lr_month_quality`, `lr_seasonality`, `v_lr_clean`); `postcode_coords`,
  `outcode_borough`, `sector_hub_commute`.
- `area_intel.db` — `council_tax_rates`; `london_outcodes.json` — rent/yield precompute.

```bash
cd agent/tools && export WTL_DATA_DIR=/path/to/data
python tool_runner.py get_postcode_scores '{"postcode": "SW11"}'   # one call, JSON out
python mcp_all.py                                                  # stdio MCP server, all tools
```

Or point an MCP client at `mcp.single.json` (one server) or `mcp.json` (one per
tool), launched from this directory.

| Env var | Purpose |
|---|---|
| `WTL_DATA_DIR` | Directory holding the databases (default `./data` next to the code). |

No secrets are needed: every tool here is local and read-only.

## Tests

```bash
cd agent/tools && python -m pytest -q      # 271 passed (2026-09-30)
```

Fixtures are temporary or in-memory SQLite; no network, no production data.
`tests/conftest.py` stubs `mcp` for pure-function tests, but the subprocess
tests need the real SDK. Tests that need the production database or modules
outside this extract were left out; each trimmed file says so in its docstring.

## Not in this extract / known gaps

- Sibling tools named in descriptions (`search_properties`, `fetch_listing`,
  `get_commute`, `get_area_profile`, `lookup_address`, `create_radar`, image
  tools) depend on modules or services outside this extract.
- `get_price_trend(by_type=true)` needs the site's repeat-sales engine
  (`price_analysis_evaluator.py`); here it raises `ImportError`.
- Inherited drift: `openai_tool_schemas.json` advertises `compare_postcodes`,
  `get_price_trend` and `screen_areas`, but `tool_runner.TOOL_MAP` does not route
  them, so on the Qwen path they return `unknown tool`. The registry-sync test
  covers only the MCP path.
