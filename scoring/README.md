# scoring — London postcode scoring engine

Scores any London postcode on five dimensions (transport, community, environment,
price, schools) from public data sources, then calibrates every dimension, and the
total, onto one stable scale: **65 = London median, SD 15**. This is the engine behind
the neighbourhood scores on [wheretolive.xyz](https://wheretolive.xyz). The site is
bilingual (EN/ZH), which is why some output fields carry a `*_zh` label.

Full method: [docs/METHOD.md](docs/METHOD.md). Transit, commute and long-distance
detail: [docs/transit.md](docs/transit.md).

## How it works

```
postcode ─► validate + geocode (postcodes.io)
            │
            ├─ thread: commute · transit · crime · prices    (evaluator.py collectors)
            ├─ thread: demographics / IMD                     ─┐
            ├─ thread: schools                                 │ each result is streamed
            ├─ thread: flood · air · parks                     │ as one JSON line as soon
            ├─ thread: noise                                   │ as it is ready
            └─ thread: hub commute                            ─┘
            │
            ▼  raw data per dimension (cached in SQLite, per-dimension TTL)
     SimpleScorer: raw 0-100 ─PCHIP─► London percentile ─probit─► N(65,15)
                   weighted mean of calibrated dims ─► "total" calibration ─► score + rating
```

| Dimension (weight) | Inputs | Sources |
|---|---|---|
| Transport (20%) | commute percentile (60%: point-to-point to a central destination blended with sector→20-hub access); nearby rail stations × line weights × walking distance (40%) | TfL Unified API (Journey Planner, StopPoint) |
| Community (25%) | LSOA crime rate per 1,000 residents, London percentile (40%); IMD 2025 decile (60%) | data.police.uk (OGL), Census 2021 (OGL), English Indices of Deprivation 2025 (OGL), postcodes.io (ONS/OS data, OGL) |
| Environment (20%) | road/rail/airport noise, flood risk, air quality, green space (25% each) | DEFRA noise maps Round 4 (OGL), Environment Agency NaFRA2 (OGL), LondonAir/LAQN (Imperial College ERG API), OS Open Greenspace (OGL), OpenStreetMap Overpass fallback (ODbL) |
| Price (25%) | price level, 10/5/3-year repeat-sales growth, return volatility, market activity | HM Land Registry Price Paid Data (OGL), EPC register (floor area, via a local service) |
| Schools (10%) | Ofsted ratings within 1.5 km, distance-weighted | DfE GIAS + Ofsted management information (OGL) |

Google Routes/Places/Geocoding are optional (paid, `--all`). The site runs `--free`,
which uses TfL inside London and skips the Google-only sub-dimensions.

## Calibration, and why

Each dimension has its own formula, so raw scales are not comparable. With the current
breakpoints the median London postcode has a raw transport score of 31.6 and a raw
schools score of 85.6. Any formula change also moves the whole distribution. So every
raw score is mapped through 22 empirical `(raw, percentile)` breakpoints, using PCHIP
(monotone like linear interpolation, but without a slope jump at every breakpoint, and
unlike a plain cubic spline it cannot overshoot), and then through an inverse normal
CDF onto N(65, 15).

What this buys:
- **Comparable across dimensions.** A 75 in schools and a 75 in transport both mean
  "about the 75th percentile of London".
- **Stable across scorer versions.** When a formula changes (e.g. v9 swapped the crime
  signal), `recalibrate_all.py` re-derives the breakpoints from the scored population,
  so the published distribution, and the meaning of a number, stay put.
- **A total with real spread.** Averaging five calibrated scores shrinks the SD (the
  code comment says to ~7.5), so the weighted mean is calibrated once more. The rating
  is then just `Better than Φ((s−65)/15) of postcodes`.

## Design decisions and trade-offs

- **Cache raw data, not scores.** Per-dimension SQLite tables with TTLs (commute 7 d,
  transit/safety/schools 30 d, demographics/price 90 d; flood 90 d, air 365 d). The key
  includes `DATA_VERSION` + `SCORER_VERSION`. On a cache hit the scorer re-runs on the
  cached raw data (pure arithmetic), so a scoring change takes effect immediately and
  does not trigger a full re-fetch.
- **Failed is not the same as empty.** Output separates `_failed_dims` (the call failed),
  `_no_data_dims` (it answered with nothing) and `_paused_dims` (a paid API hit its
  monthly quota). Failures are never cached. A missing dimension's weight is
  redistributed instead of it being scored as median, and with fewer than 3 dimensions
  there is no total. Until 2026-09-27, refused flood-map requests were cached as "very
  low risk"; `tests/test_flood_wms_errors.py` now guards against that.
- **Offline data where the live API misleads.** The v9 crime signal uses a per-capita
  LSOA rate from an offline police.uk + Census build; the code records that the old
  live-API count correlated with it at Spearman 0.544 only. The live API remains the
  fallback. Parks use OS polygons (distance to the nearest edge, area-weighted) instead
  of counting OSM centroids.
- **Free by default, paid on purpose.** `--free` uses only free APIs. Paid Google calls
  are gated by `api_limits` in `config.json`.
- **Honest limitations.** The legacy `evaluator.py` layer (inherited from an earlier
  design) still computes a location/environment/economic score with several
  placeholder sub-scores; that score is only printed in the text report and never
  published. Air quality is London-only (2023 LAQN annual means). The sector→hub
  commute table, the offline crime database and the Land Registry bulk table are
  built by jobs outside this module; without them the scorer falls back as described
  in METHOD.md.

## Layout

```
simple_scorer.py                 5 dimensions + PCHIP calibration (the published score)
evaluate_single.py               CLI orchestrator: threads, streaming JSON, cache, ranking
evaluator.py                     legacy collector layer ─► location_convenience.py,
                                   surrounding_environment.py, economic_factors.py
transit_convenience_evaluator.py rail-station score (line weights, walk coefficient, de-dup)
long_distance_travel_evaluator.py airports / termini (Google-only, not in the published score)
amenities_evaluator.py           Google Places amenities (paid, legacy layer)
area_demographics_evaluator.py   IMD 2025 + Census 2021
school_evaluator.py, school_geo.py   GIAS + Ofsted
noise_evaluator.py, environment_evaluators.py, parks_index.py   noise, flood, air, parks
price_analysis_evaluator.py      Land Registry repeat sales, CAGR, EPC £/m²
recalibrate_all.py               re-derive CALIBRATION_BREAKPOINTS from the cache
apis/                            thin clients: TfL, police.uk, postcodes.io, Land Registry,
                                   EPC, ONS, IMD, Google; api_usage.py = monthly quotas
cache/                           versioned per-dimension SQLite cache + ranking
config.json, commute_config.json, property_params.example.json, data/london_transport_hubs.json
```

## Run one postcode

```bash
pip install -r requirements.txt            # requests numpy scipy shapely python-dotenv (+ pytest)
cp property_params.example.json property_params.json
export TFL_APP_KEY=...                     # optional, raises TfL rate limit
python3 evaluate_single.py "EC4M 8AD" --free --json --stream   # what the site runs
python3 evaluate_single.py "EC4M 8AD" --free                   # plain-text report
python3 evaluate_single.py --list-dimensions
```

Run from this directory: the configs are read relative to the working directory.
Local data (optional; missing pieces degrade gracefully, and the dimension is marked
failed or falls back): `data/edubasealldata.csv` + `data/ofsted_ratings.csv` (schools),
`data/london_parks.json` (OS Open Greenspace polygons), `data/area_intel.db` (offline
crime rates), `data/evaluations.db` (created on first run; holds the caches). The IMD 2025
CSV is downloaded from gov.uk on first use.

| Env var | Purpose |
|---|---|
| `TFL_APP_KEY` | TfL app key (optional; unauthenticated works, with a lower rate limit) |
| `GOOGLE_API_KEY` | enables the paid Google Routes/Places/Geocoding paths (optional) |
| `EPC_SERVICE_URL` | local EPC lookup service for £/m² (default `http://localhost:8400`; price scoring works without it) |
| `EVALUATION_DB_PATH` | override the per-dimension cache DB (default `data/evaluations.db`) |

## Tests

```bash
python -m pytest -q        # offline: no network, no production DB, no secrets
```

The tests cover calibration helpers and the community, environment and price
sub-scores, the walking and travel-time coefficients, line weights, station
de-duplication, the police.uk query polygon, TfL client behaviour (app key, UA,
HTTP 300 disambiguation, default modes, all mocked), NaFRA2 failure handling,
repeat-sales selection, the EPC client, cache key normalisation and the school
geo helpers. Diagnostic scripts that call live APIs (TfL, Google Places,
data.police.uk, full evaluations) were left out of this copy.
