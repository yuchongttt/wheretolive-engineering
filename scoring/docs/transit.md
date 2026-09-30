# Transit, commute and long-distance access

Condensed from the original design notes (line weights, distance coefficient, station
de-duplication, long-distance scoring, commute guide) and checked against the current
code. Parts of those notes described designs that were later replaced (for example a
"diminishing returns" line weighting and 13–20-point line weights); only what the code
does today is kept here.

## Rail-station score (`transit_convenience_evaluator.py`)

This is the `transit` input of the transport dimension. In production (`--free`) it
uses the TfL path:

1. **Find stations.** TfL `StopPoint` search within 1,200 m for metro and rail stops
   (tube, DLR, Overground, Elizabeth line, National Rail), each with its lines.
2. **Walking time.** Straight-line distance × 1.3 (street-network detour factor) at
   5 km/h (83.33 m/min).
3. **Merge duplicates.** Entries whose names normalise to the same station
   ("X Underground Station", "X DLR Station", "X") are merged; their lines are unioned
   and the shortest walk is kept.
4. **Drop redundant stations.** Nearest first, a station is kept only if it adds at
   least one line not already served by a nearer one; at most 3 are kept. A second
   entrance to the same lines does not count twice.
5. **Score each station.** `min(100, Σ line weights × distance coefficient)`.
6. **Combine.** `min(100, best station + 0.1 × Σ next three stations)` — the best
   station dominates; alternatives add a little.

**Line weights** (points per line; summed, no discount for extra lines — the cap is
applied after the distance coefficient, so two top-tier lines close by give 80 and
three give 100):

| Weight | Lines |
|---|---|
| 40 | Central, Northern, Elizabeth, Victoria, Jubilee |
| 35 | Piccadilly, District, Metropolitan, Circle |
| 30 | Bakerloo, Hammersmith & City, DLR |
| 25 | Overground (all six named lines) |
| 20 | Waterloo & City, National Rail, any unknown line |

**Distance coefficient** (walking minutes → multiplier):

| Walk | Coefficient | Rationale from the design notes |
|---|---|---|
| ≤ 5 min | 1.0 | ~400 m, "on the doorstep" |
| 5–10 min | 1.0 → 0.7 linear | ~830 m at 10 min, fine for daily use |
| 10–15 min | 0.7 → 0.4 | ~1.25 km, the edge of convenient |
| 15–20 min | 0.4 → 0.2 | most people would take a bus or cycle |
| > 20 min | −0.01/min, floor 0.05 | |

With `--all` and a Google key, stations come from Google Places instead. Each
candidate's lines are looked up by name through TfL (non-rail results are dropped), and
stations less than 400 m apart are treated as one interchange: the higher-scoring one
is kept, greedily in score order (a name comparison is used when coordinates are
missing). The notes chose 400 m because the buildings and entrances of a large
interchange are typically 200–400 m apart, while separate stations nearby are 500 m or
more apart. That path adds a diversity bonus of 2 points per extra station (max 10).
Both de-duplication paths are covered by `tests/test_station_deduplication.py`.

## Commute

- **Point-to-point (A).** TfL Journey Planner, departing the next weekday at 09:00,
  modes `tube,dlr,overground,elizabeth-line,national-rail,bus,walking`
  (`apis/tfl.DEFAULT_JOURNEY_MODE`). National Rail must be in that list: the module
  comment records that without it, outer-London journeys were routed over buses (one
  test route went from 62 to 95 minutes). `tests/test_tfl_journey_modes.py` pins this.
  TfL sometimes answers HTTP 300 with a list of candidate stops; the client picks the
  best `matchQuality` option and retries once (`tests/test_tfl_disambiguation.py`).
  Outside London, or if TfL fails, Google Routes is used when a key is configured
  (morning 60% / evening 40% from `commute_config.json`).
- **Hub accessibility (B).** Weighted average from the postcode sector to 20 employment
  hubs, precomputed offline into `sector_hub_commute` (see METHOD.md §2).
- The streamed `hub_commute` block (live average to 5 central hubs, cached for 7 days)
  is returned in the output but is not a scorer input.

## Long-distance travel (legacy layer only)

`long_distance_travel_evaluator.py` scores access to the 5 London airports and main
termini (local list in `data/london_transport_hubs.json`). It needs the paid Google APIs,
so it is disabled with `--free` and does not feed the published score. For each
destination it takes the better of transit or driving, with the reference time set to
next Saturday 12:00, when long trips usually start. The score is
`importance × time coefficient × 100`, airports 60% and stations 40%. The time
coefficients are stricter than for commuting because luggage and early flights matter:

| Destination / mode | Full marks up to | 0.7 at |
|---|---|---|
| Airport, transit | 30 min | 60 min |
| Airport, drive | 20 min | 40 min |
| Station, transit | 20 min | 40 min |
| Station, drive | 15 min | 30 min |
