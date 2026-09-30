# ml — applied ML experiment logs

Two tracks of applied ML on London property data, written up as experiment logs: question, data,
validation, results (with date and source file) and what shipped. Scripts read the production SQLite
databases, which are not included; tests run offline on synthetic data or on recorded results.

| Track | Question | Outcome |
|---|---|---|
| [valuation/](valuation/) | Predict a London sale price from public records + listing data | v8 GBR (temporal MAPE 9.18%, 2 outcodes) is used as one input to the site's valuation; v9–v12 never got recorded results |
| [floorplan/](floorplan/) | Read which way is north from a floorplan's compass symbol, then derive garden / room orientation | north-angle lane shipped (76.3% yield @ 100% precision, pre-registered double-blind); garden side passed its gate; window-to-wall binding missed its bar and did not ship |

## Track 1 — valuation models (May 2026)

**Data.** Sales joined from HM Land Registry price-paid data, EPC certificates (floor area, energy
rating), listing attributes and SigLIP photo scores computed on the GPU box. Sandbox: two outcodes (E15,
W7), ~7.3K sales; later 20 outcodes. Tables read: `sold.rm_sold_properties`, `sold.rm_sold_transactions`
(`data/sold.db`, attached as `sold`), `lr_transactions`, `lr_repeat_sales` (`data/evaluations.db`), and
`image_luxury_score_v2` on the GPU box (pulled once over ssh by `valuation_v6_gbr.py`, then cached).

**Progression.** Metric = MAPE on sales from 2020 on. ★ = script included here.

| Ver | Model / change | Validation | Data | MAPE |
|---|---|---|---|---|
| v1 | spatial kNN: 20 nearest same-type comps, local CAGR time adjustment | leave-one-out | E15 | 22.2% |
| v1.1 | + 2006+ pool, recency weight exp(−years/τ) | leave-one-out | E15 | 16.5% |
| ★ v1.2 | + drop comps below 0.5× local £/m² (shared ownership / right-to-buy proxy) | leave-one-out | E15 | **14.5%** (baseline) |
| v2 | GBR on log(price) + bathrooms, council band, SigLIP photo features | 5-fold, row-level | E15 | 11.25% |
| v3 | + the £/m² filter applied to the training pool | 5-fold, row-level | E15 | 9.03% |
| ★ v4 | folds grouped by property + area/price hard caps | GroupKFold by property | not stated | 9.37% (went **up**: leak removed) |
| v5 | + leasehold flag, − bathrooms (65% coverage, imputation faked a signal) | GroupKFold | E15+W7 | 9.08% |
| ★ v6 | + 7 enrichment groups (repeat-sale anchor, EPC, new-build, …) + ablation harness | GroupKFold | E15+W7 | 8.47% |
| ★ v7 | 16 candidate features, 40+ combination battery | GroupKFold | E15+W7 | 7.89% — **invalid, target leak** |
| ★ v8 | leaky feature dropped, + area/bedroom; temporal walk-forward added | GroupKFold / temporal | E15+W7 | **8.41% / 9.18%** |
| v8 min-input | address + bedrooms + LR/EPC lookups only (22 features) | temporal | E15+W7 | 9.70% |
| v9–v12 | LightGBM+Optuna; OOF target encoding; recency kNN + local HPI; 3-model stack | GroupKFold + temporal | 20 outcodes | **not recorded** |
| ★ shipped | v8_best refit on all sales of 20 outcodes (`predict_one_v8.py`) | in-sample only | 20 outcodes | 13.9% in-sample |

Sources: v1–v5 and v6/v7 from the project's technical notes dated 2026-05-21; v8 rows from
[`v8-evaluation.md`](valuation/v8-evaluation.md) (2026-05-21, which also has per-year, drop-one and
bias-correction tables); v9–v12 note (2026-05-22) records only a target of 6.5–7.5%; the shipped label is
hard-coded in the valuation API route. Script choice: the baseline, each change of validation scheme (v4
grouping, v8 temporal), the ablation harness (v6), the leak (v7, also the shared library for v8), and the
shipped entry point. v2/v3/v5 are small steps between these; v9–v12 are left out because no result was
ever recorded and their entry points reference a `SOLD_DB` name they never define, so they stop before
loading data.

**Validation decisions**
- *Group by property* (v4). Row-level folds can put a flat's 2018 sale in training and its 2023 sale in
  test, so the model can memorise the flat. v4 (grouping + hard caps) moved MAPE from 9.03% to 9.37%; the
  worse number is the honest one.
- *Temporal walk-forward* (v8). For each test year Y (2020–2026): train on sales before 1 Jan Y, test on
  Y. GroupKFold still lets the model see other properties' sales from the test year (the market level).
  Gap on v8_best: 8.41% → 9.18%.
- *Leak audit* (v7 → v8). `prev_annualised_growth = (sold_price / prev_price)^(1/years)` contains the
  target, so the model can rebuild the price. The 7.89% was withdrawn and the lesson written down: for
  every feature, ask whether its formula touches Y.
- *Features from strictly earlier data.* Repeat-sale anchor = previous sale of the same property
  (`shift(1)` after sorting by date); street £/m² and kNN £/m² use only earlier sales. The new tests check
  the kNN feature and both split schemes (with a spy in place of the model).

**Limitations (found in the code, not only the write-ups)**
- The kNN leave-one-out lets comparables sold *after* the subject in (price adjusted backwards), so the
  v1.x numbers are not temporal.
- Pool-level steps run before the split: the £/m² outlier filter uses (type, 5-year) medians of the whole
  pool, and it also removes low-priced test sales, so the metric excludes shared-ownership sales by design.
  Missing-value medians and the kNN feature's no-comparable fill are also pool-wide (the notes estimate
  the cross-fold kNN leak at ≤ 0.1pp; the others are not measured).
- Two outcodes are a homogeneous sandbox; there is no held-out 20-outcode number and no external AVM
  comparison.
- `predict_one_v8.py` builds the target row as a leasehold flat with type key `"F"`, which matches no
  training bucket (all type one-hot columns become 0), and the API does not pass the previous sale, so
  the repeat-sale anchor is unused in production. The recommended ×1.035 bias correction is not applied.

**What shipped.** `/api/value-property` runs `predict_one_v8.py --json` as a subprocess (45 s timeout) and
passes the price plus per-group counterfactual contributions to Claude (Sonnet), labelled "a sanity
bound, not gospel", next to comparable sales. The mortgage panel's down-valuation flag does *not* use the
model: running it per request is too slow for that code path, so it uses a bedroom-controlled peer median,
threshold 56% (calibrated 2026-06-06 on 1,722 listings; far above v8's ~10% error), enabled only in v8's
20 outcodes.

## Track 2 — floorplan orientation (Sep 2026)

Full logs: [compass](floorplan/research/compass-orientation/README.md) ·
[garden side](floorplan/research/garden-facing/README.md) · [room windows](floorplan/research/room-aspect/README.md).

```
floorplan ─► VLM detect (Qwen, bbox 0-1000) ──────── gate 1: compass present?
          ─► 320px white-background crop
          ─► compass-ness CNN (ResNet18, 4×90° TTA) ─ gate 2: p ≥ 0.4
          ─► reader CNN (ResNet18 → sin/cos, 4×90° TTA) gate 3: rotation spread ≤ 10°
          ─► θ + 8-way label ("A|B" within ±7° of a bin edge)
property  ─► all readings pairwise ≤ 15°, else abstain
```

**Design.** Abstain first: a wrong orientation is usually a 180° flip, worse than no answer. The reader is
a small CNN trained on 170–233 real compass crops with synthetic rotations (label rotated with the image).
At inference the crop is rotated 4× by 90°; a correct reading rotates with it, so the disagreement between
the four un-rotated readings (spread) is the abstain signal. The VLM (local Qwen) only finds the box;
reading the angle directly topped out at ~60%. An LLM was used offline for blind labelling and QA,
not for per-image inference.

| Step (date) | Eval set | Result |
|---|---|---|
| v2.2 rule pipeline (09-01) | batch4, double-blind | 36% yield @ 83% (earlier: 95% on calibration batch1 → 68% on blind batch3) |
| Reader CNN, train batches 1–3 (09-02) | batch4, 64 real crops | acc@22.5° 92.2%; gate ≤15° → 82.8% yield @ 100% |
| Reader, 4-fold leave-one-batch-out (09-02) | 233 crops | acc@22.5° 94.8%, 1 flip; gate ≤10° → 62.2% @ 100% |
| Production lane end-to-end (09-02) | batch5: 120 floorplans, pre-registered gate ≥50% × ≥95% | **76.3% yield @ 100% (58/58)**; 0 of 44 no-compass images emitted |
| Garden side, G1 + eq4 (09-03) | batch2, 50, pre-registered ≥90% | 46% yield @ 96% (22/23); constant "top" baseline 64% |
| Room windows, best ensemble (09-03) | 50 plans / 318 rooms | 76.3% precision @ 29% yield — below the 90% bar; not shipped |

**Evaluation discipline.** Predictions are exported before labelling; labellers (context-free Claude
sub-agents working from the image only, with pixel measurements) cannot see them. Gates are frozen after
the calibration batch and written down (`garden-facing/spike50/PREREGISTERED.md`) before the confirmation
batch is sampled. Garden accuracy is compared with the trivial "always top" baseline (88% of batch1
gardens are drawn at the top of the page). The recorded fold results are in `reader/results/`, and a test
re-derives the quoted numbers from them. Limits: ground truth is from LLM agents (spot-checked by eye), not human annotators; images were split
between labellers rather than double-labelled, so there is no inter-annotator agreement figure; only 7
non-top gardens were emitted across both garden batches.

**What shipped.** The north lane went on a 30-minute launchd schedule on 2026-09-02
(`enrich_floorplan_north.py`, not included) and writes `floorplan_north` / `property_north`. The
garden-side route moved into a production lane (not included) and agreed with agent-stated orientation
on 58/60 live listings; both disagreements were checked by eye and were errors in the agent's text.
Room-window detection did not ship; the log recommends a small room-level model instead.

## Layout

```
valuation/   scripts/ (8 scripts, ★ above) · v8-evaluation.md · tests/ (harness + leakage checks, new)
floorplan/   scripts/lib/ (detect, north, presence, reader model) · scripts/train_compass_*.py
             research/ compass-orientation/ · garden-facing/ · room-aspect/ (protocols, scorers, results)
             tests/ (4 source test files + a recorded-results test, new)
```

## Running the tests

```bash
cd ml && python -m pytest -q
```

Needs `numpy`, `pandas`, `scikit-learn`, `torch`, `torchvision`, `Pillow`, `pytest` (`scipy` for the CV
locator; `lightgbm`/`optuna` are not needed). Model weights (`*.pt`), images and databases are not in git,
so the weight smoke tests are not included. Scripts expect `data/evaluations.db` and `data/sold.db` under
`valuation/` or `floorplan/`.

## Environment variables

| Var | Used by | Meaning |
|---|---|---|
| `GPU_HOST` | valuation v4, v6 (v7 only reads the cache) | ssh target holding the SigLIP score dataset (default `gpu-box`) |
| `VLM_URL` | `lib/compass_detect.py`, garden route | OpenAI-compatible VLM endpoint (default `http://gpu-box:8200`) |
| `COMPASS_EXCLUDE_SEEDS` | compass training scripts, `run_lobo.sh` | comma-separated seed ids to drop (default none) |
| `PYTHON` | `run_lobo.sh` | interpreter (default `python3`) |
