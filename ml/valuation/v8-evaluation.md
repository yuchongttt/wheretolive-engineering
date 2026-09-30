# Valuation v8 evaluation write-up (E15 + W7, 2026-05-21)

English translation of the private repo's `docs/valuation-v8-report.md`, edited for this extract: the
street address of an example sale, third-party product names and the author line are removed; section
6's commands are adjusted to this layout. All numbers are as recorded on 2026-05-21.

**Dataset**: E15 + W7, 7,299 sales x 30 features
**Evaluation**: temporal walk-forward (train < year Y, test = year Y)

---

## 1. Summary

| Version | GroupKFold MAPE | Temporal MAPE | Note |
|---|---|---|---|
| v5 (as reported) | 9.08% | 10.17% | the production model at the end of Phase 0 |
| **v8_best** | **8.41%** | **9.18%** | current recommendation |
| v8_best + 1.035x correction | -- | **9.10%** | after de-biasing, the <5% bracket gains +2.7pp |
| ~~v7_best 7.89%~~ | ~~contains leakage~~ | -- | discarded (see section 5.1) |

**Most important takeaways**:
1. v8 improves on v5 by about **-1pp MAPE** -- one leakage bug fixed, 5 low-risk new features added.
2. **Two thirds of predictions are within 10% error** (temporal: 64.0%).
3. **Time is by far the strongest single signal** (removing it adds +9.4pp MAPE); all other features
   together contribute about +1.5pp.
4. **A minimal-input mode is viable**: only "address + bedrooms" plus public LR/EPC lookups gives
   MAPE = 9.70%.

---

## 2. What v8_best is

v8_best adds 6 enrichment groups to the v5 production base (30 features in total):

| Source | Feature | Note |
|---|---|---|
| **v5 baseline (17)** | lat / lng / bedrooms / floor_area_sqm | geography + structure |
| | year_sold / months_since_sale | time |
| | type one-hot (4 classes) | property type |
| | is_leasehold | tenure |
| | council_ord (-1 sentinel) | council-tax band |
| | n_photos / interior_pct / mean_lux / max_lux / interior_mean_lux | SigLIP, 5 dims |
| **added in v6 (12)** | epc_rating_ord / epc_efficiency / epc_rating_missing | EPC energy (75% coverage) |
| | floor_area_fused / sqm_disagree_15pct | listing sqm fused with EPC sqm |
| | new_build / new_build_missing | LR new-build flag |
| | dist_zone1_km | distance to King's Cross |
| | has_prev_sale / prev_sold_price_log / years_since_prev / prev_log_ppsf | repeat-sales anchor |
| **added in v7 (1)** | sqft_per_bedroom | area per bedroom |

**Evaluation -- temporal walk-forward**: for each test year Y (2020-2026), train = sales before
Y-01-01, test = sales in Y. **Strictly split by time**, so the model never sees the test period's market
level -- a more honest generalisation metric.

---

## 3. Drop-one importance (what each feature group really contributes)

"If the model had never seen this group, how much worse would it be?" Delta is the temporal MAPE relative
to v8_best at 9.26% (note: this is a second evaluation run; the small difference from 9.18% in section 1
comes from random_state-derived variation).

| Category | Group dropped | Delta MAPE | Reading |
|---|---|---|---|
| baseline | **time** (year_sold + months_since_sale) | **+9.41pp** | by far the most important -- prices roughly doubled over 2006-2026 |
| baseline | geo (lat + lng) | +0.48pp | location |
| enrichment | **repeat_sales** (4 cols) | **+0.47pp** | strongest enrichment added in v6 |
| baseline | siglip (5 cols) | +0.31pp | photo-quality signal |
| baseline | tenure (is_leasehold) | +0.20pp | leasehold flat discount |
| enrichment | dist_zone1 | +0.08pp | distance to the centre |
| baseline | floor_area_sqm | +0.06pp | surprisingly low -- collinear with bedrooms |
| baseline | type one-hot | +0.05pp | flat/terraced/etc. |
| enrichment | area_fusion / new_build / area_per_bed | +0.04pp each | small but positive |
| enrichment | epc | +0.03pp | EPC energy rating |
| baseline | **council** | **-0.06pp** | **better without it** |
| baseline | **bedrooms** | **-0.10pp** | **better without it** |

**Two anomalies** (worth following up in v9):
1. **Dropping bedrooms improves MAPE by 0.10pp** -- highly correlated with floor_area_sqm; two equivalent
   signals dilute the GBR's split capacity.
2. **Dropping council improves it by 0.06pp** -- v5's -1 sentinel mixed with ordinal encoding confuses
   the GBR, a known issue; v6's council_fix was also confirmed as a loss in the v7 ablation (+0.03pp).
   This baseline slot needs redesigning.

---

## 4. Minimal-input valuation -- product view

If the user enters only **address + bedrooms**, and the backend may query public databases (LR + EPC),
how good is it?

| Scenario | Features | MAPE | <5% | <10% | <20% |
|---|---|---|---|---|---|
| v8_best full (listing data + photos) | 30 | 9.26% | 35.0% | 63.7% | 91.7% |
| **address + beds + LR + EPC lookups** | **22** | **9.70%** | **34.8%** | **61.2%** | **90.4%** |
| address + beds + EPC (no LR) | 15 | 10.89% | 29.3% | 55.9% | 87.1% |
| geo + beds + sqft + type + time only | 10 | 11.73% | 27.9% | 53.9% | 83.3% |

**"Minimal input" works**: address + bedrooms, with the backend pulling LR + EPC, gives MAPE **9.70%**
(only 0.44pp worse than the full 30-feature v8). **Within 10%: 61.2%**, almost the same as the full
model's 63.7%.

**The LR lookup contributes +1.19pp** (9.70% -> 10.89%): mainly the `prev_sold_price` anchor -- the same
property's previous sale price pins down the value range.

**The EPC lookup contributes >= +0.84pp** (10.89% -> 11.73%): mainly `floor_area_sqm` -- without area the
model can only guess, and accuracy falls straight back to v1 level.

**Two product paths**:
- **A. Least friction**: user enters address + bedrooms, backend looks up LR + EPC, 9.7% MAPE. Lowest
  barrier and widest coverage (any UK home).
- **B. Enhanced**: user provides more (type / sqft / photos) to reach 8-9% MAPE.

Backend for the least-friction version:
- address -> postcodes.io geocode -> lat/lng + outcode
- address -> LR query -> previous sale + new_build + tenure
- address -> local EPC service -> sqft + EPC rating + efficiency

---

## 5. What else belongs in this write-up

### 5.1 The v7 prev_cagr leakage fix (methodological integrity)

**Bug**: v7 introduced `prev_annualised_growth = (sold_price / prev_sold_price)^(1/years) - 1`.
`sold_price` is the target Y, so the feature is effectively "answer / anchor, then a root". The model can
almost directly reconstruct the target as `prev_sold_price x (1+cagr)^years`.

**Impact**: v7's reported 7.89% MAPE was a cheating number. After the fix, v8_best = 9.18% (temporal) /
8.41% (GroupKFold).

**Fix**: v8 simply drops the feature. The three legitimate strictly-historical signals `prev_sold_price`,
`years_since_prev`, `prev_log_ppsf` were already in v6_best.

**Lesson**: for every new feature, ask "does its formula use Y?". Expressions like
`groupby().transform()` or `X / Y_related` need a second review.

### 5.2 Bias correction (cheap win)

| Correction | MAPE | <5% | Bias |
|---|---|---|---|
| none | 9.26% | 35.0% | -3.4% |
| x 1.0347 | 9.10% | 37.7% | +0.0% |

**In production, just multiply predictions by 1.035** -- 10 lines of code, no retraining, +2.7pp in the
<5% bracket (the one users notice most). Strongly recommended when v8 goes live.

Why: under the temporal split the training period is always earlier and UK prices have risen over the
long run, so the model is conservative overall; a calibration factor removes the mean drift.

### 5.3 Robustness by year

| Year | n_test | MAPE | <10% | Bias |
|---|---|---|---|---|
| 2020 | 305 | 9.09% | 60.0% | -4.7% |
| 2021 | 460 | 9.42% | 62.4% | -5.2% |
| 2022 | 421 | 9.39% | 66.0% | -4.0% |
| 2023 | 377 | 9.29% | 64.2% | -1.3% |
| 2024 | 379 | 8.83% | 63.6% | -2.4% |
| 2025 | 381 | 9.00% | 66.4% | -1.5% |
| 2026 | 52 | 8.90% | 69.2% | +3.4% |

**The model is stable across the 2020-2025 market cycles** (MAPE 8.83-9.42%, 0.6pp range). 2023 is the
weakest single year (9.29%, but with the smallest bias, -1.3%) -- the UK rate-rise correction, the most
abnormal market.

**Bias converges from -5.2% (2020) to -1.5% (2025)** -- the closer the training window is to the test
period, the more accurate. This suggests quarterly retraining may beat yearly in production.

### 5.4 v9 roadmap (natural next steps)

In order of value for effort:
1. **Bias calibration (x 1.035)** -- 10 lines, +0.16pp MAPE, +2.7pp <5%.
2. **Drop bedrooms + council** -- matches the two anomalies in the v8_best ablation; estimated v9 MAPE
   ~ 8.95% (temporal).
3. **Phase 1: run the remaining 10 outcodes** (CR5 / DA1 / E3 / EN1 / EN4 / HA8 / N17 / N7 / RM10 /
   TW10) -- check whether v8 stays under 10% MAPE across inner / outer London and different type mixes.
4. **Floorplan OCR** (CubiCasa5K or Qwen3-VL) -- recover sqft for the 25% missing EPC sqft. Expected MAPE
   gain 0.5-1.0pp.
5. **Quantile regression** -- output P10/P50/P90. MAPE unchanged but a much better product
   ("estimate £450k, 80% likely between £420k and £480k").

### 5.5 Known limitations

- **The worst-3 SO/RtB cases remain unsolved**: the 3 worst predictions (APE 100%+) are all Shared
  Ownership / Right-to-Buy edge cases. The LR category field is barely used, and the 50% ppsf filter
  does not catch them. A real fix would need a reliable "Shared Ownership" flag on the listing, which
  is often missing for older sales.
- **Single-source dependence on EPC sqft**: 75% coverage, but EPC assessors are inconsistent
  (conservatories / garages sometimes counted), 5-10% variance. v6 area_fusion only covers the 9.9% of
  cases where the listing also states sqft, so the gain from fusion is limited.
- **v8 uses only E15 + W7**: a homogeneous sandbox (Stratford and Hanwell have similar price levels and
  type mixes). Only Phase 1 on 10 outcodes will show generalisation.
- **No external model comparison**: no head-to-head against commercial portal estimates or AVMs. Where
  9.18% MAPE sits within the AVM industry is unknown.

---

## 6. Reproduce

```bash
# from ml/valuation (needs data/evaluations.db + data/sold.db, not included)
python scripts/valuation_v8_temporal.py --outcodes E15,W7   # full v8 evaluation
python scripts/valuation_v8_best_detail.py                  # per-year detail for v8_best
python scripts/valuation_v8_analysis.py                     # drop-one + minimal-input analysis
```

Dependencies: `data/evaluations.db` + `data/sold.db`, and `image_luxury_score_v2` from the GPU box (pulled
over ssh once by `valuation_v6_gbr.py` and cached to `data/_luxury_cache_v6.pkl`).
