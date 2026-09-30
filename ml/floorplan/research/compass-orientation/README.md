# Floorplan compass -> page north angle: research log (2026-09-01 to 09-03)

English translation of the private research README plus the fifth-batch result note, lightly edited
for this extract: listing IDs, hosts and data-source details are removed; files that are not in this
extract are marked *(not included)*. Numbers are as recorded on the dates shown.

A one-day technical validation, from "can orientation be read off a floorplan at all" to a frozen
pipeline and four double-blind evaluation batches. The product question was whether search and saved
searches could filter by orientation (e.g. north-facing gardens). The production alternative already live
at the time was the **agent-reported garden facing** (`garden_facing` column); this work validates the
next route: inferring orientation from the floorplan.

## Final conclusions (numbers only from new double-blind batches; calibration-set numbers don't count)

| Fact | Number | Source |
|---|---|---|
| Floorplans with an explicit compass | 56-67% (reproduced over four batches) | batch1-4 detection |
| Local VLM (Qwen3.6-27B) reading the crop directly | ~60% (ceiling) | reproduced on batch2/3 |
| v2.2 frozen pipeline (multi-window agreement + veto gates) | **36% yield @ 83% precision of emitted readings** | batch4 double-blind |
| Excluding "convention-ambiguous symbols" flagged in advance by the blind labeller | **19/20 = 95%** | batch4 |
| All-letter geometric cross-check arbiter, strong tier | 100% (9/9), but it fires on only 7% | batch2+3 |
| **Reader CNN spike (small CNN trained on synthetic rotations of batch1-3 seed crops)** | **batch4 real crops: acc@22.5 92.2%, median error 4.2 deg, 0 flips; equivariance gate <=15 -> 82.8% yield @ 100%** | batch4 double-blind (2026-09-02, below) |

**Core insight**: most remaining errors are not pipeline misreads but **symbols that cannot be resolved**
(a fully symmetric star with an upright N: the N's 180-degree symmetry means even a human can only guess
from the trade convention "the one filled point = north"). The production answer is to **abstain** on
these detectable symbols, not to guess.

## Dead routes (do not revisit)

1. VLM reading the angle / the letter's clock position directly: 180-degree flips plus a default-to-12
   noise; two prompt styles contradicted each other.
2. Generic rotation-search template matching: 5/32 -- scale not searched, 8-fold symmetry of roses, false
   peaks across styles; structurally dead.
3. Generic letter anchoring: where the N sits changes meaning by style (around / centre / at the tip);
   there is no style-independent anchor.
4. Pure-CV single-letter arbiter (no OCR): 41%, a coin toss (N/S half each).
5. Rotation-consistency gate with the local model: after rotation it refused on a large scale; pair
   completion rate 10%.
6. **Same-modality self-consistency is not independent verification**: a self-consistent systematic flip
   (wide-wedge dart symbols) passes every consistency gate; the collapse from 95% on the calibration set
   to 68% double-blind is this lesson. Only trust accuracy reported on a double-blind new batch.

## Components that worked (v2.2 frozen pipeline: `pipeline_v2.py` + `compass_letter.py`, *not included*)

- **Locator v2** (`scripts/compass_locate.py`): CV isolated ink clusters on white margins (drop long
  structures -> merge neighbours -> rank by isolation) unioned with the VLM bbox; cut zero-reading cases
  from about half to a third.
- **Two-scale framing agreement gate** on the same area (crops at r and 1.8r; emit only if both agree)
  plus a veto on the default "12 o'clock" reading.
- **All-letter geometric cross-check arbiter**: N-S diametrically opposite / N->E clockwise 90 degrees
  must be self-consistent -> the strong tier is 100%; hallucinated letters fail the cross-check. The weak
  tier was 67% and dropped.
- Rose ring-fit support (`scripts/compass_extract.py`, `_rose_method`).

## Next steps as decided late on 2026-09-01

**Reader bench**: a Claude sub-agent reading the batch4 crops blind, purely visually = **33/36 = 92%** of
the crops it answered (local Qwen: 60%); 2 of its 3 errors were items it had itself flagged as uncertain
beforehand (~97% with a confidence field); its 23 NONE answers were all honest abstentions on bad crop
windows (a localisation problem, not a reading problem).

**Production route**: a hosted LLM was ruled out for per-image production inference -- at a projected
~1,000 listings/day it does not fit. An LLM is used offline only, to label the seed and evaluation
sets and for QA; the production reader is a **self-trained small local model**.

1. ~~Small CNN trained on synthetic rotations~~ **done as a spike on 2026-09-02 (see "Reader spike")**:
   batch4 double-blind 82.8% @ 100% (equivariance gate <=15). Remaining for production: rejecting
   non-compass symbols, confirming with four-fold cross-validation, cross-checking multiple floorplans
   per property, a fifth double-blind batch -- per design draft P1.
2. Symmetric-symbol abstain gate (half a day, pure CV) -> abstain on unresolvable symbols.
3. Aggregate multiple floorplan pages per property (yield grows as 1-(1-p)^k, for free).
4. Feed multiple locator windows to the reader (the fix for the 23 NONE cases).
5. Footprint alignment (OSM) for the ~40% without a compass (separate lane).

The bar did not move: fifth double-blind batch >= 50% yield x >= 95% precision before entering the
production dispatcher.

## Reader spike (2026-09-02): small CNN on synthetic rotations -- passed the P0 gate

P0 gate from the design draft: acc@22.5 >= 90% and post-gate yield >= 50%.

**First, a research-phase bug was fixed**: Qwen's `bbox_2d` is in **0-1000 normalised coordinates** (all
236 boxes are <= 1000, and several exceed the native image width), but the frozen pipeline's
`pipeline_v2.build_regions` used them as 1288-px coordinates. After fixing the mapping, 232 of the 234
seed crops land on the compass (the two misses: one landed on a text banner, one on a tiny compass inside
a site-plan inset). The earlier note "VLM bbox alone lands ~60% of windows" was an underestimate caused by
this bug.

**Reader**: torchvision ResNet18 (ImageNet pretrained) + sin/cos regression head, angular loss 1-cos.
Training data = 170 real seed crops from batch1-3 with **random synthetic rotations** (label rotated to
match) + scale/offset jitter + floorplan clutter lines + photometric jitter; 48 samples per seed per epoch,
25 epochs, ~22 minutes on Mac MPS. **Test = the 64 real batch4 crops (not used in training; the same batch
as the v2.2 double-blind)**:

| Metric | Value |
|---|---|
| Median angular error | 4.2 deg |
| acc@15 / @22.5 / @30 | 92.2% / 92.2% / 93.8% |
| 180-degree flips | 0 |
| 4x90 rotation equivariance gate, spread <= 10 | 70.3% yield @ 100% (45/45) |
| spread <= 15 | **82.8% yield @ 100% (53/53)** |
| spread <= 20 | 87.5% yield @ 98.2% |

Same batch for comparison: v2.2 frozen pipeline 36% @ 83%; Claude blind reading 92% (answered only 36/64).
The 5 errors: a dark disc N on a beige background (Claude also marked it uncertain), a very thin-line star,
a **Sunrise/Sunset diagram (not a compass)**, and a rose8 on a photo background -- these 4 had spreads of
68-178 degrees and were all stopped by the gate; the one that got through was a 25-degree error (spread 18).

**Four-fold leave-one-batch-out cross-validation** (each batch scored by a model that never saw it;
15 epochs; one bad seed excluded), 233 real crops pooled (recorded in `reader/results/`, re-computable with
`reader/aggregate_lobo.py`):

| Fold (test batch) | n | Median error | acc@22.5 | Flips | Gate <= 15 |
|---|---|---|---|---|---|
| batch1 | 39 | 4.0 | 97.4% | 0 | 84.6% @ 100% |
| batch2 | 69 | 4.8 | 94.2% | 1 | 79.7% @ 98.2% |
| batch3 | 61 | 4.3 | 95.1% | 0 | 75.4% @ 100% |
| batch4 | 64 | 5.0 | 93.8% | 0 | 76.6% @ 100% |
| **Pooled** | **233** | **4.6** | **94.8%** | **1** | **spread<=10: 62.2% @ 100%; <=15: 78.5% @ 99.5% (single miss 23.4 deg); <=20: 85.0% @ 99.0%** |

Exact 8-way match inside the gate is 94.0% (172/183); the misses are small errors near 45-degree bin
boundaries, so **the product layer must abstain, or give the two adjacent directions, within +-7 degrees
of a boundary** -- not a reading problem. The four batches differ a lot in style mix (a third of batch2/3
is "other"), yet the numbers are stable, so the long tail of styles is no longer the main risk.

**Conclusion**: on this route the reading problem is solved (consistent across four double-blind batches).
Remaining risks: (1) non-compass symbols read as compasses (sunrise diagrams, entrance arrows; batch3 once
had 2/64 false detections) -- this needs a separate "is it a compass" decision, the equivariance gate
cannot catch it; (2) distribution shift between production crop windows and seed windows (production uses
the same normalised-bbox mapping, expected to match; verified by the fifth batch).

**Reproduce** (`reader/`): `download_truth.py` *(not included)* fetches the 236 labelled images ->
`prep_seeds.py` (normalised-bbox crops + a contact sheet with truth arrows for visual checks) ->
`train_reader.py --train 1,2,3 --test 4` -> `run_lobo.sh` (four folds) -> `aggregate_lobo.py`. Seed crops,
`seeds.tsv` and `seg_probe.py` (a first probe of a CubiCasa5K UNet on UK plans: walls usable, windows not)
are *not included*.

## Production P1 (2026-09-02 evening): the floorplan-north lane goes live

Code: `scripts/lib/compass_reader_model.py` (model / synthesis / TTA), `compass_north.py` (geometry / gates
/ schema), `compass_detect.py` (VLM detection), `compass_presence.py` (compass-ness classifier),
`scripts/train_compass_reader.py` / `train_compass_presence.py` (training). The lane script itself
(`enrich_floorplan_north.py`) is *not included*: it is coupled to the private image-fetch layer.

**Weights** (`data/models/`, not in git; the training scripts are reproducible with seed=0):
- `compass_reader_v1.pt`: all 233 seeds x 25 epochs; train-real self-check acc@22.5 100%, median 2.7 deg;
  the held-out estimate is the four-fold LOBO above.
- `compass_presence_v1.pt`: positives = 233 seeds, negatives = 836 isolated ink clusters from the same
  images (excluded by centre distance) + random patches, 8 epochs. Held-out batch4: FN 5/64, FP 2/224;
  held-out batch2: FN 1/69, FP 4/234. **Of the 6 missed positives, 5 are already stopped by the
  equivariance gate and the last has prob 0.488** -> threshold set to 0.4 (`compass_north.PRESENCE_MIN`),
  so no readable real compass is lost. The FPs are honest confusions (a circular "B" logo, a double-headed
  stair arrow), not label noise (checked one by one).

**Three gates + binning**: VLM detects -> compass-ness >= 0.4 -> 4x90 equivariance spread <= 10
(`SPREAD_TAU`); 8-way bins within +-7 degrees of a 45-degree boundary (`BIN_MARGIN`) -- abstained at first,
changed on 2026-09-03 to return the adjacent pair "A|B"; property level: all readings pairwise <= 15
(`PROPERTY_TOL`). Tables: `floorplan_north` (per image, PK property_id/idx/model_version) and
`property_north` (per property). Runs from launchd every 30 minutes with `--limit 120`; failed rows are
retried after 24 h.

**First real run** (2026-09-02 18:25 UTC, first launchd round of 120 images, ~2 s/image with the VLM idle,
252 s in total): emitted 55 / absent 47 / gated 18 / failed 0 across 118 properties; the first 12 emitted
images were all checked visually and point at the N marker; the 4 gated ones checked were genuinely
ambiguous symbols (Z-shaped double-headed arrow, V-shaped logo, round N badge). With the VLM fully loaded
by other lanes, detection rose to ~26 s/image, so a 120-image round takes ~50 minutes; launchd does not
overlap runs.

### Fifth double-blind batch (P1 gate: >= 50% yield x >= 95% precision) -- PASS

Sampling with `batch5/prepare.py` (seed 20260902): 120 floorplans the lane had already processed, not in
batch1-4; predictions exported to `predictions_lane.jsonl` **before** labelling. Blind labelling: two
sub-agents with no context from this session, 60 images each, following `batch5/LABEL_BRIEF.md` (images
only; every reading measured and justified), merged into `truth_angles.tsv` / `present.txt`. Scored with
`batch5/score.py`.

| Metric | Value |
|---|---|
| Ground truth has a compass | 76 / 120 (63%) |
| VLM detection recall | 76 / 76 = 100% |
| Lane emitted | 58 |
| **Yield** (emitted / has compass) | **76.3%** |
| **Precision** (emitted with \|delta\| <= 22.5) | **58 / 58 = 100%** |
| Angular error of emitted readings | median 3.0, p90 7.7, max 15.5 deg |
| Emitted on floorplans with no compass | 0 / 44 |
| Exact 8-way (excluding boundary abstentions) | 40 / 41 = 97.6% |
| Labeller flagged AMBIG but emitted | 1 (reading matches the labeller's final reading) |
| Of the 76 not emitted | 18 were all stopped by the equivariance `spread` gate; 16 of those first readings were in fact within 22.5 |

- The only 8-way disagreement: predicted 329.5 (NW) vs truth 345 (N); a 15.5-degree error, within
  tolerance but across the 337.5 bin line and 8 degrees from it, outside the +-7 abstain band. That is the
  inherent cost of hard binning; if the 8-way field must be >= 99% externally, raise `BIN_MARGIN` to 10 or
  move to 16 directions.
- Yield headroom: 16 of the 18 images stopped by `spread <= 10` were read correctly. Four-fold LOBO shows
  99.5% precision at spread <= 15; if the product needs more coverage, tau=15 can be validated on the next
  double-blind batch (the threshold is not changed on this batch -- pre-registration discipline).
- Compass-ness gate: the lowest presence_prob among the 76 emission candidates was 0.9998, so it stopped
  nothing here; its value is on the no-compass side (VLM false detections), and the 44 no-compass images
  produced zero emissions.

## Assets (in the private repo)

- `batch{1..4}/truth_angles.tsv` -- **236 ground-truth labels** (id, clock position, style, evidence) from
  independent blind-labelling agents, with PCA / circle-fit numeric checks; batch4 includes
  convention-ambiguity flags. *Not included* (keyed by listing IDs).
- `batch*/sample.tsv`, `present.txt`, `detect_results.json`, `predictions_*.json(l)` -- *not included*.
- `scripts/` -- the full v2.2 pipeline and experiment scripts; only `compass_locate.py` (needed by
  `train_compass_presence.py`) and `compass_extract.py` (needed by `reader/prep_seeds.py`) are included.
