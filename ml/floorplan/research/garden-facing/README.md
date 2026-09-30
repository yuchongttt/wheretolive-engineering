# Derived garden-facing spike (2026-09-03)

English translation of the private research README, edited for this extract (listing IDs and
floorplan-producer names removed; files not in this extract marked *(not included)*).

Question: without detecting windows, can we use only "which side of the building the outdoor block is
on" plus the lane's existing page north angle theta to derive the garden's **compass orientation**, and
so extend `garden_facing` beyond the 6.6% of listings where the agent states it?

Conversion (same convention as room-aspect): `bearing(top) = 360-theta`, `right = 90-theta`,
`bottom = 180-theta`, `left = 270-theta`, mod 360.

## Why not detect windows

P2 showed the local VLM tops out at 66% on "which wall does this window belong to" (see
`../room-aspect/README.md`). "Which side of the building is the garden block on" is a **coarser**
decision: the block is large, often labelled, and usually attached as a whole to one outer wall. This
spike measures whether that coarse decision is accurate enough.

## Sampling

`spike50/prepare.py` (seed 20260903) samples 50 images from the pool where the lane has emitted theta
(passed all three gates) and the VLM flagged has_garden/terrace/balcony (937 properties); `batch2/`
(seed 20260904) samples another 50 with zero overlap with batch1 (pool 887). Each batch was blind-labelled
by two sub-agents with no context, following `LABEL_BRIEF.md` (25 images each).

## Two structural findings (batch1)

**(1) The `has_garden/terrace/balcony` flags are trustworthy**: on 50/50 images an outdoor area was
**actually drawn**; none was "garden only mentioned in text". So the reachable surface of this route
(has theta and has an outdoor block, ~70%) is real.

**(2) 88% of ground truth is `top`** -- UK floorplans conventionally draw the rear garden at the top of the
page. Two important consequences:

- **A constant "top" baseline already scores 88%.** Any route not compared with this baseline is passing
  class imbalance off as skill.
- The derived garden orientation is therefore **mostly a re-expression of theta** (bearing ~ 360-theta).
  The real incremental value is not the average accuracy but **whether it recognises the 12% that are
  not on top** -- otherwise it just renames theta.

## batch1 (calibration set) results

| Route / gate | Yield | Precision | Non-top hits |
|---|---|---|---|
| G1 direct side word, no gate | 100% | 72% | 6/6 emitted |
| G1 / maj3 (>= 3 frames agree) | 84% | 93% | 5/5 |
| **G1 / eq4 (all four frames agree)** | **46%** | **100%** (23/23) | **3/3** |
| G2 bbox + geometry / maj3 | 86% | 79% | 4/4 |
| G2 / eq4 | 62% | 87% | 4/4 |

Constant baseline: 88% @ 100% yield. G1+eq4 is 100% on its 23 emitted images, and all 3 non-top ones are
right (the constant baseline must get those 3 wrong) -- the incremental ability is real, but with n=3 the
evidence is thin.

**End-to-end independent check (not relying on our own labels)**: 7 of the 50 have an agent-stated garden
orientation. Using the true side + our theta to derive the compass orientation, **7/7 match the stated
value**, across four different answers (E / W / SW / S), which a constant guess cannot achieve. This one
check validates theta, the geometric conversion and the whole concept at once. Running the same check with
G1+eq4 predictions: eq4 2/2, maj3 3/4 (the one conflict was independently confirmed by the stated value --
exactly the kind maj3 lets through and eq4 stops).

## batch2 (double-blind confirmation batch)

The gate was frozen right after seeing the batch1 numbers and written into `spike50/PREREGISTERED.md`; no
tuning on batch2.

### Result: **PASS**

Pre-registered gate G1 + eq4, batch2 double-blind **46% yield @ 96% precision (22/23)**, >= 90% threshold
**PASS**.

| | batch1 (calibration) | batch2 (double-blind) | Combined |
|---|---|---|---|
| eq4 yield | 23/50 = 46% | 23/50 = 46% | 46/100 = 46% |
| eq4 precision | 23/23 = 100% | 22/23 = 96% | **45/46 = 97.8%** |
| Constant "top" baseline | 88% | 64% | 76% |

Yield **reproduced exactly** across the two batches (23/50 each). Combined record by ground-truth stratum:

- truth top: **38/38**
- truth non-top: **7/7** <- the key point: not a constant predictor; it recognises the 12-24% that are
  not on top
- truth has no outdoor block: **0/1** <- the only error

**The single error**: the plan had only a **text-only dimension box** "Garden 54.90x24.40" (bordered, no
area inside), and the model, consistently across all four frames, treated it as a drawn garden and said
`left`. Post-hoc diagnosis: adding a gate "all four frames say outdoor_drawn=true" would stop **0 images** --
the model was confident in all four frames, so this error **cannot be caught by frame agreement**; it
needs an independent "text box vs real area" discriminator.

**End-to-end falsification against stated values**: across both batches, 3 emitted images had an
agent-stated orientation, **3/3 match** (plus the 7/7 upper-bound check on batch1 using the true side).

### batch2's second finding: the `has_garden` flag over-reports by ~16%

Only **42 of the 50** batch2 images actually drew an outdoor block (batch1 was 50/50; not reproduced). The
false flags were all "text-only dimension box" or "Summer House, Not Shown In Actual Location" cases.
Combined drawn rate over both batches: **92/100**. This number feeds the coverage estimate; batch1's 100%
must not be used.

### Coverage estimate (live denominator, 2026-09-03)

67,250 for-sale listings with a floorplan x theta emitted 46.8% x outdoor flag 69.2% x actually drawn 92%
x eq4 gate 46% -> **~13.7%, about 9,228 listings**. Compared with 4,715 (7.0%) with an agent-stated value
-- **roughly double**, with little overlap (only 12.6% of theta-emitted listings have a stated value).

## Known limitations

- The two labelling agents labelled **different shards**; no cross-labelling, so no inter-annotator
  agreement measure.
- Severe class imbalance (88% top); with n=50 there are only 6 non-top images, so evidence of non-top
  discrimination is thin by construction.
- The stated-value set cannot serve as an acceptance set: of the 4,706 stated values across all listings
  for sale, south/west variants dominate and only 15 are north-facing, so using it as ground truth would
  systematically overstate accuracy. Here it is used only as a **one-sided falsifier**.

## Assets

`spike50/` (calibration) and `batch2/` (double-blind confirmation): `prepare.py`, `LABEL_BRIEF.md`,
`route_g_vlm.py` (one call yields G1/G2 plus the raw readings of all four frames), `score.py`,
`spike50/PREREGISTERED.md`. batch2 used byte-identical copies of `route_g_vlm.py` and `score.py`. Samples,
truth files and predictions are *not included* (keyed by listing IDs); source images are not in git.

## Live cross-check (option 2, 2026-09-03)

The production lane (*not included*) ran on **165** for-sale listings that have theta emitted + a VLM
outdoor flag + an agent-stated orientation (not 4,715 -- theta backfill had only reached 4.2%; this
cross-check set grows linearly with the backfill).

| | |
|---|---|
| Emitted (passed eq4) | 60/165 = 36% |
| Agrees with stated value (<= 45 deg) | **58/60 = 96.7%** |
| >= 135 deg, near-opposite | 2 |

**Both red flags were checked image by image; in both, the floorplan reading holds and the direction in
the listing text is imprecise**:

| Compass symbol | theta | Garden block | Derived | Agent's text |
|---|---|---|---|---|
| arrow pointing up, marked N | 2.4 (correct) | top of the page (correct) | N | "Southerly facing rear garden" |
| arrow pointing up-left | 311.3 (correct) | above the kitchen's rear wall (correct) | NE | "SOUTH FACING REAR GARDEN" |

The two plans come from different floorplan producers, so both compasses cannot be wrong at once. **So this
field does more than extend coverage: it is a detector that checks an agent's south-facing claim against
the agent's own floorplan.**

### What stated "south" really contains

Of the 30 listings stated as south-facing, only **18 (60%)** derive to S:

| Derived | S | SE | SW | N | NE |
|---|---|---|---|---|---|
| Listings | 18 | 8 | 2 | 1 | 1 |

Agents round SE/SW to "south" (4 of the 14 stated west are really SW). This agrees with the impossible
south:north = 153:1 ratio across all stated values.

### Re-validating with the production prompt (2026-09-03, `validate_prod_prompt.py`, *not included*)

Re-running the same 100 frozen ground-truth images through the **production code path** (the lane's
`process_image` + `garden_detect.ask_side`, *not included*), with theta fixed at 0 so only the side is
scored and no theta error is mixed in:

| | Frozen prompt | **Production prompt** |
|---|---|---|
| Yield | 46/100 | **48/100** |
| Precision | 45/46 = 97.8% | **47/48 = 97.9%** |
| Truth top / non-top | -- | 40/41 / **7/7** |
| Gate breakdown | not_drawn 12% / spread 42% | not_drawn 21% / spread 31% |

**Both yield and precision are slightly better.** The text-box guard converts `spread` (four frames
disagree) into `not_drawn` (explicitly "not drawn"), and actually emits 2 more images -- the frozen
prompt's only error (the text-only dimension box) is now correctly judged `not_drawn`, so the guard works
as designed.

The production prompt's single error is one of the three images the blind labeller flagged in advance, in
their words: a 0.9 m2 patio recess walled on three sides, too small to tell which wall it belongs to --
a human cannot judge it either.

**Correction**: an earlier version of this document said "the guard cut yield from 46% to 36%". That was
wrong. The live 36% reflects a **different listing mix** (the subset with a stated garden orientation), not a
prompt regression.

### (Corrected) Production prompt vs the pre-registered configuration

When moving to production I changed the prompt: removed `bbox_2d` (route G2 was not adopted) and **added
"a text-only dimension box does not count as a drawn outdoor area"** (motivated by batch2's only
double-blind error). The live set then showed `not_drawn` rising from 12% in the spike to **24%** and the
emission rate falling from 46% to **36%**, i.e. yield traded for that class of false positive -- but see
the correction above: the like-for-like re-run on the 100 frozen images (previous section) is the clean
number for the production configuration.

So **45/46 on the frozen batches is not the production prompt's number**. The 165-listing cross-check
against stated values is an independent check of the production configuration (96.7%, both conflicts
verified to be on the stated side).

### Proposed revision to design spec section 3 S1 point 4

The original text said "if the stated garden_facing disagrees with the derived result -> abstain on the
whole listing". This evidence shows **both disagreements were errors in the stated value**; under the
original rule the wrong side would veto the right one. Proposed instead: **keep both values, each with a
`source`, and surface disagreements explicitly in the product layer**, with no mutual veto.
