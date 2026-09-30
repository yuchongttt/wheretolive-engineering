# Room / window geometry spike (P2, 2026-09-02/03)

English translation of the private research README, edited for this extract (route scripts, samples,
truth and prediction files are *not included*; only the labelling brief and the scorer are).

Question: which route can read "which walls of each room have windows" from UK floorplans at >= 90%
precision (design spec section 3 S2, section 5 P2)? Sides are in **page coordinates**
(top/right/bottom/left, relative to the room's own walls); combined with the lane's page north angle
theta they give a compass bearing: bearing(top) = (360-theta), right = (90-theta), bottom = (180-theta),
left = (270-theta), mod 360.

## Ground truth (`sample50/`)

A seeded sample of 50 for-sale floorplans where the lane had emitted theta (known, and 100% on the
double-blind batch); two sub-agents with no context each blind-labelled 25 following `LABEL_BRIEF.md`
(images only; each room cropped, zoomed and its window symbols counted), merged into `truth.jsonl`:

| | |
|---|---|
| Rooms (confidence=high) | 318 (another 9 low-confidence rooms not scored) |
| Rooms with windows | 250 (78.6%) |
| Window sides in total | 297; one side 207 / two sides 39 / three sides 4 / no window 68 |
| Rooms with glazed external doors | 39 |

Difficulties the labellers reported: internal windows / sliding doors / wardrobe symbols confused with
windows, rotated garage blocks, lofts with only skylights -- a ceiling for any route.

## Scoring (`score.py`)

Rooms are matched within a pid by normalised label (then by a unique same-type room). **Product precision
= side_precision**: of every "this room has a window on side X" we predict, the share that is right;
**yield** = rooms with a non-empty, fully correct prediction / truth rooms with windows. `--apertures` takes
truth as windows ∪ glazed_doors (route H v0 does not separate windows from external doors).

## Results

| Route | Room recall | **Side precision** | Side recall | Exact set | Yield | Notes |
|---|---|---|---|---|---|---|
| A. VLM on the whole plan (one question per image, ~12 s/image) | 93.7% | **59.6%** (202/339) | 69.9% | 48.7% | 61.3% | 148 one-side rooms: same side 107, opposite 12, adjacent 29; top/left systematically over-reported |
| H. CV v0 (CubiCasa UNet walls/openings + footprint closing + label-to-room + probing outward along edges, ~18 s/image) | 53.1% | **53.4%** (109/204) | 69.0% | 46.7% | 44.3% | half the room recall lost at label -> region (thin partitions / door gaps / unlabelled rooms); the UNet reads text as walls/openings and cannot tell windows from doors |
| A2. VLM per room, zoomed (locate the label bbox, crop a square 45% of the long edge, describe the four walls first then output JSON, ~26 s/image) | 96.5% | **65.9%** (143/217) | 51.1% | 49.8% | 49.8% | the first prompt version was eaten by prose + a 120-token cap (370/373 rooms empty) -- description first, JSON last, take the last {} |
| A ∧ A2 (emit only the intersection of both routes) | 73.6% | 71.6% (83/116) | 37.6% | 46.6% | 42.7% | |
| A ∧ A2 ∧ H | 39.6% | **76.3%** (29/38) | 24.4% | 38.1% | 29.0% | still 1 in 4 wrong when all three agree |

A self-reported confidence gate (`--high-only`) had no effect: the 43 rooms the VLM called low were already
empty.

All three routes are far below the 90% threshold; intersecting them tops out at 76% while yield drops to
about a third. Route A's errors are not random: top/left over-reported 58/50 times, bottom missed 37 times
-- at whole-plan scale the VLM's spatial binding of "which wall this window belongs to" is unreliable, the
same lesson as the compass work ("reading the whole plan ~60%, only a zoomed crop is usable").

## Conclusion and recommended P3 route

1. **The local VLM cannot reliably bind windows to walls** (whole plan 60%, zoomed 66%), just like the
   compass experiments: zooming helps but the ceiling is 60-70%; two- or three-route agreement raises
   precision to 72-76% at the cost of yield falling to 30-40% -- that is the ensemble's ceiling, not a route.
2. **Off-the-shelf CV (CubiCasa5K UNet) does not work directly on UK plans**: text read as walls/openings,
   windows and doors not separated, thin partitions and door gaps missed -> half of room regions lost. A
   geometric route would need a segmentation model retrained on UK plans (pixel-level labels, expensive).
3. **Recommended P3 route = the same playbook as the compass reader: a small room-level model +
   an equivariance abstain gate.**
   - Labels are cheap: the truth is "room x four walls, window yes/no", not a pixel mask; in this spike the
     blind-labelling protocol produced 318 rooms in 2 hours; ~300 images / ~2,000 rooms labelled with the
     same brief would be enough to start training.
   - Training samples = crops centred on room labels (A2's cropping, label localisation measured reliable)
     x random 90-degree rotations (the four wall labels rotate with it) = 4x data plus a built-in
     equivariance gate: rotate 4 times at inference and abstain if the readings don't rotate with it -- on
     the compass reader this lifted precision from 94.8% to 100%.
   - Model: ResNet18 with four sigmoid outputs (top/right/bottom/left) plus a "which room" centre prior;
     acceptance = a new double-blind batch, side_precision >= 90% (after abstentions), yield secondary.
   - Estimate: labelling 1 day, training + gate 1 day, double-blind acceptance
     half a day.
4. **Not doing for now**: UK pixel-level segmentation labels (costly, and v0 shows room regionisation is a
   second hard problem); shipping VLM direct reading (60-66% precision cannot be exposed).
