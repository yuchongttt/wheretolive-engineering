# Pre-registration (2026-09-03, written after scoring batch1 and before sampling batch2)

batch1 (spike50) is a **calibration set**: I looked at the numbers for three gate variants (raw / maj3 /
eq4) before choosing the gate, so its 100% cannot be extrapolated. This project has already fallen into
exactly this trap once (compass: 95% on the batch1 calibration set -> 68% on the batch3 double-blind set).

## Frozen configuration under test (must not change on batch2)

- Route: **G1** -- the VLM reads the side word directly (`side` field); G2's bbox geometry is not used
  (clearly worse on batch1, 87%).
- Gate: **eq4** -- ask once for each clockwise rotation 0/90/180/270; map readings back to the original
  frame; emit only if **all four frames agree**, otherwise abstain. maj3 is not used (93% on batch1,
  below the 95% exposure threshold).
- Inclusion: `floorplan_north.emitted=1` (theta passed all three gates) and the VLM flagged
  has_garden/terrace/balcony.
- Prompt, max_tokens, temperature and concurrency stay exactly as in the current `route_g_vlm.py`.

## Pre-registered criteria

- **Primary metric**: side precision = share of emitted images whose predicted side is in the
  ground-truth primary_side.
- **Pass line**: >= 90% (the threshold in spec section 5, P2). Missing it means failing; no switching to a
  different gate afterwards to rescue the result.
- **Mandatory comparison**: the constant "top" baseline (batch1 = 88%). Precision not clearly above the
  baseline = no incremental ability.
- **Mandatory second item**: number of non-top ground-truth images emitted, and how many are correct --
  that is the only incremental value of this route.
- Yield is secondary; no minimum.

## Known limitations (written up front, so I cannot talk myself out of them later)

- The two labelling agents labelled **different shards**; there is no cross-labelling, hence no
  inter-annotator agreement measure.
- batch1 ground truth is 88% `top` (UK floorplan convention), a severe class imbalance; with n=50 there
  are only 6 non-top images, so evidence of non-top discrimination is thin by construction. batch2 will
  be thin too -- judge the two batches combined.
