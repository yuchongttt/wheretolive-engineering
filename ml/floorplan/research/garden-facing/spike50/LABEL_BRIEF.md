# Outdoor-block side: ground-truth labelling brief (for independent labelling agents)

Look only at the floorplans in `imgs/` (file name = `pid.png`). You must **not** read any other file in
this directory (`sample.tsv` may be used only for its pid column; do not look at the other columns), any
prediction output, or any database. You are labelling the **page side**, which has nothing to do with the
real compass -- where north is on the page is not your concern, and do not try to guess it.

## Record for each image

- `outdoor_drawn`: is an outdoor area **actually drawn** on the plan (a garden, terrace, balcony or
  courtyard with a boundary line / fill / label; parking excluded)? `yes` / `no`. **If "garden" is only
  mentioned in text and no area is drawn, it is `no`.** This item is itself something we are measuring,
  so fill it in honestly.
- `blocks`: every drawn outdoor block (empty list if none):
  - `type`: garden / terrace / balcony / patio / yard / other
  - `label`: the label text on the plan (`""` if none)
  - `side`: which side of **the building block it is attached to** it sits on, on the page: top / right /
    bottom / left. If it wraps a corner, list both (e.g. `["bottom","right"]`).
- `primary_side`: the side(s) of the **main** outdoor space (the largest garden/terrace; if there is no
  garden, the largest balcony), a subset of top/right/bottom/left. `[]` when `outdoor_drawn=no`.
- `confidence`: high / low (low when the boundary is unclear, the block is too small to tell which wall it
  belongs to, or on a multi-floor plan you cannot tell which floor it belongs to).
- `note`: optional; write down why you hesitated.

## Method

1. Look at the whole plan first and find the **main building's** outer wall outline (walls are thick
   solid lines with rooms inside). On multi-floor plans (Ground Floor / First Floor side by side), first
   work out which outer wall of which floor the outdoor block is attached to -- a garden usually belongs
   to one floor only.
2. Outdoor blocks usually sit outside the building, drawn with a thin boundary and possibly a grass or
   paving fill, or words such as "Garden" / "Terrace" / "Balcony".
3. If unsure, zoom in: crop with PIL, write it to the scratchpad, then Read it. **If you can't see it
   clearly, write low; do not guess.**
4. Common traps: (1) scale bars / legends / logos are not outdoor blocks; (2) "Garden Flat" is a listing
   name, not a drawn garden; (3) parking bays / driveways are not the main outdoor space -- they may be
   recorded as a block with `type: other`; (4) a roof plan is not a terrace.

## Output

`truth_<your shard>.jsonl`, one line per image:

```
{"pid": "...", "outdoor_drawn": "yes", "blocks": [{"type": "garden", "label": "Garden", "side": ["bottom"]}], "primary_side": ["bottom"], "confidence": "high", "note": ""}
```

Every image must have a line. When done, report: total images, number with outdoor_drawn=yes, number
with high confidence, and the 3 pids you are least sure about.
