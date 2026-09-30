# Room / window geometry: ground-truth labelling brief (for independent labelling agents)

Look only at the floorplans in `imgs/` (file name = pid.png). Do not read any other file in this
directory (except `sample.tsv`, only to get the pid list), any prediction output or any database.

For each image, record every **named room** (Bedroom / Kitchen / Reception / Bathroom / Hall / Landing /
Utility / Study / Garage etc.; unnamed corridors may be skipped):
- `floor`: the floor heading the room is under, as written on the plan (e.g. "Ground Floor" /
  "First Floor"; "single" if none).
- `label`: the room label text (without dimensions), e.g. "Bedroom 2", "Kitchen / Dining Room".
- `type`: bedroom / living / kitchen / dining / bathroom / hall / utility / study / other.
- `windows`: **the page sides of the walls that have windows**, a subset of top / right / bottom / left
  (relative to the room itself: a window in the room's upper wall -> top).
  Window = thin parallel lines drawn in a wall gap, or a rectangular bay projecting out (a bay window is
  recorded on the wall it sits in); skylights / roof windows are not recorded.
- `glazed_doors`: glazed doors to the outside (patio / bi-fold / French doors, drawn as a door swing
  opening onto a garden/balcony), same top/right/bottom/left subset; ordinary internal doors are not
  recorded.
- `confidence`: high / low (low when lines are too thin, the image is too small, or window symbols are
  unclear).
- `note`: optional.

Method: look at the whole plan first to find the outer wall outline and the garden / street sides, then
zoom into each room (crop with PIL, write to the scratchpad, then Read it) and count window symbols.
**If you can't see it clearly, write low; do not guess.**
In a terraced house the two side walls are party walls and usually have no windows; do not assume a
window just because a wall is external.

Output: `truth_<your shard>.jsonl`, one line per image:
`{"pid": "...", "rooms": [{"floor": "...", "label": "...", "type": "...", "windows": ["top"], "glazed_doors": [], "confidence": "high", "note": ""}, ...], "image_note": "..."}`
Every image must have a line (if no rooms can be identified, write rooms=[] and explain).
