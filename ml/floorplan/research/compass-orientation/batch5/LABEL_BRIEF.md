# Fifth-batch blind-labelling brief (for independent labelling agents)

You may look only at the floorplans in `imgs/`. You must **not** read `predictions_lane.jsonl` in this
directory, any prediction file other than `sample.tsv`, or the database tables `floorplan_north` /
`property_north`.

For each image (file name = `pid.png`):
1. Decide whether there is an **explicit compass indicator** (north arrow / compass rose / an N with a
   pointer). `To Garden` arrows, entrance arrows and Sunrise/Sunset diagrams **do not count**.
2. If there is one, give the direction of north as a **clock hour**: 12 = straight up, 3 = right,
   6 = down, 9 = left; halves are allowed (e.g. 7.5). State your evidence clearly: which way the arrow
   tip / filled point faces, where the letter N sits relative to the centre, and pixel measurements when
   needed (Bash + PIL for PCA / circle fitting is allowed).
3. Record the symbol **style** (e.g. compass_rose / four_point_star / plain_arrow / letters_nesw / other).
4. If the symbol itself cannot be resolved (fully symmetric four-point star with an upright N, a
   double-headed arrow whose head and tail look the same, ...), write `AMBIG` in the evidence column and
   still give your most likely reading.

Write two files to this directory:
- `truth_angles.tsv`: `pid\tclock\tstyle\tevidence`, one row per floorplan that has a compass.
- `present.txt`: the pids that have a compass, one per line.
