#!/bin/bash
# Leave-one-batch-out: 4 folds, each trains on the other 3 batches and tests on the held-out real crops.
SP=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
# Seeds to exclude (whitespace-separated ids). The private run excluded one batch3 seed whose VLM box
# landed on a text banner, not a compass.
echo ${COMPASS_EXCLUDE_SEEDS//,/ } > $SP/exclude.txt
for T in 1 2 3 4; do
  TR=$(echo 1 2 3 4 | tr ' ' '\n' | grep -v "^$T$" | paste -sd, -)
  $PY $SP/train_reader.py --train $TR --test $T --epochs 15 --exclude $SP/exclude.txt --out $SP/lobo_t$T > $SP/lobo_t$T.log 2>&1
  echo "fold $T done: $(grep -o '"acc22": [0-9.]*' $SP/lobo_t$T/results.json | head -1)"
done
echo ALL-DONE
