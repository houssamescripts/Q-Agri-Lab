# Compare training-set sizes

Keep the **150 validation and 221 test fields fixed**, with all three crop classes. More training data can help or hurt: compare validation macro-F1, accuracy and every class's recall across seeds 0, 1 and 2.

Run from the repository root:

```bash
.venv/bin/python -m challenge.coverage check
.venv/bin/python -m challenge.coverage train --size 600 --results outputs/size600
.venv/bin/python -m challenge.coverage train --size 914 --results outputs/size914
.venv/bin/python -m challenge.coverage compare --baseline outputs/size600 --candidate outputs/size914
```

You can also use `--size 100` or `--size 200` for faster experiments. Each run fits scaling/PCA on its selected training rows, uses the same inputs for QNN and control, and saves validation predictions, scores and optimization histories. The SVM uses the same rows with all 13 features. The commands run all three seeds with the starter's 100-evaluation budget; edit the model construction in `challenge/coverage.py` to compare your own approach.

`comparison.json` reports candidate-minus-baseline differences and one-sided 95% lower bounds from paired spatial-block bootstrapping. A positive lower bound supports an improvement; a bound below zero leaves possible loss unresolved. Seed standard deviations describe a different source of variation. These are validation comparisons, not a guarantee of future test accuracy.

## More than 914 training fields

Request a checked training extension from the organizers. It must keep crop definitions, the ±2-day date window, physical-field grouping and spatial separation valid, while leaving validation/test unchanged. Do not duplicate rows, move evaluation fields into training or add unverified source records. The source's >10,000 records cover other crops, dates and regions as well as repeated observations; they are not all eligible for this task.

After receiving the extension folder (complete `train.csv` and its manifest):

```bash
# Replace this path with the supplied extension folder.
.venv/bin/python -m challenge.coverage check --extension-dir /path/to/extension
.venv/bin/python -m challenge.coverage train --extension-dir /path/to/extension --results outputs/extended
.venv/bin/python -m challenge.coverage compare --baseline outputs/size914 --candidate outputs/extended
```

Inspect all three classes, record the tradeoffs and freeze your settings before final test prediction. No command can guarantee that expanding a dataset preserves accuracy. Save experiments under `outputs/`; keep the supplied `results/` intact.
