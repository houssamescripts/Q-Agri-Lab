# Prepared three-crop data

Class order: **alfalfa, maize, wheat**. These map to the source's seasonal-crop labels Alfalfa, Corn and Wheat. Mixed or conflicting overlapping labels are excluded.

| Split | Alfalfa | Maize | Wheat | Total |
| --- | ---: | ---: | ---: | ---: |
| Training | 247 | 242 | 425 | 914 |
| Validation | 50 | 50 | 50 | 150 |
| Test | 42 | 77 | 102 | 221 |

## Features and dates

`train.csv` and `validation.csv` contain `sample_id`, 13 numerical features and `target`; `test.csv` has no target. Feature order is **B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, NDVI**. Bands retain the source's numerical scales; calibrated reflectance and upstream cloud handling are undocumented. NDVI is dimensionless and bounded to [-1, 1]. Daily medians summarize finite, nonnegative-band measurements after same-date duplicates are collapsed. No interpolation or zero filling is used.

The region is **Doukkala**, with imagery dated **25 April 2024** and surveys from **23–27 April 2024**. Crop labels are assumed unchanged within this **±2-calendar-day window**; this is not independently verified crop-persistence evidence. Interpret results within this local cohort, rather than all Moroccan agriculture.

## Keep the splits fixed

Each row represents a distinct physical-field group. Overlapping polygons are grouped; linked 2 km spatial cells stay in the same split. Cross-split polygon distances are at least 100 m (measured minima are in `manifest.json`). Validation means development data: use it for choosing settings. Use test features only for final predictions; do not recover their labels from the source data. IDs and spatial-group codes are bookkeeping, never predictive features.

## Prepared views and subsets

- `views/pca4.npz` and `views/pca8.npz`: aligned `X_train`, `X_validation`, `X_test`, `ids_*`, training/validation labels and class order. There is no `y_test`. StandardScaler → full-SVD PCA → StandardScaler is fitted on training only. The notebook reconstructs the four-component transform; the eight-component view is available for your experiments.
- `starter_ids.json`: fixed quick-run subset of 100 training fields (34 alfalfa, 33 maize, 33 wheat).
- `training_600_ids.json`: nested 600-field subset for learning curves. [Coverage commands](../docs/COVERAGE.md) refit preprocessing on each selected training subset.
- `validation_groups.json`: opaque spatial-block membership for paired bootstrap comparisons, without coordinates.
- `manifest.json`: checksums, feature/class order, counts, dates and spatial separation. Keep supplied files unchanged so loading can verify their integrity.

The QNN's fixed input rule is `(π/2) * (1 + tanh(x))`. For exact reproduction, use the notebook's `fit_transform` path and pinned versions; small numerical differences can affect derivative-free optimization.

## Attribution

Youness Ouassanouan, Jamal Elfarkh, Said Grich, Abderrahman Liblab and Abdelghani Chehbouni, [Crop and irrigation type dataset for Moroccan agricultural regions, Figshare v2](https://doi.org/10.6084/m9.figshare.28486022.v2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Modifications: crop/date filtering, daily medians, physical-field grouping, spatial splits, opaque IDs and training-only PCA. Attribute the source and your changes; the source authors do not endorse this challenge.

Source archive SHA-256: satellite features `31229f73fb10ff0cf480edef171ae5ea5b47c66d76009187400e6b781b302415`; shapefile `80d499f3e88dc37fef4d2116f2c1a782cf25e0ad6c6e8b07a564d00de4183bfc`.
