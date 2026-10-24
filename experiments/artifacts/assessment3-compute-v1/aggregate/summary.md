# Assignment 3 compute results

Status: **complete**; validated evaluations: 120.

## External datasets

| Dataset | Variant | Mean fold NLL | Mean fold RMSE | Cap hits |
|---|---|---:|---:|---:|
| superconductivity-full | natural | 3.515762 | 11.558460 | 3 |
| superconductivity-full | ordinary | 4.710811 | 33.858083 | 4 |
| superconductivity-full | homoscedastic | 3.797393 | 10.762413 | 5 |
| airfoil-grouped | natural | 2.545291 | 3.440964 | 0 |
| airfoil-grouped | ordinary | 3.158560 | 5.928334 | 0 |
| airfoil-grouped | homoscedastic | 2.455856 | 2.719077 | 1 |

## Fresh synthetic replicates

| Regime | Variant | Replicates | Mean NLL | NLL SD | Mean scale MAE |
|---|---|---:|---:|---:|---:|
| A | homoscedastic | 10 | 1.037912 | 0.032357 | 0.258793 |
| A | natural | 10 | 0.824916 | 0.034767 | 0.063344 |
| A | ordinary | 10 | 0.856316 | 0.041246 | 0.088789 |
| B | homoscedastic | 10 | 0.936413 | 0.024881 | 0.012132 |
| B | natural | 10 | 0.945564 | 0.024153 | 0.042713 |
| B | ordinary | 10 | 0.991553 | 0.029435 | 0.088954 |
| C | homoscedastic | 10 | 1.721975 | 0.031817 | 0.515749 |
| C | natural | 10 | 1.487641 | 0.033874 | 0.107166 |
| C | ordinary | 10 | 1.516366 | 0.033524 | 0.135785 |

## Interpretation

- Fold summary SE is descriptive, not an iid inferential SE for overlapping holdouts
- Synthetic SE uses fresh independent data/split replicates, paired within replicate
- Do not pair row-split Airfoil with grouped Airfoil folds or pool subset/full Superconductivity
- v1 synthetic seed is historical reference and not included here
