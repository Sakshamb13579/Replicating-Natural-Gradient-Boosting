# Kin8nm data check

- Source: bundled official file `data/uci/kin8nm.csv` in upstream repo
- Worktree path: `.RnD/work/ngboost/repair-worktree/data/uci/kin8nm.csv`
- SHA-256: `7b9bf0301ac936d88122557a151e1ba8f1ebc278fcf46d9f3c6d462debdbc8ad`
- Shape: `(8192, 9)` — 8 feature columns (`theta1`–`theta8`) + target `y`
- Missing values: 0
- Target `y`: mean 0.714, std 0.264, range [0.040, 1.459]
- Loader: official `regression_exp.py` reads CSV; script uses last column as label → `X.shape=(8192, 8)`

Verified before NG-1C runs (2026-08-22).
