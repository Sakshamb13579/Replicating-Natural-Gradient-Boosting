# Yacht data check

- Loader: official `regression_exp.py` UCI HTTP URL
- URL: `http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data`
- Observed shape: `(308, 6)` — 308 rows, 6 features + target in last column (script uses `iloc[:, :-1]` / `iloc[:, -1]`)
- Download: succeeded at run time for all Phase 0 and Phase 1 runs (2026-08-22)
- Local cache present: `.RnD/work/ngboost/data/yacht_hydrodynamics.data`, SHA-256 `00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1`
- Experiment protocol: all authorized runs used the official runtime URL via the script loader, not the local cache file
