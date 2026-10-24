# Original-paper datasets

**Yacht Hydrodynamics** (`yacht_hydrodynamics.data`) is the unchanged UCI file
(CC BY 4.0), SHA-256
`00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1`.

**Kin8nm is not included.** No separate licence statement was located for it,
so the report cites it rather than redistributing it. The file used is the one
distributed with NGBoost:

- path: `upstream/ngboost/data/uci/kin8nm.csv` (present once the submodule is
  checked out)
- SHA-256: `7b9bf0301ac936d88122557a151e1ba8f1ebc278fcf46d9f3c6d462debdbc8ad`
- description: <https://www.cs.toronto.edu/~delve/data/kin/desc.html>

The scripts read it from the submodule. For the same reason the saved Kin8nm
selector predictions in
`artifacts/original-selector-sensitivity-v1/kin8nm/` have their `y_true`
column removed. `python src/kin8nm_targets.py restore` puts it back from the
submodule file and checks each restored file against the run's original
`checksums.txt`.
