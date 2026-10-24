# Historical constructed-data audit for Assignment 3

Status: **METRICS PASSED; historical manifest checksum exceptions retained**.

Verified 42 checksum entries and all 7,200 saved test predictions.
Recomputed NLL, RMSE, oracle NLL, scale MAE, coverage and widths match stored metrics.

## Conditional 95% coverage by fixed |x1| bins

| Variant | Regime | [0,.5) | [.5,1) | [1,1.5) | [1.5,2) |
|---|---|---:|---:|---:|---:|
| natural | A | 0.984 | 0.938 | 0.961 | 0.908 |
| natural | B | 0.969 | 0.962 | 0.975 | 0.934 |
| natural | C | 0.979 | 0.933 | 0.966 | 0.908 |
| ordinary | A | 0.938 | 0.914 | 0.956 | 0.913 |
| ordinary | B | 0.927 | 0.928 | 0.970 | 0.908 |
| ordinary | C | 0.953 | 0.957 | 0.966 | 0.918 |
| homoscedastic | A | 1.000 | 0.990 | 0.970 | 0.832 |
| homoscedastic | B | 0.969 | 0.962 | 0.985 | 0.944 |
| homoscedastic | C | 1.000 | 0.986 | 0.970 | 0.832 |

## Scope

This is artifact verification, not retraining or repeated-seed robustness.
Historical v1 remains unchanged. Fresh replicates are reported separately.

## Historical checksum exceptions

The following manifest.json files do not match their recorded checksums.
Their manifests list a hash of manifest.json itself, which cannot describe the final self-containing file.
All data, source, prediction and metrics checksums checked above match. No historical files were rewritten.
- ordinary: recorded `e8f1018b0ebe9cd8fd903494398387de69ccab23cd48458affc91e8eb0109c3c`, current `7adabc44a278035cba61f8919c700afde4747b0e6642795c1c3c7f6c44f7a8df`.
- homoscedastic: recorded `e7575c4508f914b622a31b381755688aae2d44e1d79886f19a0556e1d886de72`, current `696940d7ace9daefe631db97d2b238abda63f630827878b6134ff1dc375d57ae`.
