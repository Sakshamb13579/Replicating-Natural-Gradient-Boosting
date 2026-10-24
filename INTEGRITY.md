# Checksum exceptions

Every saved run records SHA-256 values in a `checksums.txt` file. Across
`experiments/artifacts/` there are 1,717 such entries. All of them match the
committed files except the 9 below. The historical files are published exactly
as they were produced; their recorded hashes have not been edited.

## Intentional: Kin8nm targets removed (6 entries)

Kin8nm is cited rather than redistributed, so the `y_true` column was removed
from the saved Kin8nm selector predictions before publication.

| Run folder under `experiments/artifacts/original-selector-sensitivity-v1/kin8nm/` | Files |
|---|---|
| `ngboost-natural/kin8nm-natural-s1-a01/` | `predictions.csv`, `predictions.jsonl` |
| `ngboost-ordinary/kin8nm-ordinary-s1-a01/` | `predictions.csv`, `predictions.jsonl` |
| `ngboost-homoscedastic/kin8nm-homoscedastic-s1-a01/` | `predictions.csv`, `predictions.jsonl` |

`python experiments/src/kin8nm_targets.py restore` rejoins the column from
`upstream/ngboost/data/uci/kin8nm.csv` and stops with an error unless all six
restored files match their original recorded hashes.

## Historical: present in the original records (3 entries)

These three mismatches already existed in the original working records. The
published files are byte-identical to those originals, so packaging did not
introduce them.

| File checked | Recorded in | Cause |
|---|---|---|
| `manifest.json` | `experiments/artifacts/heteroscedastic-v1/ngboost-ordinary/029-ng-p5-heteroscedastic-ordinary-s0-a01/checksums.txt` | In the original records `manifest.json` was written about 1 ms after `checksums.txt`, so the recorded hash is of an earlier version of the manifest |
| `manifest.json` | `experiments/artifacts/heteroscedastic-v1/ngboost-homoscedastic/030-ng-p5-heteroscedastic-homoscedastic-s0-a01/checksums.txt` | As above |
| `checksums.txt` | `experiments/artifacts/yacht/ngboost-natural/017-ng-d1-yacht-natural-s1-a01/checksums.txt` | The file lists a hash of itself, which cannot match once the list is written |

The first two belong to the earlier v1 synthetic study, which the report keeps
only as a historical reference. The third belongs to the first attempt of the
cap diagnostic, kept but not used because its inner validation splits were not
fixed (`experiments/reports/017f1-ng-d1-controlled-cap-diagnostic.md`). The
reported diagnostic is the second attempt (`-a02`), whose checksums all match.
