# External NGBoost paired fold ledger

Deltas are variant A minus variant B; negative values favor A. Wins use lower metric values with `1e-12` tie tolerance.
All pairs were validated to use identical split hashes at each fold.

| Dataset | A | B | Folds | Mean fold NLL delta | NLL wins A/tie/B | Mean fold RMSE delta | RMSE wins A/tie/B |
|---|---|---|---:|---:|---:|---:|---:|
| airfoil-self-noise | natural | ordinary | 5 | -0.699475 | 5/0/0 | -2.821648 | 5/0/0 |
| airfoil-self-noise | natural | homoscedastic | 5 | 0.090299 | 0/0/5 | 0.518856 | 0/0/5 |
| airfoil-self-noise | ordinary | homoscedastic | 5 | 0.789775 | 0/0/5 | 3.340505 | 0/0/5 |
| forest-fires | natural | ordinary | 20 | -0.004065 | 13/0/7 | 0.005783 | 10/0/10 |
| forest-fires | natural | homoscedastic | 20 | -0.000191 | 9/0/11 | -0.005207 | 12/0/8 |
| forest-fires | ordinary | homoscedastic | 20 | 0.003874 | 8/0/12 | -0.010991 | 11/0/9 |
| parkinsons-telemonitoring | natural | ordinary | 5 | -0.018485 | 4/0/1 | 0.214257 | 3/0/2 |
| parkinsons-telemonitoring | natural | homoscedastic | 5 | -0.036127 | 2/0/3 | -0.428201 | 2/0/3 |
| parkinsons-telemonitoring | ordinary | homoscedastic | 5 | -0.017642 | 3/0/2 | -0.642458 | 3/0/2 |
| real-estate-valuation | natural | ordinary | 20 | -0.392615 | 19/0/1 | -5.446673 | 20/0/0 |
| real-estate-valuation | natural | homoscedastic | 20 | -0.148747 | 15/0/5 | 0.104168 | 9/0/11 |
| real-estate-valuation | ordinary | homoscedastic | 20 | 0.243868 | 4/0/16 | 5.550841 | 0/0/20 |
| superconductivity | natural | ordinary | 5 | -1.144691 | 5/0/0 | -21.226555 | 5/0/0 |
| superconductivity | natural | homoscedastic | 5 | -0.264301 | 5/0/0 | 1.236920 | 0/0/5 |
| superconductivity | ordinary | homoscedastic | 5 | 0.880390 | 0/0/5 | 22.463475 | 0/0/5 |
