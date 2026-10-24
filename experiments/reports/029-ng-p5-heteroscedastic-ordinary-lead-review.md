# Lead Review: 029-NG-P5 Heteroscedastic Ordinary

**Reviewed:** 2026-08-23  
**Verdict:** FUNCTIONAL PASS / provisional natural advantage

## Verified

- Protocol, frozen data, features, seeds, validation-NLL selector, source status, and outputs match 028 except `natural_gradient=False`.
- All scales/predictions are finite; constant regime correlation is correctly null.
- Paired deltas confirm natural lower NLL on A/B/C, mean ordinary-minus-natural NLL `+0.0361`, and better natural scale recovery on heteroscedastic regimes.

## Decision

Accept the provisional natural-over-ordinary result. Final constructed-data verdict remains blocked on fixed-variance completion. No fix required.
