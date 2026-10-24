# Lead Review: 030-NG-P5 Heteroscedastic Three-Way

**Reviewed:** 2026-08-23  
**Verdict:** STRONG CONSTRUCTED-DATA PASS

## Verified

- Fixed-variance stage selection estimates one sigma per stage from held-out validation residuals and retains it for test; it does not re-estimate after validation enters full location training.
- Constant scales, null correlations, data/source integrity, predictions, and three-way calculations pass.
- Natural wins NLL on heteroscedastic A/C; homoscedastic wins constant B. Mean NLL ranks natural 1.0573, ordinary 1.0934, homoscedastic 1.2211.
- Natural scale correlation is about 0.97 and scale error is lowest among input-dependent variants on A/C.
- All four pre-registered hypotheses are supported, including the control where natural need not win.

## Decision

Accept a strong constructed-data result directly aligned with NGBoost's uncertainty motivation. This evidence is more informative than a generic synthetic performance test because ground-truth conditional scale and a homoscedastic control are known. Scope remains limited to the frozen generator and validation-NLL selector.

Checkpoint `030-NG-P5` and the NGBoost constructed-data pilot are closed.
