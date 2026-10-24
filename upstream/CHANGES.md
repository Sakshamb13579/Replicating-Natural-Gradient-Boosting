# Changes to upstream NGBoost

`upstream/ngboost` is the official NGBoost repository
(<https://github.com/stanfordmlgroup/ngboost>, Apache License 2.0), pinned at
commit `672034041fc05d9d4b7a80c1e6163ea102579752`. The submodule itself is
unmodified.

## The one change

`patches/repair-imports.diff` adds two import lines to
`examples/experiments/regression_exp.py`:

```python
from ngboost.distns import Normal, NormalFixedVar
from ngboost.scores import MLE
```

The released script builds the distribution and score from command-line
strings with `eval`, but never imports those names, so it stops with
`NameError` at `Dist=eval(args.distn)`. The patch changes nothing else: the
model, the splits and the stage selection are as released (report, Section 3).

Apply it with:

```sh
git -C upstream/ngboost apply ../patches/repair-imports.diff
```

The patch is kept as a separate file so that the upstream code stays
byte-identical to the pinned commit and the modification is identified, as the
Apache License 2.0 requires for modified files.
