# Migrating a runscript from the pre-refactor uberMS

This is for runscripts written for uberMS before the refactor, up to
commit `892a73c`. It covers `uberMS.runSVI` (`sviMS`, `sviTP`) and
`uberMS.runNUTS` (`nutsMS`, `nutsTP`).

The sub-mode drivers (`uberMS.dva`, `uberMS.smes`, `uberMS.smes_dva`,
`uberMS.spots`) are unchanged so far. Scripts that use them need no edits;
keep `'numflows'` there.

Most runscripts need no more than the first two steps below.

## 1. Required: these break old scripts

- [ ] **Reinstall uberMS** in the environment you run from. A non-editable
  install is a copy and does not see the new code:
  `pip install /path/to/uberMS`, or `pip install -e /path/to/uberMS` to
  track the repository.
- [ ] **Rename `indict['svi']['numflows']` to `'num_flows'`.**
- [ ] **Old `'guide'` values have new names.** The old `'Normal'` is now
  `'lowrank'`, the low-rank multivariate normal it gave before. Any other
  old string (the default was the normalizing flow) is now `'flow'`, the
  default, so you can also delete the key.
- [ ] **Unknown `indict['svi']` keys are now an error.** They used to be
  ignored silently, so a typo did nothing. The error message lists the
  allowed keys; delete or fix the ones it names.
- [ ] **If your script uses what `run()` returns:** it is still
  `(svi, guide, svi_result)`, but `svi_result` is `None` when the
  Laplace result is used (see `'method'` below). Code that reads
  `svi_result.losses` must handle that, or set `'method': 'svi'`.

Before and after, for the settings that changed:

```python
# before
indict['svi'] = {'steps': 30000, 'numflows': 3, 'guide': 'Normal'}
# after
indict['svi'] = {'steps': 30000, 'num_flows': 3, 'guide': 'lowrank'}
```

## 2. Defaults that changed

No edit is needed for these, but results will differ from the old version.

- [ ] **Posterior method `'auto'`** (README, "SVI fitters: posterior
  method"). uberMS first fits a Laplace approximation and checks it by
  Pareto-smoothed importance sampling. If the check passes (k-hat < 0.7)
  the weighted draws are the posterior; otherwise SVI runs as before.
  Spectral fits that pass take ~2 min instead of ~30. To keep the old
  behavior, set `'method': 'svi'`.
- [ ] **Search for several solutions** (README, "Several solutions").
  `'auto'` also looks for other modes, e.g., main sequence vs past the
  turnoff. With more than one it falls back to SVI and prints a warning
  listing them. This costs ~15 s for photometry and ~1-2 min for a
  spectrum. To turn it off, set `'mode_search': False`.
- [ ] **SVI loss: `Trace_ELBO`, 8 particles.** It was `RenyiELBO`, with
  alpha 1.25 in `sviMS`, 0 in `sviTP`, and 2 particles. Posteriors are
  closer to NUTS, but spectral SVI costs ~3x more.
- [ ] **Learning rate: halves every 20000 steps** (was 3000). The old
  schedule froze the guide before it had widened. The default `start_tol`
  (1e-3) did best in the tests; a larger value such as 1e-2 was worse at
  every step count, so drop `start_tol` unless you have tested it.
- [ ] **The flow guide starts at `initpars`.** The old flow ignored them
  and started in the middle of the priors. Make the initial values
  sensible.
- [ ] **`sviTP` samples log(R/dist) by default** (`indict['reparam']`,
  True for photometry and fluxed spectra). Your `initpars` `log(R)` is
  converted for you, and the output has a `log(R/dist)` column. The NUTS
  drivers default to `False`.
- [ ] **Small numerical fixes to priors:** the IMF prior is normalized
  over its range, and the sigmoid prior edges are numerically stable.
- [ ] **Random numbers differ.** The fit and the posterior draw use
  separate keys, so even with the same seed a fit does not reproduce the
  old version exactly.

## 3. New options

All are optional.

- [ ] `indict['seed'] = N`: seed for one fit (default 0).
- [ ] `indict['svi']`:
  - `'method'`: `'auto'` (default), `'svi'` or `'laplace'`; tuning:
    `'is_draws'`, `'is_draws_max'`, `'is_nu'`, `'k_threshold'`.
  - `'steps': 'auto'`: stop SVI when the loss converges, with
    `'max_steps'`, `'stop_window'`, `'stop_tol'`. Tested on photometry
    fits so far.
  - `'mode_search'`: True (default) or False.
  - Loss: `'loss'` (`'trace'` or `'renyi'`), `'num_particles'`,
    `'alpha'`.
  - Guide: `'guide'` (`'flow'`, `'mvn'`, `'lowrank'`), `'init_scale'`,
    `'num_flows'`, `'hidden_factors'`, `'rank'`.
  - Schedule: `'start_tol'`, `'decay_steps'`, `'decay_rate'`,
    `'opt_tol'`.
- [ ] **3D dust prior on Av:** `priors['Av'] = ['3D', {'l': l, 'b': b}]`
  (README, "3D dust map prior on Av").
- [ ] **`indict['reparam']`:** set it to choose the log(R/dist) sampling in
  TP mode (README, "Sampling log(R) along the radius-distance ridge").

## 4. Output to expect

- [ ] **New FITS header cards** in the samples table: `METHOD`, `PARETOK`,
  `KTHRESH`, `IS_ESS`, `IS_DRAWS` and `NMODES` from the `'auto'` check;
  `NSTEPS` and `SVISTOP` from SVI. The fitter's `fitinfo` attribute has
  the same information.
- [ ] **`[uberMS]` progress lines** say which stage the fit is in (Laplace
  mode, importance sampling, SVI, posterior draw) and how long each took.
  A `WARNING` is printed when the posterior has more than one solution.
