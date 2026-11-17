# uberMS
Uber Fast MINESweeper using NUT or SVI

## Spectral fitting modes

Spectral NNs that predict F_lambda with extinction inputs (the FlxEm family,
`sNNtype='FlxEm_v1'`, and `MLP_v*`) can fit a spectrum in two ways, set with
`specmode` in the dictionary passed to `run()`:

```python
indict = {
    'data':     {'spec': [wave, flux, eflux], 'parallax': [plx, eplx]},
    'initpars': {...},
    'priors':   {'lsf': ['fixed', 18000.0], ...},
    'specmode': 'normalized',   # default, or 'fluxed'
    'fluxscale': 1.0,           # fluxed mode only, see below
    'outfile':  'samples.fits',
}
```

The modes work the same way in all of the single-star fitters: `sviTP` and
`nutsTP` (ThePayne), and `sviMS` and `nutsMS` (MIST + ThePayne). Pass the NN types
when creating the fitter, e.g.
`sviMS(specNN=..., sNNtype='FlxEm', photNN=..., pNNtype='MLP_v2', cwcversion='v2.0', nnversion='v5', mistNN=...)`.

- **`'normalized'`** (default): the model is computed with Av = 0 and Rv = 3.1,
  divided by its mean, and multiplied by a Chebyshev polynomial (pc0-pc3 by
  default, or pc0-pcN where N is the highest pc term with a prior). The
  polynomial absorbs the continuum shape, so pc0 is ~1 for a spectrum
  normalized to ~1. The model still carries the full SED shape, so if the input
  spectrum was continuum-normalized, the polynomial has to undo that shape. Over
  a wide wavelength range you may need more pc terms (uberMS warns when
  max(wave)/min(wave) > 1.5).
- **`'fluxed'`**: the spectrum is fit as an observed flux,
  f_lambda = F_lambda (R/dist)^2 10^(-0.4 A_lambda), with R = 10^log(R) in
  R_sun, dist in pc, and the G23 extinction law. This adds `log(R)`, `dist`,
  `Av`, and `Rv` to the fit (shared with the photometry when both are fit), and
  uses the parallax when one is given. No polynomial is applied unless you set
  priors on pc terms. Then it acts as a flux-calibration correction, with pc0
  fixed to 1 unless you also set a prior on it, because a free pc0 is
  degenerate with (R/dist)^2.

**Units of a fluxed spectrum.** The model is in erg/s/cm^2/A. With the default
`fluxscale = 1.0`, the input spectrum and its errors must be in erg/s/cm^2/A.
For data in other units, set `fluxscale` to the size of one data unit in
erg/s/cm^2/A; for example, `fluxscale = 1e-17` for SDSS/DESI spectra. In fluxed mode
`specjitter` is a fractional jitter on the model flux (0.01 adds 1% of the
model flux in quadrature to the errors), instead of an additive term in the
units of the spectrum as in normalized mode.

Notes for these NNs:

- The `lsf` prior (scalar FWHM resolving power) must stay below the resolution of
  the NN (`R_fwhm` in the model file), since the emulator cannot sharpen its
  spectra. uberMS raises an error otherwise, including for the default `lsf` prior.
- Default priors on Teff, log(g), [Fe/H], and [a/Fe] are narrowed to the NN's
  training ranges. User priors outside those ranges trigger a warning. In
  `sviMS`/`nutsMS` these labels are predicted by MIST, so the defaults are not
  narrowed. uberMS prints the NN range instead, and you can set priors on
  `Teff`, `log(g)`, `[Fe/H]`, or `[a/Fe]` to keep MIST inside it.
- In `sviMS`/`nutsMS` fluxed mode, `log(R)` comes from MIST; `dist`, `Av`, and `Rv`
  are sampled.
- Model wavelengths are vacuum.
- The spectrum uses the G23 extinction law. When fitting spectra and photometry
  together, check that the photometric NN uses a consistent extinction law.

## SVI fitters: posterior method

`sviMS` and `sviTP` choose how to get the posterior with
`indict['svi']['method']`:

- `'auto'` (default): find the posterior mode, approximate the posterior
  there by a Gaussian (Laplace approximation), and check it by importance
  sampling: draws from a Student-t around the mode are weighted by
  posterior / proposal and Pareto-smoothed (PSIS). If the Pareto shape
  k-hat of the weights is below 0.7, the weighted draws are the posterior;
  otherwise SVI runs as usual.
- `'svi'`: always SVI, skipping the check.
- `'laplace'`: always the Laplace + importance sampling result, with a
  warning if k-hat is above 0.7.

When the check passes it matches NUTS at a fraction of the SVI cost (about
2 min instead of about 30 for a spectral fit on a CPU). Posteriors that are
far from Gaussian, which includes most photometry-only fits, fail it and
run SVI; the check then costs ~5 s for photometry and ~1-4 min for a
spectrum. The method used and the check go in the header of the output
FITS table (`METHOD`, `PARETOK`, `KTHRESH`, `IS_ESS`, `IS_DRAWS`, `NMODES`) and in the
fitter's `fitinfo` attribute. When the Laplace result is used,
`run()` returns `None` for the SVI result.

```python
indict['svi'] = {
    'method': 'auto',
    'is_draws': 10000,       # importance draws
    'is_draws_max': 40000,   # draws when k-hat is within 0.1 above the threshold
    'is_nu': 4,              # Student-t proposal dof; None for a Gaussian
    # 'k_threshold': 0.7,    # default min(1 - 1/log10(draws), 0.7)
    'mode_search': True,     # look for other solutions (below)
}
```

### Several solutions (e.g., main sequence vs past the turnoff)

Near the main-sequence turnoff a star's photometry can fit both a younger
main-sequence star and an older one past the turnoff. One Gaussian, and
the SVI guide, describe only one of them, and k-hat does not always notice
a distant second solution. So `'auto'` and `'laplace'` repeat the mode
search from more starting points: `initpars` moved in EEP by -150 to +100,
also with the mass x0.8 and x1.25 (MIST models), or in log(g) by +-0.5 and
+-1 (free atmospheres). Solutions at least 3 Laplace sigma apart in some
parameter, with at least 1% of the (Laplace-estimated) posterior mass,
count as separate. With more than one, `'auto'` falls back to SVI and
prints a warning listing them:

```
[uberMS] WARNING: the posterior has 2 separate solutions:
[uberMS]     EEP 481, initial_Mass 1.32, log(g) 3.64, Teff 6242, Age 2.64: weight ~0.91
[uberMS]     EEP 369, initial_Mass 1.72, log(g) 3.80, Teff 6610, Age 1.37: weight ~0.09
```

The header gets `NMODES` (absent with `'method': 'svi'` or
`'mode_search': False`). SVI still describes only one solution, so treat
these fits, the ages in particular, with care, or fit them with NUTS
started at each solution (NUTS chains rarely cross between solutions, so
one chain is not enough either). The weights are rough. On an MSTO mock
built from the regression star `ms_phot` (EEP 470, its photometric and
parallax errors), the main-sequence solution holds 12% of the posterior;
the Laplace weight said 9%, and SVI gave 0%. The search costs ~15 s for
photometry and ~1-2 min for a spectrum with a MIST model; it found one
solution on every single-solution regression star. Turn it off with
`'mode_search': False`.

### Number of SVI steps

`indict['svi']['steps']` is a fixed number of steps (default 30000), or
`'auto'` to stop when the loss has converged:

```python
indict['svi'] = {
    'steps': 'auto',
    'max_steps': 40000,    # stop here if the loss has not converged
    'stop_window': 2000,   # steps per check
    'stop_tol': 0.5,       # convergence: median loss per window changes by less than this
}
```

SVI then runs in windows of `stop_window` steps and stops once the median
loss of a window has changed by less than `stop_tol` from the window
before, twice in a row. It prints the median loss after each window, and
warns if it reaches `max_steps` first. The number of steps used goes in
the FITS header (`NSTEPS`, and `SVISTOP`, False if it hit `max_steps`)
and in `fitinfo['svi_steps']`. With `'auto'`, `post_resample` defaults to
10000. Stopping after N steps gives the same fit as `'steps': N`.

On a SPHEREx photometric fit (sviTP, 3 seeds, compared with NUTS) this
stopped at 14000-16000 steps, where the posterior medians were within
~0.2 sigma of NUTS and the widths ~0.9 of NUTS'. It needs the default
learning rate: with `'start_tol': 1e-2` the loss stays too noisy to
converge, and the posterior was still up to 0.5 sigma off at 40000 steps.
It has not yet been validated on spectral fits.

## NUTS

`uberMS.runNUTS.nutsTP` and `nutsMS` take the same arguments as `sviTP` and
`sviMS`, and `run(indict, dryrun=False)` takes the same `indict`. The NUTS
settings go in `indict['nuts']`:

```python
indict['nuts'] = {
    'steps': 500,               # posterior samples
    'warmup': 150,
    'dense_mass': True,
    'target_accept_prob': 0.8,
    'max_tree_depth': 10,
    'progress_bar': True,
}
```

Older versions of `nutsTP` read these settings from `indict['svi']`. That still
works, with a warning.

### Multiple spectra (`uberMS.dva`)

`uberMS.dva.runSVI.sviTP` fits several spectra at once, each with its own NN.
`sNNtype`, `specmode` and `fluxscale` can each be a single value for all spectra
or a list with one entry per spectrum, so fluxed and normalized spectra can be
fit together:

```python
from uberMS.dva.runSVI import sviTP

SVI = sviTP(specNN=[via_nn, boss_nn], sNNtype='FlxEm', photNN=..., pNNtype=...)
indict['data']['spec'] = [spec_via, spec_boss]
indict['specmode'] = ['normalized', 'fluxed']
indict['priors'] = {'lsf_0': ['fixed', 18000.0], 'lsf_1': ['fixed', 4500.0], ...}
```

The per-spectrum parameters carry the spectrum index as a suffix: `vrad_<i>`,
`lsf_<i>`, `specjitter_<i>`, and `pc0_<i>`, `pc1_<i>`, .... The stellar
parameters, `vstar`, and (for fluxed spectra) `Av`, `Rv`, `log(R)` and `dist`
are shared by all spectra and the photometry. The default label priors are
narrowed to the range covered by all of the FlxEm NNs in the fit.

## 3D dust map prior on Av

`Av` can be given a prior conditioned on `dist` from the 3D dust maps in the
[dustmaps](https://dustmaps.readthedocs.io) package (top-level `uberMS`
fitters: `sviMS`, `sviTP`, `nutsMS`, `nutsTP`):

```python
indict['priors']['Av'] = ['3D', {'l': l, 'b': b}]
indict['priors']['dist'] = ['GAL', {'l': l, 'b': b, 'dist_ll': 10.0, 'dist_ul': 20000.0}]  # optional
```

Before the fit, uberMS reads the maps along (l, b) and tabulates the Av
prior versus distance. During sampling, the prior at the sampled `dist` is
linearly interpolated from that table. The map mean is saved as `Av_3Dmap`.

- **3D maps** (`maps='auto'`): every map whose footprint covers (l, b) is
  used: `bayestar` (Bayestar19, dec > -30), `decaps` (southern plane), and
  `edenhofer2023` (all sky, 69 pc to 1.25 kpc). Only each map's reliable
  distance range is used. Where maps overlap, the mean is their weighted
  mean, and the variance adds the scatter between the maps to the
  uncertainties each map reports. There, the prior is a truncated normal.
- **Closer than the 3D maps** (`near='flat'`, default): the prior is flat
  from 0 up to Av(d0), with a Gaussian fall-off of width sigma(d0) above it,
  where d0 is the first reliable distance. The only assumption is that
  extinction cannot decrease with distance. `near='linear'` instead assumes
  Av rises linearly in distance.
- **Beyond the 3D maps:** Av ramps over `ramp_dm` (1 mag in distance
  modulus) to the total Av from a 2D map (`map2d='sfd'`, or `'csfd'`).
- **No 3D coverage:** a dust slab with a 125 pc scale height, scaled to the
  2D total.
- **Uncertainty:** sigma^2 = map variance + `sigma`^2 (0.05 mag) +
  (`fracsigma` mu)^2 (0.1), plus (`fracsigma_extrap` mu)^2 (0.2) beyond the
  3D maps.
- **Support:** set with `Av_ll` and `Av_ul`.

### Local or online maps

`source='auto'` (default) reads a map from the dustmaps data directory if it
is installed. Otherwise uberMS reads just this line of sight from the map's
online archive with HTTP range requests, without downloading the map:
Edenhofer+2023 from Zenodo (~3-13 s per line of sight) and Bayestar19 and
the DECaPS mean map from Harvard Dataverse (~7-15 s). Set `source='local'`
or `'remote'` to force one or the other.

The online archives can rate limit many requests in a row, so download the
maps with dustmaps (`dustmaps.bayestar.fetch()`,
`dustmaps.edenhofer2023.fetch()`, `dustmaps.decaps.fetch(mean_only=True)`)
for large batches of stars. The online DECaPS mean map has no per-pixel
uncertainty, so there sigma comes from the scatter between maps and the
floor terms. The 2D map (SFD) is always read locally (`dustmaps.sfd.fetch()`,
~130 MB). Lines of sight already read are cached for the python session.

### Conversions and the table

Conversions to Av (dustmaps documentation): Bayestar19 and SFD x 2.742
(Schlafly & Finkbeiner 2011 Table 6, Rv = 3.1), DECaPS E(B-V) x 3.32, and
Edenhofer+2023 x 2.8.

To reuse or save the table for a line of sight, build it directly:

```python
from uberMS.dustprior import build_av3d_table
tab = build_av3d_table(l, b)      # same keywords as above
indict['priors']['Av'] = ['3D', tab]
```

Rv: the photometric NN now uses the sampled `Rv` (default fixed at 3.1).
Before, the photometry always used Rv = 3.1, whatever the `Rv` prior.

## Sampling log(R) along the radius-distance ridge (TP mode)

With photometry and a parallax, log(R) and dist are tightly correlated:
the photometry constrains R/dist much better than either one. The TP
fitters can sample log(R/dist) with dist instead of log(R), which turns
that narrow ridge into two nearly independent parameters. log(R) is still
saved in the output, and its prior is applied as before (with soft edges of
width 0.01 dex at the edges of a bounded prior).

```python
indict['reparam'] = True    # default for sviTP
indict['reparam'] = False   # default for nutsTP
```

For SVI the reparameterization keeps the guide from collapsing onto the
ridge, which can otherwise give distance widths ~10x too small. NUTS
already handles the correlation with its dense mass matrix, and the
reparameterization makes it ~2x slower, so it is off by default there.
Turn it off for SVI when log(R) is set mostly by its prior, where the soft
edges of the prior make the fit harder.

The SVI normalizing-flow guide starts at `initpars`, with an initial width
set by `indict['svi']['init_scale']` (default 0.1 in unconstrained space).
