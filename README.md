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
  training ranges. User priors outside those ranges trigger a warning.
- Model wavelengths are vacuum.
- The spectrum uses the G23 extinction law. When fitting spectra and photometry
  together, check that the photometric NN uses a consistent extinction law.
