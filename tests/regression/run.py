"""
uberMS regression harness.

    python tests/regression/run.py simulate  [--cases ...]
        Simulate each case with the current models and freeze the data in
        tests/regression/data/.

    python tests/regression/run.py reference [--cases ...] [--seeds 0 1 2]
        Fit each case once per seed with the current code and store the
        posterior summaries in tests/regression/reference/<case>.json.
        Several seeds measure the run-to-run scatter that sets the
        comparison tolerances. Seeds can run in parallel processes
        (one --seeds value each); `collect` re-merges their parts.

    python tests/regression/run.py compare   [--cases ...] [--seeds 0 1 2]
        Fit each case with the current code once per seed and compare the
        ensemble with the reference ensemble (see compare_ensemble). Exits
        with status 1 if any parameter is out of tolerance.

    --fast runs a few hundred steps instead of the case settings. Use it to
    check that every case runs; its results are not comparable with the
    references.

Posterior samples are written to $UBERMS_REGRESSION_OUT (default
~/Astro/ai/claude/uberMS/regression), outside the repository.

The fitting call is isolated in fit_case(). It runs the legacy fitters
(sviMS, sviTP, nutsMS, nutsTP); when the new uberMS.fit(indict) exists, a
second adapter will translate the same case into the new indict.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# test the working tree, not an installed copy of uberMS
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from cases import CASES, NN, FILTERS, resolve  # noqa: E402

DATADIR = os.path.join(HERE, 'data')
REFDIR = os.path.join(HERE, 'reference')
OUTDIR = os.environ.get('UBERMS_REGRESSION_OUT',
                        os.path.expanduser('~/Astro/ai/claude/uberMS/regression'))

FAST = {'svi': {'steps': 300, 'post_resample': 500, 'progress_bar': False},
        'nuts': {'steps': 50, 'warmup': 50, 'progress_bar': False}}

# observed / bookkeeping columns that are not posterior parameters
SKIP_COLUMNS = {'mu', 'specobs', 'photobs', 'para'}


# ----------------------------------------------------------------- drivers
def make_driver(name, verbose=False):
    """Legacy fitter instance with the regression NNs."""
    if name in ('sviMS', 'sviTP'):
        from uberMS.runSVI import sviMS, sviTP
        cls = {'sviMS': sviMS, 'sviTP': sviTP}[name]
    else:
        from uberMS.runNUTS import nutsMS, nutsTP
        cls = {'nutsMS': nutsMS, 'nutsTP': nutsTP}[name]
    kw = dict(NN)
    if name.endswith('TP'):
        kw.pop('mistNN')
        kw.pop('mNNtype')
    return cls(verbose=verbose, **kw)


def settings_key(driver):
    return 'svi' if driver.startswith('svi') else 'nuts'


# ----------------------------------------------------------------- data io
def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def load_frozen(case):
    """Frozen data, truth and (if any) 3D Av table for a case."""
    base = os.path.join(DATADIR, case['data_name'])
    with np.load(base + '.npz') as f:
        arrs = {k: f[k] for k in f.files}
    with open(base + '_truth.json') as f:
        meta = json.load(f)
    return arrs, meta


def build_indict(case, arrs, meta, outfile, settings=None):
    """Legacy indict for a case from its frozen data."""
    data = {}
    if 'spec' in case['data']:
        data['spec'] = [arrs['wave'], arrs['flux'], arrs['eflux']]
    if 'phot' in case['data']:
        filters = [str(f) for f in arrs['filters']]
        data['phot'] = {f: [float(m), float(e)]
                        for f, m, e in zip(filters, arrs['phot'], arrs['photerr'])}
        data['filtarr'] = filters
    if 'parallax' in case['data']:
        data['parallax'] = [float(arrs['parallax'][0]), float(arrs['parallax'][1])]

    priors = copy.deepcopy(case['priors'])
    initpars = dict(case['initpars'])
    if meta.get('av3d') is not None:
        priors['Av'] = ['3D', copy.deepcopy(meta['av3d'])]
        if initpars.get('Av') == '3D':
            initpars['Av'] = meta['truth']['Av'] + 0.1

    indict = {'data': data, 'initpars': initpars, 'priors': priors, 'outfile': outfile}
    indict.update(copy.deepcopy(case['indict']))
    indict[settings_key(case['driver'])] = dict(settings or case['settings'])
    return indict


# ----------------------------------------------------------------- simulate
def simulate(name):
    """Simulate one case from its truth with the current models."""
    import jax
    from numpyro import handlers

    case = resolve(name)
    if case['data_name'] != name:
        print(f'[simulate] {name}: reuses the data of {case["data_name"]}')
        return
    sim = case['sim']
    truth = dict(case['truth'])
    rng = np.random.default_rng(zlib.crc32(name.encode()))

    av3d = None
    priors = copy.deepcopy(case['priors'])
    if priors.get('Av', [None])[0] == '3D':
        from uberMS.dustprior import dustprior_setup
        priors = dustprior_setup(priors, verbose=False)
        av3d = _jsonable(priors['Av'][1])
        truth['Av'] = float(np.interp(truth['dist'], av3d['dist_grid'], av3d['Av_mean']))

    # placeholder data with the right shapes, only used to build the model
    arrs = {}
    if 'spec' in case['data']:
        wave = np.exp(np.linspace(np.log(sim['wave_min']), np.log(sim['wave_max']), sim['npix']))
        arrs.update(wave=wave, flux=np.ones_like(wave), eflux=0.01 * np.ones_like(wave))
    if 'phot' in case['data']:
        arrs.update(filters=np.array(list(FILTERS)), phot=np.full(len(FILTERS), 10.0),
                    photerr=np.array(list(FILTERS.values())))
    if 'parallax' in case['data']:
        arrs['parallax'] = np.array([1000.0 / truth['dist'], sim['parallax_err']])

    simcase = dict(case, priors=priors, initpars=truth,
                   driver=case['driver'].replace('nuts', 'svi'))
    if simcase['driver'] == 'sviTP':
        # simulate in log(R), not log(R/dist)
        simcase['indict'] = dict(case['indict'], reparam=False)
    indict = build_indict(simcase, arrs, {'av3d': None}, outfile=os.devnull)
    driver = make_driver(simcase['driver'])
    _, model, _, modelkw = driver.run(indict, dryrun=True)

    tr = handlers.trace(handlers.substitute(handlers.seed(model, 0), data=truth)).get_trace(**modelkw)
    latent = [k for k, s in tr.items() if s['type'] == 'sample' and not s['is_observed']]
    missing = [k for k in latent if k not in truth]
    if missing:
        raise ValueError(f'{name}: truth is missing sampled parameters {missing}')
    derived = {k: float(np.asarray(s['value'])) for k, s in tr.items()
               if s['type'] in ('sample', 'deterministic') and not s.get('is_observed', False)
               and np.size(s['value']) == 1}

    out = {}
    if 'spec' in case['data']:
        mean = np.asarray(tr['specobs']['fn'].loc)
        if case['indict'].get('specmode', 'normalized') == 'fluxed':
            eflux = np.abs(mean) / sim['snr']
        else:
            eflux = np.full_like(mean, np.median(np.abs(mean)) / sim['snr'])
        out.update(wave=arrs['wave'], flux=mean + eflux * rng.normal(size=mean.size),
                   eflux=eflux, flux_true=mean)
    if 'phot' in case['data']:
        mean = np.asarray(tr['photobs']['fn'].loc)
        err = arrs['photerr']
        out.update(filters=arrs['filters'], phot=mean + err * rng.normal(size=mean.size),
                   photerr=err, phot_true=mean)
    if 'parallax' in case['data']:
        plx = 1000.0 / truth['dist']
        out['parallax'] = np.array([plx + sim['parallax_err'] * rng.normal(), sim['parallax_err']])

    os.makedirs(DATADIR, exist_ok=True)
    np.savez_compressed(os.path.join(DATADIR, name + '.npz'), **out)
    meta = {'case': name, 'truth': truth, 'derived': derived, 'av3d': av3d,
            'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'provenance': provenance()}
    with open(os.path.join(DATADIR, name + '_truth.json'), 'w') as f:
        json.dump(_jsonable(meta), f, indent=1)
    shown = {k: round(v, 4) for k, v in derived.items()
             if k in ('Teff', 'log(g)', '[Fe/H]', 'log(R)', 'Age', 'Av', 'dist')}
    print(f'[simulate] {name}: {shown}')
    jax.clear_caches()


# ----------------------------------------------------------------- fit
def fit_case(name, seed, settings=None, tag=''):
    """Fit a case with the legacy fitters; returns (samples table, seconds)."""
    from jax import random as jrandom
    from astropy.table import Table

    case = resolve(name)
    arrs, meta = load_frozen(case)
    os.makedirs(OUTDIR, exist_ok=True)
    outfile = os.path.join(OUTDIR, f'{name}{tag}_seed{seed}.fits')
    indict = build_indict(case, arrs, meta, outfile, settings=settings)
    driver = make_driver(case['driver'])
    driver.rng_key = jrandom.PRNGKey(seed)
    t0 = time.perf_counter()
    driver.run(indict)
    dt = time.perf_counter() - t0
    return Table.read(outfile), dt


FIT_HEADER = ('METHOD', 'PARETOK', 'KTHRESH', 'IS_ESS', 'IS_DRAWS')


def fit_method(tab):
    """How the fit got its posterior (SVI fitters): the header cards uberMS writes."""
    return {k: tab.meta[k] for k in FIT_HEADER if k in tab.meta}


def summarize(tab):
    """Per-parameter posterior summary of a samples table."""
    out = {}
    for col in tab.colnames:
        if col in SKIP_COLUMNS:
            continue
        x = np.asarray(tab[col], dtype=float)
        if x.ndim != 1:
            continue
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        p = np.percentile(x, [2.5, 16, 50, 84, 97.5])
        out[col] = {'median': p[2], 'p16': p[1], 'p84': p[3], 'p2.5': p[0], 'p97.5': p[4],
                    'mean': float(np.mean(x)), 'std': float(np.std(x)),
                    'nfinite': int(x.size)}
    return _jsonable(out)


def provenance():
    """Software versions and the uberMS commit."""
    import jax, numpyro
    info = {'python': sys.version.split()[0], 'jax': jax.__version__,
            'numpyro': numpyro.__version__, 'x64': bool(jax.config.read('jax_enable_x64'))}
    for mod in ('Payne', 'misty', 'uberMS'):
        try:
            m = __import__(mod)
            info[mod + '_file'] = os.path.dirname(m.__file__)
        except ImportError:
            pass
    try:
        repo = os.path.dirname(os.path.dirname(HERE))
        info['uberMS_commit'] = subprocess.check_output(
            ['git', '-C', repo, 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', repo, 'status', '--porcelain', 'uberMS'],
                                        text=True).strip()
        info['uberMS_dirty'] = bool(dirty)
    except Exception:
        pass
    return info


# ----------------------------------------------------------------- reference
PARTDIR = os.path.join(OUTDIR, 'reference_parts')


def make_reference(name, seeds, fast=False):
    """
    Fit a case once per seed. Each seed's summary is written to its own
    file in PARTDIR, so seeds can run in separate processes, and then all
    parts found for the case are collected into reference/<case>.json.
    """
    import jax
    case = resolve(name)
    settings = FAST[settings_key(case['driver'])] if fast else None
    for seed in seeds:
        tab, dt = fit_case(name, seed, settings=settings, tag='_fast' if fast else '')
        print(f'[reference] {name} seed {seed}: {dt:.0f} s', flush=True)
        jax.clear_caches()
        if fast:
            continue
        part = {'case': name, 'driver': case['driver'], 'settings': case['settings'],
                'seed': seed, 'seconds': round(dt, 1), 'summary': summarize(tab),
                'fit_method': fit_method(tab),
                'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'provenance': provenance()}
        os.makedirs(PARTDIR, exist_ok=True)
        with open(os.path.join(PARTDIR, f'{name}_seed{seed}.json'), 'w') as f:
            json.dump(part, f, indent=1)
    if not fast:
        collect(name)


def collect(name):
    """Merge the per-seed parts of a case into reference/<case>.json."""
    import glob
    parts = []
    for path in sorted(glob.glob(os.path.join(PARTDIR, f'{name}_seed*.json'))):
        with open(path) as f:
            parts.append(json.load(f))
    parts = [p for p in parts if p['case'] == name]  # guard against prefix matches
    if not parts:
        return
    ref = {'case': name, 'driver': parts[0]['driver'], 'settings': parts[0]['settings'],
           'created': max(p['created'] for p in parts), 'provenance': parts[0]['provenance'],
           'seeds': {str(p['seed']): p['summary'] for p in parts},
           'seconds': {str(p['seed']): p['seconds'] for p in parts},
           'fit_method': {str(p['seed']): p.get('fit_method', {}) for p in parts}}
    os.makedirs(REFDIR, exist_ok=True)
    with open(os.path.join(REFDIR, name + '.json'), 'w') as f:
        json.dump(ref, f, indent=1)
    print(f'[reference] {name}: collected seeds {sorted(ref["seeds"])}')


# ----------------------------------------------------------------- compare
# SVI results depend on the random key: one key can move a median by ~1
# sigma along a degeneracy (measured on ms_phot), and a refactored model
# follows a different random path even with the same seed. So the new code
# is fit with several seeds and the ensemble is compared with the reference
# ensemble: a parameter passes when the difference of the mean medians (and
# of the mean log widths) is within SCATTER_MULT standard errors, using the
# scatter of both ensembles, or within a floor relative to the width.
SCATTER_MULT = 3.0
MED_FLOOR = 0.1      # in units of the reference posterior sigma
WID_FLOOR = 0.15     # fractional


def _sig(s, par):
    return 0.5 * (s[par]['p84'] - s[par]['p16'])


def compare_ensemble(ref, news):
    """Compare a list of new summaries with the reference seed summaries."""
    rows, ok = [], True
    refs = list(ref['seeds'].values())
    for par in refs[0]:
        if any(par not in n for n in news):
            rows.append((par, 'MISSING', '', '')); ok = False
            continue
        rmed = np.array([s[par]['median'] for s in refs if par in s])
        nmed = np.array([n[par]['median'] for n in news])
        rsig = np.array([_sig(s, par) for s in refs if par in s])
        nsig = np.array([_sig(n, par) for n in news])
        sig = float(np.mean(rsig))
        if sig <= 0:  # fixed or deterministic constant
            good = np.allclose(nmed, rmed.mean(), rtol=1e-6, atol=1e-10)
            rows.append((par, 'ok' if good else 'FAIL', 'fixed', '')); ok &= bool(good)
            continue
        se = np.sqrt(1.0 / rmed.size + 1.0 / nmed.size)
        # pooled scatter between seeds, in units of sigma
        smed = np.std(np.concatenate([rmed - rmed.mean(), nmed - nmed.mean()]),
                      ddof=2 if rmed.size + nmed.size > 2 else 0) / sig
        dmed = (nmed.mean() - rmed.mean()) / sig
        tol_m = max(SCATTER_MULT * smed * se, MED_FLOOR)
        if np.all(nsig > 0):
            lr, ln = np.log(rsig), np.log(nsig)
            slw = np.std(np.concatenate([lr - lr.mean(), ln - ln.mean()]),
                         ddof=2 if lr.size + ln.size > 2 else 0)
            lrat = ln.mean() - lr.mean()
        else:
            slw, lrat = 0.0, -np.inf
        tol_w = max(SCATTER_MULT * slw * se, np.log(1.0 + WID_FLOOR))
        good = (abs(dmed) <= tol_m) and (abs(lrat) <= tol_w)
        rows.append((par, 'ok' if good else 'FAIL',
                     f'{dmed:+.3f} (tol {tol_m:.3f})',
                     f'{np.exp(lrat):.3f} (tol {np.exp(-tol_w):.3f}-{np.exp(tol_w):.3f})'))
        ok &= bool(good)
    return ok, rows


def compare(names, seeds, fast=False):
    import jax
    allok = True
    for name in names:
        path = os.path.join(REFDIR, name + '.json')
        if not os.path.exists(path):
            print(f'[compare] {name}: no reference, skipped')
            continue
        with open(path) as f:
            ref = json.load(f)
        settings = FAST[settings_key(ref['driver'])] if fast else ref['settings']
        news, dts, methods = [], [], []
        for seed in seeds:
            tab, dt = fit_case(name, seed, settings=settings, tag='_compare')
            news.append(summarize(tab)); dts.append(dt)
            methods.append(fit_method(tab).get('METHOD', '-'))
            jax.clear_caches()
        ok, rows = compare_ensemble(ref, news)
        ref_dt = np.mean(list(ref['seconds'].values()))
        print(f'\n[compare] {name}: {"PASS" if ok else "FAIL"}  '
              f'({len(seeds)} seeds vs {len(ref["seeds"])} reference seeds; '
              f'{np.mean(dts):.0f} s per fit, reference {ref_dt:.0f} s)')
        ref_methods = [m.get('METHOD', '-') for m in ref.get('fit_method', {}).values()]
        if any(m != '-' for m in methods + ref_methods):
            print(f'  method per seed: {methods}, reference {ref_methods}')
        print(f'  {"parameter":22s} {"status":7s} {"d(mean median)/sigma_ref":28s} sigma/sigma_ref')
        for r in rows:
            print(f'  {r[0]:22s} {r[1]:7s} {r[2]:28s} {r[3]}')
        allok &= ok
    return allok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['simulate', 'reference', 'collect', 'compare', 'list'])
    ap.add_argument('--cases', nargs='+', default=list(CASES))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--fast', action='store_true')
    args = ap.parse_args()
    bad = [c for c in args.cases if c not in CASES]
    if bad:
        ap.error(f'unknown cases {bad}; known: {list(CASES)}')

    if args.action == 'list':
        for name in CASES:
            c = resolve(name)
            print(f'{name:24s} {c["driver"]:7s} {"+".join(c["data"])}')
    elif args.action == 'simulate':
        for name in args.cases:
            simulate(name)
    elif args.action == 'reference':
        for name in args.cases:
            make_reference(name, args.seeds, fast=args.fast)
    elif args.action == 'collect':
        for name in args.cases:
            collect(name)
    else:
        sys.exit(0 if compare(args.cases, args.seeds, fast=args.fast) else 1)


if __name__ == '__main__':
    main()
