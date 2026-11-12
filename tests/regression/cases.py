"""
Regression cases for the uberMS refactor.

Each case is a simulated star with known truth, the data it was observed
with, and the priors, initial values, and settings used to fit it. The
simulated data are frozen in data/<name>.npz by `run.py simulate`, so the
references do not depend on the simulator or on model code that the
refactor changes.

Field meanings
--------------
driver    : legacy fitter class ('sviMS', 'sviTP', 'nutsMS', 'nutsTP')
data      : which data the star has, a subset of ('spec', 'phot', 'parallax')
truth     : values of the sampled parameters used to simulate the data
initpars  : starting point of the fit
priors    : uberMS priors; '3D' Av priors are tabulated once at simulation
            time and frozen in the data file
indict    : other top-level indict keys (specmode, fluxscale, reparam, ...)
settings  : 'svi' or 'nuts' settings
sim       : simulation settings (spectral S/N, wavelength grid, errors)
sim_like  : optional, name of a case whose frozen data this case reuses
"""
import os

MODELS = os.environ.get('UBERMS_NN_DIR', os.path.expanduser('~/Astro/SPHEREx/runs/models'))

NN = {
    'specNN': f'{MODELS}/specNN/cwc_v2.0_APOGEE_R25000_FlxEm_v1.h5',
    'sNNtype': 'FlxEm',
    'photNN': f'{MODELS}/photNN/',
    'pNNtype': 'MLP_v2',
    'cwcversion': 'v2.0',
    'nnversion': 'v5',
    'mistNN': f'{MODELS}/mistNN/mistyNN_2.3_v256_v0.h5',
    'mNNtype': 'LinNet',
}

FILTERS = {
    'gaia_g': 0.01, 'gaia_bp': 0.01, 'gaia_rp': 0.01,
    'twomass_J': 0.02, 'twomass_H': 0.02, 'twomass_Ks': 0.02,
    'wise_w1': 0.03, 'wise_w2': 0.03,
}

APOGEE = {'wave_min': 15200.0, 'wave_max': 16900.0, 'npix': 8000}

SVI = {'steps': 30000, 'post_resample': 10000, 'progress_bar': False}
NUTS = {'steps': 1000, 'warmup': 500, 'progress_bar': False}

# priors shared by the MIST cases
MS_PRIORS = {
    'EEP': ['uniform', [200.0, 808.0]],
    'initial_Mass': ['uniform', [0.5, 3.0]],
    'initial_[Fe/H]': ['uniform', [-2.0, 0.45]],
    'initial_[a/Fe]': ['uniform', [-0.15, 0.55]],
    'dist': ['uniform', [10.0, 5000.0]],
    'Av': ['uniform', [0.0, 3.0]],
    'vrad': ['uniform', [-200.0, 200.0]],
    'vstar': ['uniform', [0.0, 30.0]],
    'lsf': ['fixed', 22500.0],
}

# priors shared by the free-atmosphere (ThePayne) cases
TP_PRIORS = {
    'Teff': ['uniform', [3000.0, 9000.0]],
    'log(g)': ['uniform', [0.0, 5.5]],
    '[Fe/H]': ['uniform', [-2.0, 0.45]],
    '[a/Fe]': ['uniform', [-0.15, 0.55]],
    'log(R)': ['uniform', [-1.0, 2.0]],
    'dist': ['uniform', [10.0, 5000.0]],
    'Av': ['uniform', [0.0, 3.0]],
    'vrad': ['uniform', [-200.0, 200.0]],
    'vstar': ['uniform', [0.0, 30.0]],
    'lsf': ['fixed', 22500.0],
}


def _with(base, **kw):
    out = dict(base)
    out.update(kw)
    return out


# abundance priors inside the APOGEE FlxEm NN range ([Fe/H] -0.5..0.5,
# [a/Fe] -0.2..0.2), for the cases with a spectrum
MS_SPEC_ABUND = {'initial_[Fe/H]': ['uniform', [-0.5, 0.45]],
                 'initial_[a/Fe]': ['uniform', [-0.15, 0.2]]}
TP_SPEC_ABUND = {'[Fe/H]': ['uniform', [-0.5, 0.45]],
                 '[a/Fe]': ['uniform', [-0.15, 0.2]]}


CASES = {
    # solar dwarf, APOGEE normalized + 8 bands + parallax, IMF mass prior
    'ms_specphot_norm': dict(
        driver='sviMS', data=('spec', 'phot', 'parallax'),
        truth={'EEP': 350.0, 'initial_Mass': 1.0, 'initial_[Fe/H]': 0.0,
               'initial_[a/Fe]': 0.0, 'dist': 200.0, 'Av': 0.1, 'vrad': 12.0,
               'vstar': 5.0, 'pc0': 1.0, 'pc1': 0.02, 'pc2': -0.01, 'pc3': 0.0},
        initpars={'EEP': 340.0, 'initial_Mass': 0.95, 'initial_[Fe/H]': -0.1,
                  'initial_[a/Fe]': 0.05, 'dist': 210.0, 'Av': 0.2, 'vrad': 10.0,
                  'vstar': 4.0, 'pc0': 1.0, 'pc1': 0.0, 'pc2': 0.0, 'pc3': 0.0},
        priors=_with(MS_PRIORS, initial_Mass=['IMF', {'mass_le': 0.5, 'mass_ue': 3.0}],
                     **MS_SPEC_ABUND),
        indict={'specmode': 'normalized'},
        settings=SVI, sim={'snr': 100.0, 'parallax_err': 0.05, **APOGEE}),

    # metal-poor subgiant, photometry + parallax only
    'ms_phot': dict(
        driver='sviMS', data=('phot', 'parallax'),
        truth={'EEP': 480.0, 'initial_Mass': 1.3, 'initial_[Fe/H]': -0.5,
               'initial_[a/Fe]': 0.2, 'dist': 1500.0, 'Av': 0.4},
        initpars={'EEP': 470.0, 'initial_Mass': 1.2, 'initial_[Fe/H]': -0.4,
                  'initial_[a/Fe]': 0.15, 'dist': 1400.0, 'Av': 0.3},
        priors=dict(MS_PRIORS), indict={},
        settings=SVI, sim={'parallax_err': 0.03}),

    # same star and data as ms_phot, fit with NUTS
    'ms_phot_nuts': dict(
        driver='nutsMS', data=('phot', 'parallax'), sim_like='ms_phot',
        truth=None, initpars=None, priors=None, indict={}, settings=NUTS, sim=None),

    # K dwarf, normalized APOGEE spectrum only
    'ms_spec_norm': dict(
        driver='sviMS', data=('spec',),
        truth={'EEP': 330.0, 'initial_Mass': 0.75, 'initial_[Fe/H]': 0.2,
               'initial_[a/Fe]': 0.0, 'vrad': -30.0, 'vstar': 3.0,
               'pc0': 1.0, 'pc1': -0.03, 'pc2': 0.0, 'pc3': 0.01},
        initpars={'EEP': 340.0, 'initial_Mass': 0.8, 'initial_[Fe/H]': 0.1,
                  'initial_[a/Fe]': 0.05, 'vrad': -25.0, 'vstar': 4.0,
                  'pc0': 1.0, 'pc1': 0.0, 'pc2': 0.0, 'pc3': 0.0},
        priors=_with({k: MS_PRIORS[k] for k in ('EEP', 'initial_Mass', 'vrad', 'vstar', 'lsf')},
                     **MS_SPEC_ABUND),
        indict={'specmode': 'normalized'},
        settings=SVI, sim={'snr': 50.0, **APOGEE}),

    # F dwarf in the disk, fluxed APOGEE spectrum in 1e-17 units + photometry
    # + parallax, 3D dust prior on Av (Bayestar19, frozen at simulation time)
    'ms_specphot_fluxed_3d': dict(
        driver='sviMS', data=('spec', 'phot', 'parallax'),
        truth={'EEP': 400.0, 'initial_Mass': 1.3, 'initial_[Fe/H]': -0.2,
               'initial_[a/Fe]': 0.0, 'dist': 800.0, 'Av': '3D', 'vrad': 45.0,
               'vstar': 15.0},
        initpars={'EEP': 390.0, 'initial_Mass': 1.25, 'initial_[Fe/H]': -0.1,
                  'initial_[a/Fe]': 0.05, 'dist': 750.0, 'Av': '3D', 'vrad': 40.0,
                  'vstar': 12.0},
        priors=_with(MS_PRIORS, Av=['3D', {'l': 75.0, 'b': 3.0, 'maps': ['bayestar'],
                                          'source': 'local'}], **MS_SPEC_ABUND),
        indict={'specmode': 'fluxed', 'fluxscale': 1e-17},
        settings=SVI, sim={'snr': 50.0, 'parallax_err': 0.03, **APOGEE}),

    # free-atmosphere G dwarf, fluxed spectrum + photometry + parallax,
    # log(R/dist) reparameterization (sviTP default)
    'tp_specphot_fluxed': dict(
        driver='sviTP', data=('spec', 'phot', 'parallax'),
        truth={'Teff': 5500.0, 'log(g)': 4.5, '[Fe/H]': -0.3, '[a/Fe]': 0.1,
               'log(R)': -0.05, 'dist': 300.0, 'Av': 0.2, 'vrad': -8.0, 'vstar': 4.0},
        initpars={'Teff': 5600.0, 'log(g)': 4.4, '[Fe/H]': -0.2, '[a/Fe]': 0.05,
                  'log(R)': 0.0, 'dist': 290.0, 'Av': 0.3, 'vrad': -5.0, 'vstar': 5.0},
        priors=_with(TP_PRIORS, **TP_SPEC_ABUND),
        indict={'specmode': 'fluxed', 'fluxscale': 1e-17},
        settings=SVI, sim={'snr': 75.0, 'parallax_err': 0.04, **APOGEE}),

    # free-atmosphere K dwarf, photometry + parallax, IMF log(g) prior
    'tp_phot_imf': dict(
        driver='sviTP', data=('phot', 'parallax'),
        truth={'Teff': 4800.0, 'log(g)': 4.6, '[Fe/H]': 0.0, '[a/Fe]': 0.0,
               'log(R)': -0.15, 'dist': 100.0, 'Av': 0.05},
        initpars={'Teff': 4900.0, 'log(g)': 4.5, '[Fe/H]': -0.1, '[a/Fe]': 0.05,
                  'log(R)': -0.1, 'dist': 105.0, 'Av': 0.1},
        priors=_with(TP_PRIORS, **{'log(g)': ['IMF', {}]}),
        indict={},
        settings=SVI, sim={'parallax_err': 0.05}),

    # same star and data as tp_phot_imf, fit with NUTS
    'tp_phot_imf_nuts': dict(
        driver='nutsTP', data=('phot', 'parallax'), sim_like='tp_phot_imf',
        truth=None, initpars=None, priors=None, indict={}, settings=NUTS, sim=None),
}


def resolve(name):
    """Case dict with sim_like filled in from the case it copies."""
    case = dict(CASES[name])
    case['name'] = name
    src = case.get('sim_like')
    if src is not None:
        base = CASES[src]
        for key in ('truth', 'initpars', 'priors', 'sim'):
            if case.get(key) is None:
                case[key] = base[key]
        case['indict'] = {**base['indict'], **case['indict']}
        case['data_name'] = src
    else:
        case['data_name'] = name
    return case
