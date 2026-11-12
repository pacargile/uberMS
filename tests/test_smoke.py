"""
Smoke tests: every regression case runs end to end with a few hundred
steps and gives finite posteriors. These check that the code runs, not
that the answers are right; tests/regression/run.py compare does that.

    pytest tests/test_smoke.py            # all cases, a few minutes
    pytest tests/test_smoke.py -k ms_phot

Needs the NN files (set UBERMS_NN_DIR); skipped if they are missing.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'regression'))
import run as reg  # noqa: E402
from cases import CASES, NN, resolve  # noqa: E402

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(NN[k]) for k in ('specNN', 'photNN', 'mistNN')),
    reason='regression NN files not found, set UBERMS_NN_DIR')

# parameters every case of a kind must return
CORE = {'MS': ['EEP', 'initial_Mass', 'initial_[Fe/H]', 'Teff', 'log(g)', 'log(R)', 'Age'],
        'TP': ['Teff', 'log(g)', '[Fe/H]', 'log(R)']}


@pytest.mark.parametrize('name', list(CASES))
def test_case_runs(name):
    case = resolve(name)
    settings = reg.FAST[reg.settings_key(case['driver'])]
    tab, _ = reg.fit_case(name, seed=0, settings=settings, tag='_smoke')
    summ = reg.summarize(tab)
    core = list(CORE[case['driver'][-2:]])
    if 'parallax' in case['data'] or 'phot' in case['data']:
        core += ['dist', 'Av']
    for par in core:
        assert par in summ, f'{par} missing from the output'
        assert summ[par]['nfinite'] == len(tab), f'{par} has non-finite samples'
        assert np.isfinite(summ[par]['median'])
