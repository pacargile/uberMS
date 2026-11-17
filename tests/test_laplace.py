"""
Unit tests for uberMS.laplace and fitutils.fit_posterior: the PSIS
weights, the mode finder, and the 'auto' / 'laplace' / 'svi' methods on toy
models, so no NN files are needed.

    pytest tests/test_laplace.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer import SVI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uberMS.fitutils import (build_guide, build_loss, build_optimizer,  # noqa: E402
                             check_svi_settings, fit_posterior, fitinfo_header,
                             run_svi_auto)
from uberMS.laplace import find_mode, k_threshold, psis  # noqa: E402


# ------------------------------------------------------------------ PSIS
def test_psis_matches_arviz():
    az = pytest.importorskip('arviz')
    rng = np.random.default_rng(1)
    for scale in (0.3, 1.0, 3.0):
        logw = scale * rng.standard_t(3, 4000)
        lw, k = psis(logw)
        lw_az, k_az = az.psislw(logw.copy())
        assert k == pytest.approx(float(k_az), abs=1e-10)
        assert np.allclose(lw, lw_az, atol=1e-10)


def test_psis_equal_weights():
    lw, k = psis(np.zeros(5000))
    assert np.allclose(np.exp(lw), 1 / 5000)
    assert k < 0.5


def test_psis_ignores_zero_weight_draws():
    logw = np.random.default_rng(0).normal(size=3000)
    logw[:10] = -np.inf
    lw, k = psis(logw)
    assert np.all(np.exp(lw[:10]) == 0)
    assert np.isclose(np.exp(lw).sum(), 1.0)
    assert np.isfinite(k)


def test_k_threshold():
    assert k_threshold(10000) == 0.7
    assert k_threshold(1000) == pytest.approx(1 - 1 / 3)


# ------------------------------------------------------------ mode finder
def test_find_mode_quadratic():
    A = np.array([[4.0, 1.9], [1.9, 1.0]])           # correlated, condition ~ 70
    b = np.array([1.0, -2.0])
    vg = jax.jit(jax.value_and_grad(lambda x: 0.5 * (x - b) @ A @ (x - b)))
    x, U, nit, converged, ev, V = find_mode(vg, np.array([10.0, 10.0]))
    assert converged
    # converged = within 1e-4 of the minimum in the potential, i.e. ~0.01
    # sigma from the mode
    assert 0.5 * (x - b) @ A @ (x - b) < 1e-4
    assert U < 1e-4
    assert np.allclose(np.sort(ev), np.linalg.eigvalsh(A), rtol=1e-4)


def test_find_mode_rejects_nonfinite_start():
    vg = jax.jit(jax.value_and_grad(lambda x: jnp.sum(jnp.log(x))))
    with pytest.raises(ValueError):
        find_mode(vg, np.array([-1.0]))


# --------------------------------------------------------- fit_posterior
def gaussian_model():
    """Correlated, nearly Gaussian posterior, with a deterministic site."""
    x = numpyro.sample('x', dist.Uniform(-20.0, 20.0))
    y = numpyro.sample('y', dist.Normal(0.0, 10.0))
    numpyro.deterministic('xy', x + y)
    numpyro.sample('o1', dist.Normal(x + y, 0.1), obs=1.0)
    numpyro.sample('o2', dist.Normal(x - y, 1.0), obs=0.0)


def banana_model():
    """Strongly curved posterior that the Laplace approximation misses."""
    x = numpyro.sample('x', dist.Normal(0.0, 1.0))
    numpyro.sample('y', dist.Normal(x**2, 0.1))


def fit(model, init, settings):
    settings = {'steps': 300, 'post_resample': 3000, 'progress_bar': False, **settings}
    check_svi_settings(settings)
    guide = build_guide(model, settings, init)
    svi = SVI(model, guide, build_optimizer(settings), build_loss(settings))
    fit_key, post_key = random.split(random.PRNGKey(0))
    return fit_posterior(svi, guide, model, {}, init, settings, fit_key, post_key, verbose=False)


def test_auto_uses_laplace_for_gaussian_posterior():
    post, res, info = fit(gaussian_model, {'x': 0.0, 'y': 0.0}, {})
    assert info['method'] == 'laplace' and res is None
    assert info['check']['accepted'] and info['check']['pareto_k'] < 0.7
    assert set(post) >= {'x', 'y', 'xy'}
    assert len(post['x']) == 3000
    # analytic posterior: x + y = 1 +- 0.1, x - y = 0 +- 1 (priors are wide)
    s, dlt = np.asarray(post['x'] + post['y']), np.asarray(post['x'] - post['y'])
    assert np.mean(s) == pytest.approx(1.0, abs=0.01)
    assert np.std(s) == pytest.approx(0.1, rel=0.1)
    assert np.mean(dlt) == pytest.approx(0.0, abs=0.1)
    assert np.std(dlt) == pytest.approx(1.0, rel=0.1)
    assert np.allclose(post['xy'], post['x'] + post['y'])
    hdr = fitinfo_header(info)
    assert hdr['METHOD'] == 'laplace' and 'PARETOK' in hdr


def test_auto_falls_back_to_svi_for_curved_posterior():
    post, res, info = fit(banana_model, {'x': 0.5, 'y': 0.25}, {})
    assert info['method'] == 'svi' and res is not None
    assert not info['check']['accepted']
    assert len(post['x']) == 3000


def test_svi_method_skips_check():
    post, res, info = fit(gaussian_model, {'x': 0.0, 'y': 0.0}, {'method': 'svi'})
    assert info == {'method': 'svi', 'requested': 'svi', 'svi_steps': 300}
    assert res is not None


def test_laplace_method_is_forced():
    post, res, info = fit(banana_model, {'x': 0.5, 'y': 0.25}, {'method': 'laplace'})
    assert info['method'] == 'laplace' and res is None
    assert not info['check']['accepted']


def test_gaussian_proposal_option():
    post, res, info = fit(gaussian_model, {'x': 0.0, 'y': 0.0}, {'is_nu': None})
    assert info['method'] == 'laplace'


def test_bad_method_raises():
    with pytest.raises(ValueError):
        check_svi_settings({'method': 'nuts'})


# ------------------------------------------------------------ mode search
def two_solution_model(w2=0.3):
    """Two separate solutions in log(g), at 3.0 (weight 1 - w2) and 4.0
    (w2), 50 sigma apart: farther than the heavy tails of the Student-t
    proposal reach, so k-hat alone does not see the second one."""
    g = numpyro.sample('log(g)', dist.Uniform(0.0, 5.5))
    numpyro.sample('x', dist.Normal(0.0, 1.0))
    numpyro.factor('lik', jnp.logaddexp(
        jnp.log(1 - w2) + dist.Normal(3.0, 0.02).log_prob(g),
        jnp.log(w2) + dist.Normal(4.0, 0.02).log_prob(g)))


def one_solution_model():
    g = numpyro.sample('log(g)', dist.Uniform(0.0, 5.5))
    numpyro.sample('x', dist.Normal(0.0, 1.0))
    numpyro.factor('lik', dist.Normal(3.0, 0.1).log_prob(g))


def test_mode_starts():
    from uberMS.laplace import MODE_EEP_OFFSETS, MODE_LOGG_OFFSETS, MODE_MASS_FACTORS, mode_starts
    ms = mode_starts({'EEP': 400.0, 'initial_Mass': 1.0, 'Av': 0.1})
    assert len(ms) == len(MODE_EEP_OFFSETS) * (1 + len(MODE_MASS_FACTORS))
    assert all(s['Av'] == 0.1 for s in ms)
    assert len(mode_starts({'log(g)': 4.0, 'Teff': 5000.0})) == len(MODE_LOGG_OFFSETS)
    assert mode_starts({'x': 0.0}) == []


def test_auto_finds_two_solutions_and_falls_back():
    post, res, info = fit(two_solution_model, {'log(g)': 3.0, 'x': 0.0}, {})
    check = info['check']
    assert info['method'] == 'svi' and res is not None
    assert check['n_modes'] == 2 and check['reason'] == 'multimodal' and not check['accepted']
    m = check['modes']
    assert m[0]['log(g)'] == pytest.approx(3.0, abs=0.01) and m[1]['log(g)'] == pytest.approx(4.0, abs=0.01)
    assert m[0]['weight'] == pytest.approx(0.7, abs=0.05)
    assert fitinfo_header(info)['NMODES'] == 2


def test_mode_search_from_the_minor_solution():
    # started in the minor solution: the Laplace step still reports both,
    # most probable first
    post, res, info = fit(two_solution_model, {'log(g)': 4.0, 'x': 0.0}, {'method': 'laplace'})
    assert info['check']['n_modes'] == 2
    assert info['check']['modes'][0]['log(g)'] == pytest.approx(3.0, abs=0.01)
    # forced Laplace is centered on the most probable mode
    assert np.median(np.asarray(post['log(g)'])) == pytest.approx(3.0, abs=0.05)


def test_one_solution_is_not_flagged():
    post, res, info = fit(one_solution_model, {'log(g)': 3.2, 'x': 0.5}, {})
    assert info['method'] == 'laplace' and info['check']['n_modes'] == 1
    assert fitinfo_header(info)['NMODES'] == 1


def test_mode_search_off():
    # without the search the second solution (50 sigma away) goes unnoticed:
    # the Laplace approximation at 3.0 passes the k-hat check
    post, res, info = fit(two_solution_model, {'log(g)': 3.0, 'x': 0.0}, {'mode_search': False})
    assert 'n_modes' not in info['check'] and 'NMODES' not in fitinfo_header(info)
    assert info['method'] == 'laplace'
    assert np.mean(np.asarray(post['log(g)']) > 3.5) < 0.01


def test_mode_search_setting_checked():
    with pytest.raises(ValueError):
        check_svi_settings({'mode_search': 'yes'})


# ------------------------------------------------------- 'steps': 'auto'
def make_svi(settings, init={'x': 0.0, 'y': 0.0}):
    check_svi_settings(settings)
    guide = build_guide(gaussian_model, settings, init)
    return SVI(gaussian_model, guide, build_optimizer(settings), build_loss(settings)), guide


def test_auto_steps_stops_and_matches_fixed_run():
    settings = {'steps': 'auto', 'stop_window': 200, 'max_steps': 20000, 'stop_tol': 0.5}
    svi, guide = make_svi(settings)
    key = random.PRNGKey(0)
    res, info = run_svi_auto(svi, key, {}, settings, verbose=False)
    assert info['stopped'] and info['steps'] < 20000 and info['steps'] % 200 == 0
    assert info['steps'] >= 600 and len(res.losses) == info['steps']
    # stopping after N steps gives the N-step fit (a fresh guide: numpyro's
    # guide setup on the first call uses up part of the key)
    svi2, _ = make_svi(settings)
    ref = svi2.run(key, info['steps'], progress_bar=False, stable_update=True)
    assert np.allclose(res.losses, ref.losses, rtol=1e-8, atol=1e-8)
    a, b = jax.tree.leaves(res.params), jax.tree.leaves(ref.params)   # flow params are nested
    assert len(a) == len(b) > 0
    for x, y in zip(a, b):
        assert np.allclose(x, y, rtol=1e-8, atol=1e-8)


def test_auto_steps_runs_to_max_steps():
    # max_steps not a whole number of windows: the rest runs at the end
    settings = {'steps': 'auto', 'stop_window': 200, 'max_steps': 700, 'stop_tol': 1e-12}
    svi, guide = make_svi(settings)
    res, info = run_svi_auto(svi, random.PRNGKey(0), {}, settings, verbose=False)
    assert not info['stopped'] and info['steps'] == 700 and len(res.losses) == 700


def test_auto_steps_in_fit_posterior():
    settings = {'method': 'svi', 'steps': 'auto', 'stop_window': 200, 'max_steps': 4000,
                'progress_bar': False}
    svi, guide = make_svi(settings)
    fit_key, post_key = random.split(random.PRNGKey(0))
    post, res, info = fit_posterior(svi, guide, gaussian_model, {}, {'x': 0.0, 'y': 0.0},
                                    settings, fit_key, post_key, verbose=False)
    assert len(post['x']) == 10000            # post_resample default with 'auto'
    assert info['svi_steps'] == info['svi_auto']['steps']
    hdr = fitinfo_header(info)
    assert hdr['NSTEPS'] == info['svi_steps'] and hdr['SVISTOP'] == info['svi_auto']['stopped']


@pytest.mark.parametrize('bad', [{'steps': 'many'}, {'steps': 0},
                                 {'steps': 1000, 'max_steps': 5000},
                                 {'steps': 'auto', 'stop_window': 2000, 'max_steps': 5000},
                                 {'steps': 'auto', 'stop_tol': 0}])
def test_auto_steps_bad_settings(bad):
    with pytest.raises(ValueError):
        check_svi_settings(bad)
