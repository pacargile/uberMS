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
                             check_svi_settings, fit_posterior, fitinfo_header)
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
    assert info == {'method': 'svi', 'requested': 'svi'}
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
