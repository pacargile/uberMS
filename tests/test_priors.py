"""
Unit tests for the custom prior distributions and the prior helpers.

    pytest tests/test_priors.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from uberMS.advancedpriors import (  # noqa: E402
    Sigmoid_Prior, DSigmoid_Prior, IMF_Prior, IMF_logg_Prior, Gal_Prior, PlateauNormal)
from uberMS.priors import latent_prior_dist  # noqa: E402

jax.config.update('jax_enable_x64', True)
KEY = jax.random.PRNGKey(1)


def _ks_against_density(samples, logpdf, lo, hi, n=20001):
    """KS p-value of samples against a density tabulated on [lo, hi]."""
    grid = np.linspace(lo, hi, n)
    p = np.exp(np.asarray(logpdf(jnp.asarray(grid))))
    cdf = integrate.cumulative_trapezoid(p, grid, initial=0.0)
    cdf /= cdf[-1]
    return stats.kstest(np.asarray(samples), lambda x: np.interp(x, grid, cdf)).pvalue


# ---------------------------------------------------------------- sigmoids
def test_sigmoid_matches_old_formula_and_is_finite():
    d = Sigmoid_Prior(a=2.0, b=1.0, low=-1000.0, high=1000.0)
    x = jnp.linspace(-5.0, 5.0, 101)
    old = jnp.log(1.0 / (1.0 + jnp.exp(-2.0 * (x - 1.0))))
    np.testing.assert_allclose(d.log_prob(x), old, rtol=1e-12, atol=1e-12)
    # the old formula overflowed to -inf here, with NaN gradients
    far = jnp.array([-900.0, -500.0])
    assert np.all(np.isfinite(d.log_prob(far)))
    g = jax.vmap(jax.grad(d.log_prob))(far)
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(g, 2.0, rtol=1e-12)


def test_dsigmoid_matches_old_formula_and_is_finite():
    d = DSigmoid_Prior(a=3.0, b=-1.0, c=-3.0, d=1.0, low=-1000.0, high=1000.0)
    x = jnp.linspace(-4.0, 4.0, 81)
    p1 = 1.0 / (1.0 + jnp.exp(-3.0 * (x + 1.0)))
    p2 = 1.0 / (1.0 + jnp.exp(3.0 * (x - 1.0)))
    np.testing.assert_allclose(d.log_prob(x), jnp.log(p1 * p2), rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(d.log_prob(jnp.array([-800.0, 800.0]))))


@pytest.mark.parametrize('dist', [
    Sigmoid_Prior(a=5.0, b=0.3, low=0.0, high=1.0),
    DSigmoid_Prior(a=20.0, b=0.2, c=-20.0, d=0.7, low=0.0, high=1.0)])
def test_sigmoid_sample(dist):
    s = dist.sample(KEY, (20000,))
    assert s.shape == (20000,)
    assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0
    assert _ks_against_density(s, dist.log_prob, 0.0, 1.0) > 1e-3


# ---------------------------------------------------------------- IMF
@pytest.mark.parametrize('low,high', [(0.25, 3.0), (0.1, 0.4), (0.6, 10.0)])
def test_imf_normalized_and_sampled(low, high):
    d = IMF_Prior(low=low, high=high)
    norm, _ = integrate.quad(lambda m: float(jnp.exp(d.log_prob(m))), low, high,
                             points=[0.5] if low < 0.5 < high else None)
    assert abs(norm - 1.0) < 1e-6
    s = d.sample(KEY, (20000,))
    assert float(s.min()) >= low and float(s.max()) <= high
    assert _ks_against_density(s, d.log_prob, low, high) > 1e-3


def test_imf_shape_unchanged_and_vectorized():
    # differs from the brutus version by a constant only
    d = IMF_Prior(low=0.25, high=3.0)
    m = jnp.array([0.3, 0.45, 0.6, 1.0, 2.5])
    old = jnp.where(m > 0.5, -2.3 * jnp.log(m) + 1.0 * jnp.log(0.5), -1.3 * jnp.log(m))
    diff = d.log_prob(m) - old
    np.testing.assert_allclose(diff, diff[0], rtol=0, atol=1e-12)


def test_imf_logg_sample():
    d = IMF_logg_Prior(logR=0.1, low=3.0, high=5.0)
    s = d.sample(KEY, (20000,))
    assert float(s.min()) >= 3.0 and float(s.max()) <= 5.0
    assert _ks_against_density(s, d.log_prob, 3.0, 5.0) > 1e-3


def test_gal_prior_sample():
    d = Gal_Prior(l=90.0, b=10.0, low=10.0, high=5000.0)
    s = d.sample(KEY, (5000,))
    assert float(s.min()) >= 10.0 and float(s.max()) <= 5000.0
    assert np.all(np.isfinite(d.log_prob(s)))


def test_plateau_normal_sample():
    d = PlateauNormal(0.5, 0.8, 0.1, low=0.0, high=3.0)
    s = d.sample(KEY, (20000,))
    assert _ks_against_density(s, d.log_prob, 0.0, 3.0) > 1e-3


# ---------------------------------------------------------------- helper
@pytest.mark.parametrize('info,x', [
    (['uniform', [4000.0, 6000.0]], 5000.0),
    (['normal', [5000.0, 100.0]], 5100.0),
    (['tnormal', [5000.0, 100.0, 4500.0, 5500.0]], 5100.0),
    (['sigmoid', [0.01, 4500.0, 3000.0, 9000.0]], 5000.0),
    (['dsigmoid', [0.01, 4500.0, -0.01, 6000.0, 3000.0, 9000.0]], 5000.0)])
def test_latent_prior_dist(info, x):
    lp = latent_prior_dist('Teff', info).log_prob(x)
    assert np.isfinite(lp)


def test_latent_prior_dist_rejects_unknown_kind():
    with pytest.raises(ValueError, match='not supported'):
        latent_prior_dist('Teff', ['IMF', {}])


# ---------------------------------------------------------------- model_spec
def test_model_spec_sigmoid_latent_prior():
    """model_spec used to skip sigmoid priors on MIST-predicted parameters."""
    sys.path.insert(0, os.path.join(REPO, 'tests', 'regression'))
    from cases import NN, resolve
    if not os.path.exists(NN['mistNN']):
        pytest.skip('regression NN files not found')
    import run as reg
    from numpyro.infer.util import log_density

    case = resolve('ms_spec_norm')
    arrs, meta = reg.load_frozen(case)
    lps = {}
    for label, prior in (('none', None), ('sigmoid', ['sigmoid', [0.05, 5000.0, 3000.0, 9000.0]])):
        indict = reg.build_indict(case, arrs, meta, outfile=os.devnull)
        if prior is not None:
            indict['priors']['Teff'] = prior
        _, model, _, modelkw = reg.make_driver('sviMS').run(indict, dryrun=True)
        lp, tr = log_density(model, (), modelkw, case['initpars'])
        lps[label] = float(lp)
        if prior is not None:
            assert 'LatentPrior_Teff' in tr
            teff = float(tr['Teff']['value'])
    expected = float(jax.nn.log_sigmoid(0.05 * (teff - 5000.0)))
    assert np.isfinite(lps['sigmoid'])
    np.testing.assert_allclose(lps['sigmoid'] - lps['none'], expected, rtol=1e-8, atol=1e-8)
