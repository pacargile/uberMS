"""
Unit tests for the SVI settings helpers in uberMS.fitutils: the loss,
guide and optimizer built from indict['svi'], and the check for unknown
keys. Uses a toy model, so no NN files are needed.

    pytest tests/test_svi_settings.py
"""
import os
import sys

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer import SVI, RenyiELBO, Trace_ELBO, autoguide

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uberMS.fitutils import (AutoBNAFNormalInit, build_guide, build_loss,  # noqa: E402
                             build_optimizer, check_svi_settings)


def toy_model():
    x = numpyro.sample('x', dist.Uniform(0.0, 10.0))
    y = numpyro.sample('y', dist.Normal(0.0, 1.0))
    numpyro.sample('obs', dist.Normal(x + y, 0.5), obs=3.0)


INIT = {'x': 3.0, 'y': 0.0}


def test_default_loss_is_trace_elbo_8_particles():
    loss = build_loss({})
    assert isinstance(loss, Trace_ELBO)
    assert loss.num_particles == 8


def test_renyi_loss():
    loss = build_loss({'loss': 'renyi', 'alpha': 0.5, 'num_particles': 4})
    assert isinstance(loss, RenyiELBO)
    assert loss.alpha == 0.5 and loss.num_particles == 4


@pytest.mark.parametrize('settings', [
    {'loss': 'renyi', 'alpha': 1},      # alpha = 1 is Trace_ELBO
    {'loss': 'trace', 'alpha': 0.5},    # alpha means nothing for trace
    {'loss': 'kl'},
])
def test_bad_loss_raises(settings):
    with pytest.raises(ValueError):
        build_loss(settings)


@pytest.mark.parametrize('kind,cls', [
    ('flow', AutoBNAFNormalInit),
    ('mvn', autoguide.AutoMultivariateNormal),
    ('lowrank', autoguide.AutoLowRankMultivariateNormal),
])
def test_guides_build_and_start_at_init(kind, cls):
    settings = {'guide': kind, 'init_scale': 1e-3}
    guide = build_guide(toy_model, settings, INIT)
    assert isinstance(guide, cls)
    svi = SVI(toy_model, guide, build_optimizer(settings), build_loss(settings))
    state = svi.init(random.PRNGKey(0))
    # with a tiny init_scale the guide's draws sit at the initial values
    post = guide.sample_posterior(random.PRNGKey(1), svi.get_params(state), sample_shape=(200,))
    for k, v in INIT.items():
        assert jnp.allclose(jnp.median(post[k]), v, atol=0.05), (kind, k)


def test_default_guide_is_flow():
    assert isinstance(build_guide(toy_model, {}, INIT), AutoBNAFNormalInit)


def test_unknown_guide_raises():
    with pytest.raises(ValueError):
        build_guide(toy_model, {'guide': 'Normal'}, INIT)


def test_optimizer_schedule():
    opt = build_optimizer({'start_tol': 1e-2, 'decay_steps': 10, 'decay_rate': 0.5, 'opt_tol': 1e-4})
    assert isinstance(opt, numpyro.optim.ClippedAdam)


def test_unknown_settings_key_raises():
    check_svi_settings({'steps': 10, 'loss': 'trace', 'guide': 'mvn'})
    with pytest.raises(ValueError, match='numflows'):
        check_svi_settings({'numflows': 2})
