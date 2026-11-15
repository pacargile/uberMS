"""
Laplace approximation checked by Pareto-smoothed importance sampling (PSIS).

This is the posterior step of the 'auto' and 'laplace' methods of
indict['svi']['method']:

1. Mode: damped Newton (Levenberg-Marquardt) on the negative log posterior
   in numpyro's unconstrained space, started from the initial values. The
   Hessian comes from central differences of the compiled gradient, so it
   costs no extra compilation.
2. Laplace approximation: a Gaussian at the mode with covariance H^-1.
3. Importance sampling: draws from the Gaussian, weighted by
   posterior / Gaussian and Pareto-smoothed. The Pareto shape k-hat of the
   largest weights says whether the weighted draws can be trusted (Vehtari
   et al. 2024, JMLR 25, 72): below min(1 - 1/log10(S), 0.7) for S draws
   they can, and the posterior is a weighted resample of them.

In the ELBO study (2026-11) this matched NUTS on the regression stars
whose posteriors are close to Gaussian, including a fluxed-spectrum star
that SVI placed up to 4 sigma off, in ~1/20 of the SVI time, and k-hat
picked the better of it and SVI on every star with a reliable NUTS
reference. Non-Gaussian posteriors (photometry-only fits, for example) fail
the check and 'auto' falls back to SVI; the check then costs a few
seconds for photometry and about a minute for a spectrum on a CPU.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jrandom
from jax.flatten_util import ravel_pytree
from jax.scipy.special import gammaln
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

METHODS = ('auto', 'svi', 'laplace')


# ------------------------------------------------------------------- PSIS
def _gpdfit(x):
    """
    Generalized Pareto fit to sorted exceedances x (Zhang & Stephens 2009,
    with the weakly informative prior on k of Vehtari et al.).
    Returns (k, sigma).
    """
    n = len(x)
    m = 30 + int(n**0.5)
    b = 1 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5))
    b /= 3 * x[int(n / 4 + 0.5) - 1]
    b += 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    len_scale = n * (np.log(-(b / k)) - k - 1)
    w = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, b = w[keep], b[keep]
    w /= w.sum()
    b_post = np.sum(b * w)
    k_post = np.log1p(-b_post * x).mean()
    sigma = -k_post / b_post
    k_post = (n * k_post + 10 * 0.5) / (n + 10)
    return k_post, sigma


def _gpinv(p, k, sigma):
    if np.abs(k) < np.finfo(float).eps:
        return -sigma * np.log1p(-p)
    return sigma * np.expm1(-k * np.log1p(-p)) / k


def psis(logw):
    """
    Pareto-smoothed log importance weights.

    logw : unnormalized log weights; -inf for draws with zero weight.
    Returns (normalized log weights, k-hat). k-hat is inf when the tail
    is too short to fit.
    """
    x = np.array(logw, dtype=float)
    x = np.where(np.isnan(x), -np.inf, x)
    n = len(x)
    x -= np.max(x)
    tail_len = int(np.ceil(min(0.2 * n, 3 * np.sqrt(n))))
    order = np.argsort(x)
    cutoff = max(x[order[-tail_len - 1]], np.log(np.finfo(float).tiny))
    tail = np.where(x > cutoff)[0]
    k = np.inf
    finite = x[np.isfinite(x)]
    if finite.size and np.ptp(finite) == 0:
        k = 0.0                            # equal weights: nothing to smooth
    elif len(tail) > 4:
        tail = tail[np.argsort(x[tail])]
        k, sigma = _gpdfit(np.exp(x[tail]) - np.exp(cutoff))
        if np.isfinite(k):
            p = np.arange(0.5, len(tail)) / len(tail)
            x[tail] = np.log(_gpinv(p, k, sigma) + np.exp(cutoff))
            x[x > 0] = 0.0
    x -= np.logaddexp.reduce(x)
    return x, float(k)


def k_threshold(ndraws):
    """Largest reliable k-hat for ndraws draws (Vehtari et al. 2024)."""
    return min(1.0 - 1.0 / np.log10(ndraws), 0.7)


# ----------------------------------------------------------- mode, Hessian
def fd_hessian(vg, x, h=1e-3):
    """
    Hessian by central differences of the gradient function vg, step h in
    unconstrained units. Smaller steps pick up noise in the gradient: on a
    spectral regression star h = 1e-5 gave Hessian errors of ~6% and lost
    two eigenvalues, h = 1e-3 matched jax.hessian to 1e-4.
    """
    d = len(x)
    H = np.empty((d, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        H[:, i] = (np.asarray(vg(x + e)[1]) - np.asarray(vg(x - e)[1])) / (2 * h)
    return 0.5 * (H + H.T)


def find_mode(vg, x0, maxiter=100, tol=1e-4):
    """
    Levenberg-Marquardt: Newton steps damped by lambda along the Hessian
    eigenvectors, lambda shrinking after accepted steps and growing after
    rejected ones. Converged when the Hessian is positive definite and the
    Newton step would lower the potential by less than tol
    (g^T H^-1 g / 2 < tol); a small last step alone is not enough, since
    the fit can crawl along a flat direction.

    Returns (x, potential, iterations, converged, eigenvalues, eigenvectors)
    with the Hessian eigen-decomposition at x.
    """
    x = np.asarray(x0, dtype=float)
    U, g = vg(x)
    U, g = float(U), np.asarray(g)
    if not np.isfinite(U):
        raise ValueError('the log posterior is not finite at the initial values')
    lam = None
    for it in range(maxiter + 1):
        ev, V = np.linalg.eigh(fd_hessian(vg, x))
        gv = V.T @ g
        if ev.min() > 0 and 0.5 * np.sum(gv**2 / ev) < tol:
            return x, U, it, True, ev, V
        if it == maxiter:
            break
        if lam is None:
            lam = 1e-3 * np.max(np.abs(ev))
        for _ in range(40):
            shift = ev + lam
            if shift.min() <= 0:
                lam = 4 * lam - ev.min()
                continue
            xn = x - V @ (gv / shift)
            Un, gn = vg(xn)
            Un = float(Un)
            if np.isfinite(Un) and Un < U:
                break
            lam *= 4
        else:
            break                          # no downhill step left
        x, U, g = xn, Un, np.asarray(gn)
        lam = max(lam / 3, 1e-12)
    return x, U, it, False, ev, V


# ------------------------------------------------------------ the method
def laplace_posterior(model, modelkw, initpars, rng_key, nsamples, settings=None, force=False, verbose=True):
    """
    Laplace approximation + PSIS check for a numpyro model.

    model, modelkw : the model and its keyword arguments
    initpars       : starting values (constrained), as for the SVI guide
    rng_key        : JAX key for the importance draws and the resample
    nsamples       : posterior samples to return
    settings       : indict['svi']; uses
        'is_draws'     : importance draws, default 10000
        'is_draws_max' : draws when k-hat lands just above the threshold
                         (within 0.1); k-hat is noisy there and more draws
                         settle it. Default 40000; set equal to is_draws
                         to turn this off.
        'k_threshold'  : accept below this k-hat, default
                         min(1 - 1/log10(draws), 0.7)
        'is_nu'        : degrees of freedom of the Student-t proposal,
                         default 4; None for a Gaussian. In the ELBO study
                         the t proposal was the one whose k-hat separated
                         the stars where importance sampling matched NUTS
                         (k-hat <= 0.69) from those where it did not
                         (>= 0.87); with the Gaussian a good star gave
                         0.73 and a bad one 0.74.
    force          : return the posterior even if the check fails

    Returns (posterior, info). posterior is a dict of samples of the
    latent and deterministic sites, the same keys the SVI guide gives,
    or None when the check fails (and not force). info records the mode, k-hat, effective
    sample size, timings and the reason for the decision.
    """
    settings = settings or {}
    t0 = time.perf_counter()
    info = {'method': 'laplace'}

    pinfo = initialize_model(rng_key, model, model_kwargs=modelkw,
                             init_strategy=init_to_value(values=initpars))
    x0, unravel = ravel_pytree(pinfo.param_info.z)
    potential = pinfo.potential_fn
    vg = jax.jit(jax.value_and_grad(lambda x: potential(unravel(x))))
    mu, U, nit, converged, ev, V = find_mode(vg, x0)
    # flat or unstable directions: floor the curvature (sigma <= 10 in
    # unconstrained units) so the Gaussian stays proper; k-hat then
    # reports how bad the approximation is
    floor = 1e-2
    info.update({'U_mode': U, 'newton_iter': nit, 'mode_converged': converged,
                 'n_floored': int(np.sum(ev < floor))})
    ev = np.maximum(ev, floor)
    L = V * (1.0 / np.sqrt(ev))                     # L L^T = H^-1
    logdetL = -0.5 * np.sum(np.log(ev))
    info['t_mode'] = time.perf_counter() - t0

    # importance draws in batches of one compiled size
    batch = 2000
    pot_batch = jax.jit(jax.vmap(lambda x: potential(unravel(x))))
    mu_j, L_j = jnp.asarray(mu), jnp.asarray(L)
    d = len(mu)

    nu = settings.get('is_nu', 4.0)

    def draw(key, n):
        # Gaussian (nu None) or multivariate Student-t proposal around the mode
        n = int(np.ceil(n / batch)) * batch
        kz, kg = jrandom.split(key)
        z = jrandom.normal(kz, (n, d))
        if nu is None:
            logq = -0.5 * jnp.sum(z**2, axis=1) - 0.5 * d * jnp.log(2 * jnp.pi)
        else:
            z = z / jnp.sqrt(2.0 * jrandom.gamma(kg, nu / 2.0, (n,)) / nu)[:, None]
            logq = (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi)
                    - 0.5 * (nu + d) * jnp.log1p(jnp.sum(z**2, axis=1) / nu))
        X = mu_j + z @ L_j.T
        U = jnp.concatenate([pot_batch(X[i:i + batch]) for i in range(0, n, batch)])
        return X, np.asarray(-U - (logq - logdetL))

    key_a, key_b, key_r = jrandom.split(rng_key, 3)
    ndraw = int(settings.get('is_draws', 10000))
    nmax = max(int(settings.get('is_draws_max', 40000)), ndraw)
    X, logw = draw(key_a, ndraw)
    lw, k = psis(logw)
    thr = settings.get('k_threshold', k_threshold(len(logw)))
    if thr <= k < thr + 0.1 and nmax > len(logw):
        X2, logw2 = draw(key_b, nmax - len(logw))
        X, logw = jnp.concatenate([X, X2]), np.concatenate([logw, logw2])
        lw, k = psis(logw)
        thr = settings.get('k_threshold', k_threshold(len(logw)))
    w = np.exp(lw)
    info.update({'pareto_k': k, 'k_threshold': thr, 'is_draws': len(logw),
                 'is_ess': float(1.0 / np.sum(w**2)),
                 'n_nonfinite': int(np.sum(~np.isfinite(logw))),
                 't_is': time.perf_counter() - t0 - info['t_mode']})
    info['accepted'] = bool(k < thr)
    if verbose:
        print(f"[uberMS] Laplace + PSIS: mode {'converged' if converged else 'NOT converged'} "
              f"after {nit} Newton steps, Pareto k-hat {k:.2f} (threshold {thr:.2f}), "
              f"ESS {info['is_ess']:.0f} of {len(logw)} draws, "
              f"{time.perf_counter() - t0:.1f} s", flush=True)

    if not info['accepted'] and not force:
        info['t_total'] = time.perf_counter() - t0
        return None, info

    # weighted resample, then the constrained values and deterministics
    idx = np.asarray(jrandom.choice(key_r, len(w), shape=(nsamples,), replace=True, p=jnp.asarray(w)))
    post_batch = jax.jit(jax.vmap(lambda x: pinfo.postprocess_fn(unravel(x))))
    # pad to whole batches so every call reuses one compiled shape
    pad = (-nsamples) % batch
    Xs = X[np.concatenate([idx, np.full(pad, idx[0])])]
    parts = [post_batch(Xs[i:i + batch]) for i in range(0, len(Xs), batch)]
    posterior = {kk: jnp.concatenate([p[kk] for p in parts])[:nsamples] for kk in parts[0]}
    info['t_total'] = time.perf_counter() - t0
    return posterior, info
