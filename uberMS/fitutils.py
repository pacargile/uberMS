import jax
import jax.numpy as jnp
from jax import jit, jacfwd #,lax
from jax import random as jrandom
from numpyro.infer.util import initialize_model, log_density
from numpyro.infer.initialization import init_to_feasible, init_to_sample
import optax
from scipy import constants
speedoflight = constants.c / 1000.0

# spectral fitting modes
SPECMODES = ('normalized','fluxed')

def avrv_spec_layout(nntype):
    """
    True for spectral NNs whose genspec pars carry [..., Av, Rv, ...]
    (MLP_v* and the FlxEm family: 'FlxEm', 'FlxEm_v<N>', alias 'FluxEmulator').
    Mirrors Payne.jax.genmod._avrv_spec_layout.
    """
    return (nntype in ('MLP_v0', 'MLP_v1', 'MLP_v2', 'FluxEmulator')
            or (isinstance(nntype, str) and nntype.startswith('FlxEm')))

def predict_spec(
    genspecfn, specwave, labels, vrad, vstar, vmic, lsf, pcs,
    specmode='normalized', avrv=False,
    Av=0.0, Rv=3.1, logR=0.0, dist=10.0, fluxscale=1.0):
    """
    Predict the model spectrum on specwave.

    labels : [Teff, log(g), [Fe/H], [a/Fe]]
    pcs    : list of Chebyshev coefficients [pc0, pc1, ...], may be empty

    For NNs without Av/Rv inputs (LinNet, YST1) this is the original
    behavior: a normalized model multiplied by the Chebyshev polynomial.

    For NNs that predict F_lambda with Av/Rv inputs (FlxEm, MLP_v*):
    - 'normalized' : Av = 0, Rv = 3.1, the model is divided by its mean
      (so pc0 ~ 1 for a spectrum normalized to ~1) and then multiplied by
      the Chebyshev polynomial, which absorbs the continuum shape.
    - 'fluxed' : the model is reddened by Av, Rv and scaled to the observed
      flux f_lambda = F_lambda * (R/dist)^2 / fluxscale, with R = 10**logR
      in R_sun and dist in pc. The model is in erg/s/cm^2/A, so
      fluxscale = 1.0 means the observed spectrum is in erg/s/cm^2/A.
      If pcs is not empty, the polynomial is applied as a flux-calibration
      correction.
    """
    specpars = list(labels) + [vrad,vstar,vmic,lsf]

    if not avrv:
        specpars += list(pcs)
        return jnp.asarray(genspecfn(specpars,outwave=specwave,modpoly=True)[1])

    if specmode == 'fluxed':
        specpars += [Av,Rv] + list(pcs)
        specmod = genspecfn(
            specpars,outwave=specwave,modpoly=(len(pcs) > 0),logR=logR,dist=dist)[1]
        return jnp.asarray(specmod) / fluxscale

    specpars += [0.0,3.1] + list(pcs)
    specmod = genspecfn(specpars,outwave=specwave,modpoly=True,normalize=True)[1]
    return jnp.asarray(specmod)

def _prior_support(priorinfo):
    """Approximate (low, high) support of a user prior, or None if unknown."""
    try:
        if priorinfo[0] == 'fixed':
            return (float(priorinfo[1]),float(priorinfo[1]))
        if priorinfo[0] == 'uniform':
            return (float(priorinfo[1][0]),float(priorinfo[1][1]))
        if priorinfo[0] == 'tnormal':
            return (float(priorinfo[1][2]),float(priorinfo[1][3]))
        if priorinfo[0] == 'normal':
            return (float(priorinfo[1][0]-3.0*priorinfo[1][1]),
                    float(priorinfo[1][0]+3.0*priorinfo[1][1]))
    except (TypeError,IndexError):
        pass
    return None

def specsetup(
    priors, specmode='normalized', fluxscale=1.0, avrv=False, specmeta={},
    specwave=None, specflux=None, isochrone=False, verbose=True):
    """
    Check the spectral fitting setup for a single spectrum and return a copy
    of priors with model-specific defaults applied. specmeta is the FlxEm
    file metadata. isochrone is True for the MIST models (sviMS, nutsMS).
    """
    specsetup_spec(
        priors,specmode=specmode,fluxscale=fluxscale,avrv=avrv,specmeta=specmeta,
        specwave=specwave,specflux=specflux)
    return specsetup_shared(
        priors,specmetas=[specmeta] if avrv else [],
        anyfluxed=(specmode == 'fluxed'),isochrone=isochrone,verbose=verbose)

def specsetup_spec(
    priors, specmode='normalized', fluxscale=1.0, avrv=False, specmeta={},
    specwave=None, specflux=None, suffix='', label=''):
    """
    Checks for one spectrum. Its lsf and pc priors are named 'lsf'+suffix,
    'pc0'+suffix, etc., and label is prepended to the messages.
    """
    if specmode not in SPECMODES:
        raise ValueError(f"{label}specmode must be one of {SPECMODES}, got '{specmode}'")
    if (specmode == 'fluxed') and (not avrv):
        raise ValueError(
            f"{label}specmode='fluxed' requires a spectral NN that predicts F_lambda with "
            "Av/Rv inputs (e.g., sNNtype='FlxEm_v1')")
    if not avrv:
        return

    # the NN emulator cannot sharpen its spectra, so inst_R must be
    # strictly below the resolution baked into the model
    lsfkey = 'lsf'+suffix
    R_fwhm = specmeta.get('R_fwhm',None)
    if (R_fwhm is not None) and ('lsf_array'+suffix not in priors.keys()):
        if lsfkey in priors.keys():
            lsfsupp = _prior_support(priors[lsfkey])
        else:
            lsfsupp = (32000.0,32000.0)
        if (lsfsupp is not None) and (lsfsupp[1] >= R_fwhm):
            raise ValueError(
                f"{label}{lsfkey} prior extends to R={lsfsupp[1]:.0f} but the spectral NN has "
                f"R_fwhm={R_fwhm:.0f}; set an '{lsfkey}' prior below the model resolution")

    if specmode == 'normalized':
        # a spectrum in physical flux units cannot be fit by a model normalized to ~1
        if specflux is not None:
            medflux = float(jnp.nanmedian(jnp.abs(specflux)))
            if (medflux < 1E-3) or (medflux > 1E+3):
                print(f'[uberMS] WARNING: {label}median spectrum flux is {medflux:.3g}, but in '
                      'normalized mode the model is normalized to ~1 times the polynomial; for a '
                      "flux-calibrated spectrum use specmode='fluxed'")
        # the model carries the full SED shape, which the polynomial has to absorb
        if specwave is not None:
            wratio = float(jnp.max(specwave)/jnp.min(specwave))
            if wratio > 1.5:
                print(f'[uberMS] WARNING: {label}in normalized mode the Chebyshev polynomial must '
                      f'absorb the SED shape of the F_lambda model over '
                      f'{float(jnp.min(specwave)):.0f}-{float(jnp.max(specwave)):.0f} A; '
                      'if the input spectrum is continuum-normalized, check that '
                      'the number of pc terms is sufficient')
    else:
        # pc0 is fixed to 1 in fluxed mode unless the user frees it
        pc0key = 'pc0'+suffix
        if (pc0key in priors.keys()) and (priors[pc0key][0] != 'fixed'):
            print(f'[uberMS] WARNING: {label}a free {pc0key} is degenerate with (R/dist)^2 in fluxed mode')
        # the model is in erg/s/cm^2/A, so a spectrum normalized to ~1 or in
        # other units shows up as an implausibly bright star
        if specflux is not None:
            medflux = float(jnp.nanmedian(jnp.abs(specflux)))*fluxscale
            if medflux > 1E-5:
                print(f'[uberMS] WARNING: {label}median spectrum flux is {medflux:.3g} '
                      'erg/s/cm^2/A, is the spectrum normalized or in other units? '
                      'Use fluxscale to set the units of the input spectrum')

def specsetup_shared(priors, specmetas=[], anyfluxed=False, isochrone=False, verbose=True):
    """
    Checks for the parameters shared by all spectra, returns a copy of priors.
    specmetas is the list of FlxEm file metadata for the spectral NNs.
    Default label priors are narrowed to the range covered by all of these
    NNs, and user priors outside of it give a warning.

    With isochrone=True the labels are predicted by MIST, and a prior on
    them is an extra factor in the model rather than its sampling range,
    so the defaults are not narrowed and the NN range is only reported.
    """
    import json
    priors = dict(priors)

    ranges = {}
    for meta in specmetas:
        if 'input_ranges' not in meta.keys():
            continue
        for lab,(lo,hi) in json.loads(meta['input_ranges']).items():
            if lab in ranges.keys():
                ranges[lab] = [max(lo,ranges[lab][0]),min(hi,ranges[lab][1])]
            else:
                ranges[lab] = [lo,hi]

    defaults = ({
        'Teff':  ('logt',[2500.0,10000.0]),
        'log(g)':('logg',[0.0,5.5]),
        '[Fe/H]':('feh',[-3.0,0.49]),
        '[a/Fe]':('afe',[-0.19,0.59]),
        })
    unbounded = []
    for parname,(lab,default) in defaults.items():
        if lab not in ranges.keys():
            continue
        lo,hi = ranges[lab]
        if lab == 'logt':
            lo,hi = 10.0**lo,10.0**hi
        if parname in priors.keys():
            supp = _prior_support(priors[parname])
            if (supp is not None) and ((supp[0] < lo) or (supp[1] > hi)):
                print(f'[uberMS] WARNING: prior on {parname} extends outside of '
                      f'the spectral NN range [{lo:.4g}, {hi:.4g}]')
        elif isochrone:
            unbounded.append(f'{parname} [{lo:.4g}, {hi:.4g}]')
        else:
            priors[parname] = ['uniform',[max(lo,default[0]),min(hi,default[1])]]
            if verbose:
                print(f'[uberMS] ... default {parname} prior set to spectral NN '
                      f'range: {priors[parname][1]}')
    if (len(unbounded) > 0) and verbose:
        print('[uberMS] ... spectral NN range: ' + ', '.join(unbounded) + '; MIST '
              'predictions outside of it are extrapolated by the NN, set priors on '
              'these parameters to restrict them')

    # G23 extinction law is only defined for 2.3 < Rv < 5.6
    if anyfluxed and ('Rv' in priors.keys()):
        supp = _prior_support(priors['Rv'])
        if (supp is not None) and ((supp[0] < 2.3) or (supp[1] > 5.6)):
            print('[uberMS] WARNING: Rv prior extends outside of 2.3-5.6, '
                  'the range of the G23 extinction law used by the spectral NN')

    return priors

def pcterms_from_priors(priors, specmode='normalized', suffix=''):
    """
    Chebyshev terms to sample: pc0..pcN (named 'pc0'+suffix, ...) where N is
    the highest pc index set in priors. Without pc priors, normalized mode
    uses pc0-pc3 and fluxed mode applies no polynomial.
    """
    pcinds = [int(kk[2:len(kk)-len(suffix)]) for kk in priors.keys()
              if (kk[:2] == 'pc') and kk.endswith(suffix) and kk[2:len(kk)-len(suffix)].isdigit()]
    if len(pcinds) == 0:
        if specmode == 'fluxed':
            return []
        return ['pc{0}{1}'.format(x,suffix) for x in range(4)]
    return ['pc{0}{1}'.format(x,suffix) for x in range(max(pcinds)+1)]

class SEDopt(object):
    def __init__(self, **kwargs):
        super(SEDopt, self).__init__()

        self.photobs    = kwargs.get('photobs',{})
        self.filtarray  = kwargs.get('filterarray',{})

        self.priors  = kwargs.get('priors',{})
        self.additionalinfo = kwargs.get('additionalinfo',{})

        # pull out fitting functions
        self.genphotfn = jax.jit(kwargs.get('genphotfn',None))

        self.returnsed = kwargs.get('returnsed',False)

        self.init_p0 = kwargs.get('initpars',
            {'Teff':6000.0,'log(g)':4.0,'[Fe/H]':0.0,'[a/Fe]':0.2,'log(R)':0.0,'dist':1000.0,'Av':0.0}
            )

        # figure out fixed parameters
        self.fixedpars = {}
        for kk in self.priors.keys():
            if self.priors[kk][0] == 'fixed':
                self.fixedpars[kk] = self.priors[kk][1]

        # all possible fit parameters
        fitpars_i = ['Teff','log(g)','[Fe/H]','[a/Fe]','log(R)','dist','Av']

        # fit parameters without fixedpars
        self.fitpars =  [x for x in fitpars_i if x not in self.fixedpars.keys()]
        
        if len(self.fitpars) + len(list(self.fixedpars.keys())) != len(fitpars_i):
            print('ALL PARS MUST BE EITHER FIT OR FIXED')
            print('FIT PARS:',self.fitpars)
            print('FIX PARS:',list(self.fixedpars.keys()))
            raise(IOError)
        
        self.verbose = kwargs.get('verbose',False)
        
    def __call__(self):
        p0 = jnp.array([self.init_p0[x] for x in self.fitpars])

        # init the optimizer        
        start_learning_rate = 1e-3
        optimizer = optax.adam(start_learning_rate)

        # set initial pars
        opt_state = optimizer.init(p0)
        
        # define step func
        @jax.jit
        def step(params,opt_state):
            loss_value,grads = jax.value_and_grad(self.chisq_sed)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params,opt_state,loss_value            
        
        for ii in range(10000):
            if ii == 0:
                params = p0
            params,opt_state,loss_value = step(params,opt_state)
            if self.verbose:
                if ii % 100 == 0:
                    print(f'step - {ii}, loss - {loss_value}')

        params_dict = {x:y for x,y in zip(self.fitpars,params)}
        return (params_dict,loss_value)

    # define the model
    def chisq_sed(self,pars):
        fitpars_dict = {}
        for ii,x in enumerate(self.fitpars):
            fitpars_dict[x] = pars[ii]

        parsdict = dict(**fitpars_dict,**self.fixedpars)

        # make photometry prediction
        photpars = ([
            parsdict['Teff'],parsdict['log(g)'],parsdict['[Fe/H]'],parsdict['[a/Fe]'],
            parsdict['log(R)'],parsdict['dist'],parsdict['Av'],3.1])
        photmod = self.genphotfn(photpars)

        chisq = jnp.sum(
            jnp.array([((photmod[kk]-self.photobs[kk][0])**2.0)/(self.photobs[kk][1]**2.0) 
            for kk in self.filtarray])
            )
        
        return chisq 

class RVopt(object):
    def __init__(self, **kwargs):
        super(RVopt, self).__init__()

        self.wave = jnp.asarray(kwargs.get('inwave',[]))
        self.flux = jnp.asarray(kwargs.get('influx',[]))
        self.eflux = jnp.asarray(kwargs.get('ineflux',[]))
        self.modflux = jnp.asarray(kwargs.get('modflux',[]))
        self.modwave = jnp.asarray(kwargs.get('modwave',[]))

        self.init_p0 = kwargs.get('initvel',jnp.arange(-500,500,25))
        
        self.verbose = kwargs.get('verbose',False)
        
    def __call__(self):
        if type(self.init_p0) is float:
            p0 = [jnp.array([self.init_p0])]
        else:
            p0 = [jnp.array([float(x)]) for x in self.init_p0]
        

        # init the optimizer        
        start_learning_rate = 1e-1
        optimizer = optax.adam(start_learning_rate)

        outval = []
        outlos = []
        for p0_i in p0:
            # set initial pars
            opt_state = optimizer.init(p0_i)
            
            # define step func
            @jax.jit
            def step(params,opt_state):
                loss_value,grads = jax.value_and_grad(self.chisq_rv)(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params,opt_state,loss_value            
            
            for ii in range(1000):
                if ii == 0:
                    params = p0_i
                params,opt_state,loss_value = step(params,opt_state)
                if self.verbose:
                    if ii % 100 == 0:
                        print(f'step - {ii}, loss - {loss_value}, p0_i - {p0_i[0]}, p0_f - {params[0]}')

            outval.append(params)
            outlos.append(loss_value)
        outloss_ind = jnp.argmin(jnp.array(outlos))
        return (outval[outloss_ind],outlos[outloss_ind])

    def chisq_rv(self,pars):
        
        rv = pars[0]
        
        wave = self.wave
        flux = self.flux
        eflux = self.eflux
        modflux = self.modflux
        modwave = self.modwave

        # adjust model to new rv
        modwave_i = modwave*(1.0+(rv/speedoflight))

        # interpolate it back to wave so that chi-sq can be computed
        modflux_i = jnp.interp(wave,modwave_i,modflux,left=jnp.nan,right=jnp.nan)

        chisq = jnp.nansum( ((modflux_i-flux)**2.0) / (eflux**2.0) ) / jnp.isfinite(modflux_i).sum()

        return chisq


def check_initpars(model, modelkw, initpars, rng_key=None, verbose=True):
    """
    Check the initial parameter guesses against the model priors.

    Runs the model once with its free parameters set to initpars, and
    raises a ValueError listing every initial value that falls outside its
    prior support or gives a non-finite log-probability. Free parameters
    without an initial value are drawn from their priors for this check.
    initpars keys that are not free parameters of the model are ignored by
    the fit, and are listed as warnings.
    """
    from numpyro import handlers
    from numpyro.distributions.transforms import biject_to

    if rng_key is None:
        rng_key = jrandom.PRNGKey(0)

    tr = handlers.trace(
        handlers.substitute(handlers.seed(model,rng_key),data=initpars)
        ).get_trace(**modelkw)

    def fmt(x):
        x = jnp.asarray(x)
        return f'{float(x):.6g}' if x.size == 1 else str(x)

    latent = [kk for kk,site in tr.items()
              if (site['type'] == 'sample') and (not site['is_observed'])]

    def fmtsupport(support):
        lo = getattr(support,'lower_bound',None)
        hi = getattr(support,'upper_bound',None)
        if (lo is not None) and (hi is not None):
            return f'prior range [{fmt(lo)}, {fmt(hi)}]'
        if lo is not None:
            return f'prior lower bound {fmt(lo)}'
        if hi is not None:
            return f'prior upper bound {fmt(hi)}'
        return f'prior support {type(support).__name__}'

    problems = []
    for kk in latent:
        if kk not in initpars.keys():
            continue
        site = tr[kk]
        support = site['fn'].support
        if not bool(jnp.all(support(site['value']))):
            problems.append(f'{kk} = {fmt(site["value"])} is outside the {fmtsupport(support)}')
            continue
        # the fit starts from the value mapped to an unconstrained space,
        # which is infinite for a value on the edge of the prior support
        try:
            uval = biject_to(support).inv(site['value'])
        except NotImplementedError:
            uval = None
        if (uval is not None) and (not bool(jnp.all(jnp.isfinite(uval)))):
            problems.append(f'{kk} = {fmt(site["value"])} is on the edge of the {fmtsupport(support)}, '
                            'the initial value must be strictly inside it')
            continue
        lp = jnp.sum(site['fn'].log_prob(site['value']))
        if not bool(jnp.isfinite(lp)):
            problems.append(f'{kk} = {fmt(site["value"])} gives a prior log-probability of {fmt(lp)}')

    # factors and likelihoods evaluated at the initial values
    if len(problems) == 0:
        for kk,site in tr.items():
            if (site['type'] == 'sample') and site['is_observed']:
                lp = jnp.sum(site['fn'].log_prob(site['value']))
                if not bool(jnp.isfinite(lp)):
                    problems.append(f'{kk} has a log-probability of {fmt(lp)} at the initial values, '
                                    'check the data and errors for NaN/inf values')

    # the fit also needs a finite gradient at the initial values
    if len(problems) == 0:
        latvals = {kk:tr[kk]['value'] for kk in latent}
        grads = jax.grad(lambda pp: log_density(model,(),modelkw,pp)[0])(latvals)
        badgrad = [kk for kk,gg in grads.items() if not bool(jnp.all(jnp.isfinite(gg)))]
        if len(badgrad) > 0:
            problems.append(
                f'the gradient of the log-probability is not finite for {badgrad}; this usually '
                'means the model and data are far apart or in different units (e.g., a fluxed '
                "spectrum fit with specmode='normalized')")

    if verbose:
        for kk in initpars.keys():
            if kk in latent:
                continue
            if kk in tr.keys():
                print(f'[uberMS] WARNING: initial value for {kk} is ignored, it is fixed or derived in this model')
            else:
                print(f'[uberMS] WARNING: initial value for {kk} is ignored, it is not a parameter of this model')

    if len(problems) > 0:
        missing = [kk for kk in latent if kk not in initpars.keys()]
        msg = '[uberMS] the initial parameters are not a valid starting point for the model:\n'
        msg += '\n'.join(f'  - {pp}' for pp in problems)
        if len(missing) > 0:
            msg += f'\n  (parameters without initial values were drawn from their priors: {missing})'
        raise ValueError(msg)

def check_numpyro_init(model, modelkw, initpars, rng_key=None, verbose=True):
    """
    Check that numpyro can start the fit from initpars.

    check_initpars reports the problems it can trace to a parameter, this is
    the final check of the full model in numpyro's unconstrained space. If it
    fails, it lists the non-finite values, checks if the model can be started
    from draws of its priors, and raises a RuntimeError.
    """
    import numpyro
    from numpyro.infer.util import find_valid_initial_params
    from numpyro.infer.initialization import init_to_value

    if rng_key is None:
        rng_key = jrandom.PRNGKey(0)

    (z,pe,zgrad),valid = find_valid_initial_params(
        rng_key,model,init_strategy=init_to_value(values=initpars),model_kwargs=modelkw)
    if bool(valid):
        return

    problems = []
    if not bool(jnp.isfinite(pe)):
        problems.append(f'the log-probability of the model is {-float(pe)}')
    badz = [kk for kk,vv in z.items() if not bool(jnp.all(jnp.isfinite(vv)))]
    if len(badz) > 0:
        problems.append(f'non-finite unconstrained values for {badz}')
    badg = [kk for kk,vv in zgrad.items() if not bool(jnp.all(jnp.isfinite(vv)))]
    if len(badg) > 0:
        problems.append(f'non-finite gradients for {badg}')
    if len(problems) == 0:
        problems.append('numpyro rejected the initial values')
    missing = [kk for kk in z.keys() if kk not in initpars.keys()]
    if len(missing) > 0:
        problems.append(f'parameters without initial values were drawn at random: {missing}')

    # can the model be started anywhere?
    if verbose:
        print('[uberMS] ... numpyro could not start from the initial values, '
              'trying draws from the priors')
    numpyro.enable_validation(True)
    try:
        feasible = quick_model_eval(model,mkwargs=modelkw,verbose=verbose)
    finally:
        numpyro.enable_validation(False)
    if feasible is None:
        problems.append('the model could not be started from draws of its priors either, '
                        'check the data, errors, and priors')
    else:
        problems.append('the model can be started from draws of its priors, so the problem '
                        'is in the initial values')

    msg = '[uberMS] numpyro cannot start the fit from the initial parameters:\n'
    msg += '\n'.join(f'  - {pp}' for pp in problems)
    raise RuntimeError(msg)

def quick_model_eval(model, margs=(), mkwargs=None, tries=4, verbose=True):
    """
    Returns a valid set of initial *constrained* latent values drawn from
    the priors, or None if none was found, and prints diagnostics
    (potential, per-site log-probs).
    """
    mkwargs = {} if mkwargs is None else mkwargs
    strategies = [("init_to_feasible", init_to_feasible()),
                  ("init_to_sample",   init_to_sample())]

    for name, strat in strategies:
        for seed in range(tries):
            key = jrandom.PRNGKey(seed)
            try:
                modelinfo = initialize_model(
                        key, model,
                        init_strategy=strat,
                        model_args=margs, model_kwargs=mkwargs,
                    )
                U = modelinfo.param_info.potential_energy  # negative log joint at init
                constrained = modelinfo.postprocess_fn(modelinfo.param_info.z)

                # Per-site log-probs
                total_lp, trace = log_density(model, margs, mkwargs, constrained)
                bad_sites = [k for k, site in trace.items() if site['type'] == 'sample'
                             and not jnp.isfinite(jnp.sum(site['fn'].log_prob(site['value'])))]

                if verbose:
                    print(f"[uberMS] [{name}, seed={seed}] potential={float(U)} "
                          f"(finite={bool(jnp.isfinite(U))}); bad sites={bad_sites}")

                if jnp.isfinite(U) and not bad_sites:
                    return constrained  # looks good!

            except Exception as e:
                if verbose:
                    print(f"[uberMS] [{name}, seed={seed}] FAILED during init → {e}")

    return None
