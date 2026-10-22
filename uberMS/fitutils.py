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
    specwave=None, specflux=None, verbose=True):
    """
    Check the spectral fitting setup and return a copy of priors with
    model-specific defaults applied. specmeta is the FlxEm file metadata.
    """
    if specmode not in SPECMODES:
        raise ValueError(f"specmode must be one of {SPECMODES}, got '{specmode}'")
    if (specmode == 'fluxed') and (not avrv):
        raise ValueError(
            "specmode='fluxed' requires a spectral NN that predicts F_lambda with "
            "Av/Rv inputs (e.g., sNNtype='FlxEm_v1')")

    priors = dict(priors)
    if not avrv:
        return priors

    # the NN emulator cannot sharpen its spectra, so inst_R must be
    # strictly below the resolution baked into the model
    R_fwhm = specmeta.get('R_fwhm',None)
    if (R_fwhm is not None) and ('lsf_array' not in priors.keys()):
        if 'lsf' in priors.keys():
            lsfsupp = _prior_support(priors['lsf'])
        else:
            lsfsupp = (32000.0,32000.0)
        if (lsfsupp is not None) and (lsfsupp[1] >= R_fwhm):
            raise ValueError(
                f"lsf prior extends to R={lsfsupp[1]:.0f} but the spectral NN has "
                f"R_fwhm={R_fwhm:.0f}; set an 'lsf' prior below the model resolution")

    # model label ranges: narrow the default priors to the training grid,
    # and warn if a user prior extends outside of it
    if 'input_ranges' in specmeta.keys():
        import json
        ranges = json.loads(specmeta['input_ranges'])
        defaults = ({
            'Teff':  ('logt',[2500.0,10000.0]),
            'log(g)':('logg',[0.0,5.5]),
            '[Fe/H]':('feh',[-3.0,0.49]),
            '[a/Fe]':('afe',[-0.19,0.59]),
            })
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
            else:
                priors[parname] = ['uniform',[max(lo,default[0]),min(hi,default[1])]]
                if verbose:
                    print(f'[uberMS] ... default {parname} prior set to spectral NN '
                          f'range: {priors[parname][1]}')

    if specmode == 'normalized':
        # the model carries the full SED shape, which the polynomial has to absorb
        if specwave is not None:
            wratio = float(jnp.max(specwave)/jnp.min(specwave))
            if wratio > 1.5:
                print(f'[uberMS] WARNING: in normalized mode the Chebyshev polynomial must '
                      f'absorb the SED shape of the F_lambda model over '
                      f'{float(jnp.min(specwave)):.0f}-{float(jnp.max(specwave)):.0f} A; '
                      'if the input spectrum is continuum-normalized, check that '
                      'the number of pc terms is sufficient')
    else:
        # pc0 is fixed to 1 in fluxed mode unless the user frees it
        if ('pc0' in priors.keys()) and (priors['pc0'][0] != 'fixed'):
            print('[uberMS] WARNING: a free pc0 is degenerate with (R/dist)^2 in fluxed mode')
        # G23 extinction law is only defined for 2.3 < Rv < 5.6
        if 'Rv' in priors.keys():
            supp = _prior_support(priors['Rv'])
            if (supp is not None) and ((supp[0] < 2.3) or (supp[1] > 5.6)):
                print('[uberMS] WARNING: Rv prior extends outside of 2.3-5.6, '
                      'the range of the G23 extinction law used by the spectral NN')
        # the model is in erg/s/cm^2/A, so a spectrum normalized to ~1 or in
        # other units shows up as an implausibly bright star
        if specflux is not None:
            medflux = float(jnp.nanmedian(jnp.abs(specflux)))*fluxscale
            if medflux > 1E-5:
                print(f'[uberMS] WARNING: median spectrum flux is {medflux:.3g} '
                      'erg/s/cm^2/A, is the spectrum normalized or in other units? '
                      'Use fluxscale to set the units of the input spectrum')

    return priors

def pcterms_from_priors(priors, specmode='normalized'):
    """
    Chebyshev terms to sample: pc0..pcN where N is the highest pc index set
    in priors. Without pc priors, normalized mode uses pc0-pc3 and fluxed
    mode applies no polynomial.
    """
    pcinds = [int(kk[2:]) for kk in priors.keys() if (kk[:2] == 'pc') and kk[2:].isdigit()]
    if len(pcinds) == 0:
        if specmode == 'fluxed':
            return []
        return ['pc0','pc1','pc2','pc3']
    return ['pc{0}'.format(x) for x in range(max(pcinds)+1)]

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


def quick_model_eval(model, margs=(), mkwargs=None, tries=4):
    """
    Returns a valid set of initial *constrained* latent values if possible,
    and prints diagnostics (potential, per-site log-probs).
    """
    mkwargs = {} if mkwargs is None else mkwargs
    strategies = [("init_to_feasible", init_to_feasible()),
                  ("init_to_sample",   init_to_sample())]

    for name, strat in strategies:
        for seed in range(tries):
            key = jrandom.PRNGKey(seed)
            try:
                (uc_params, potential_fn_gen, postprocess_fn, _trace, _) = initialize_model(
                        key, model,
                        dynamic_args=True,
                        init_strategy=strat,
                        model_args=margs, model_kwargs=mkwargs,
                    )
                potential_fn = potential_fn_gen(uc_params)
                U = potential_fn(uc_params)  # negative log joint at init
                constrained = postprocess_fn(uc_params)

                # Per-site log-probs
                total_lp, site_lps = log_density(model, margs, mkwargs, constrained)
                bad_sites = [k for k, v in site_lps.items() if not jnp.isfinite(v).all()]

                print(f"[{name}, seed={seed}] potential={float(U)} "
                      f"(finite={bool(jnp.isfinite(U))}); "
                      f"sites={list(site_lps)}; bad={bad_sites}")

                if jnp.isfinite(U) and not bad_sites:
                    return constrained  # looks good!

            except Exception as e:
                print(f"[{name}, seed={seed}] FAILED during init → {e}")

    raise RuntimeError("Could not find a finite initial state for the model.")