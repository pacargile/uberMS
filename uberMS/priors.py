from .advancedpriors import IMF_Prior,IMF_logg_Prior,Gal_Prior,Sigmoid_Prior,DSigmoid_Prior,PlateauNormal

import numpyro
import numpyro.distributions as distfn
import jax.numpy as jnp

def defaultprior(parname):
    # define defaults for sampled parameters
    if parname == "EEP":
        return numpyro.sample("EEP", distfn.Uniform(300,800))
    if parname == "initial_Mass":
        return numpyro.sample('initial_Mass', IMF_Prior())
    if parname == "initial_[Fe/H]":
        return numpyro.sample("initial_[Fe/H]", distfn.Uniform(-3.0,0.49))
    if parname == "initial_[a/Fe]":
        return numpyro.sample("initial_[a/Fe]", distfn.Uniform(-0.19,0.59))
    if parname == "vmic":
        return numpyro.sample("vmic", distfn.Uniform(0.5, 3.0))
    if parname == "vrad":
        return numpyro.sample("vrad", distfn.Uniform(-500.0, 500.0))
    if parname == "vstar":
        return numpyro.sample("vstar", distfn.Uniform(0.0, 25.0))
    if parname == "pc0":
        return numpyro.sample("pc0", distfn.Uniform(0.5, 2.0))
    if parname == "pc1":
        return numpyro.sample("pc1", distfn.Normal(0.0, 0.25))
    if parname == "pc2":
        return numpyro.sample("pc2", distfn.Normal(0.0, 0.25))
    if parname == "pc3":
        return numpyro.sample("pc3", distfn.Normal(0.0, 0.25))
    if parname == "Av":
        return numpyro.sample("Av", distfn.Uniform(1E-6,5.0))
    if parname == "Rv":
        return numpyro.deterministic("Rv",3.1)
    if parname == "lsf":
        return numpyro.sample("lsf", distfn.Normal(32000.0,1000.0))
    if parname == "Teff":
        return numpyro.sample("Teff", distfn.Uniform(2500.0, 10000.0))
    if parname == "log(g)":
        return numpyro.sample("log(g)", distfn.Uniform(0.0, 5.5))
    if parname == "[Fe/H]":
        return numpyro.sample("[Fe/H]", distfn.Uniform(-3.0,0.49))
    if parname == "[a/Fe]":
        return numpyro.sample("[a/Fe]", distfn.Uniform(-0.19,0.59))        
    if parname == 'log(R)':
        return numpyro.sample("log(R)", distfn.Uniform(-2,3.0))  
    if parname == 'dist':
        return numpyro.sample("dist", distfn.Uniform(1,200000.0))  
    if parname == "specjitter":
        # return numpyro.sample("specjitter", distfn.HalfNormal(0.001))
        return numpyro.deterministic("specjitter",0.0)  # default to no jitter
    if parname == "photjitter":
        # return numpyro.sample("photjitter", distfn.HalfNormal(0.001))
        return numpyro.deterministic("photjitter",0.0)  # default to no jitter
    if parname == "teffjitter":
        return numpyro.deterministic("teffjitter",0.0)  # default to no jitter        
    if parname == "loggjitter":
        return numpyro.deterministic("loggjitter",0.0)  # default to no jitter
    if parname == "logrjitter":
        return numpyro.deterministic("logrjitter",0.0)


def determineprior(parname,priorinfo,*args):
    # advanced priors
    if (parname == 'Av') and (priorinfo[0] == '3D'):
        # 3D dust map prior on Av conditioned on distance, passed in as args[0],
        # the table along the line of sight is built by dustprior_setup
        if len(args) == 0:
            raise ValueError("3D Av prior requires dist, which must be sampled before Av")
        tab = priorinfo[1]
        if 'dist_grid' not in tab.keys():
            raise ValueError(
                "3D Av prior table has not been built, run the priors through "
                "uberMS.dustprior.dustprior_setup or pass a table from build_av3d_table")
        dist = args[0]
        dgrid = jnp.asarray(tab['dist_grid'])
        # flat between Av_lo and Av_hi with Gaussian tails of width Av_sigma,
        # Av_lo = Av_hi = map mean where the 3D maps are reliable, closer
        # than that the prior is flat from 0 up to the Av at the first
        # reliable distance
        Av_lo  = jnp.interp(dist,dgrid,jnp.asarray(tab['Av_lo']))
        Av_hi  = jnp.interp(dist,dgrid,jnp.asarray(tab['Av_hi']))
        Av_sig = jnp.interp(dist,dgrid,jnp.asarray(tab['Av_sigma']))
        numpyro.deterministic('Av_3Dmap',jnp.interp(dist,dgrid,jnp.asarray(tab['Av_mean'])))
        return numpyro.sample(parname,PlateauNormal(
            Av_lo,Av_hi,Av_sig,low=tab['Av_ll'],high=tab['Av_ul']))

    if (parname == 'log(g)') and (priorinfo[0] == 'IMF'):
        # IMF-induced prior on log(g) given log(R), passed in as args[0]
        if len(args) == 0:
            raise ValueError(
                "log(g) IMF prior requires log(R), only available for models that sample log(R)")
        logR = args[0]
        imfkw = priorinfo[1] if len(priorinfo) > 1 else {}
        logg = numpyro.sample(parname,IMF_logg_Prior(logR=logR,**imfkw))
        numpyro.deterministic("Mass",10.0**(logg - IMF_logg_Prior.logg_sol + 2.0*logR))
        return logg

    if (priorinfo[0] == 'IMF'):
        mass_le,mass_ue = priorinfo[1]['mass_le'],priorinfo[1]['mass_ue']
        return numpyro.sample("initial_Mass",IMF_Prior(low=mass_le,high=mass_ue))

    if (priorinfo[0] == 'GAL'):
        dist_le,dist_ue = priorinfo[1]['dist_ll'],priorinfo[1]['dist_ul']
        GP = Gal_Prior(l=priorinfo[1]['l'],b=priorinfo[1]['b'],low=dist_le,high=dist_ue)
        return numpyro.sample("dist",GP)

    if (priorinfo[0] == 'GALAGE'):
        GP = Gal_Prior(l=priorinfo[1]['l'],b=priorinfo[1]['b'])
        return numpyro.sample("dist",GP)

    if parname == 'vmic':
        # check to see if user wants to use relationship for vmic
        if (priorinfo[0] == 'Bruntt2012'):
            teff = args[0]
            logg = args[1]
            vmic_pred = 1.095 + (5.44E-4) * (teff-5700.0) + (2.56E-7) * (teff-5700.0)**2.0 - 0.378 * (logg - 4.0)
            if priorinfo[1] == 'fixed':
                return numpyro.deterministic(parname,vmic_pred)
            if priorinfo[1] == 'normal':
                return numpyro.sample(parname,distfn.TruncatedDistribution(
                    distfn.Normal(loc=vmic_pred,scale=0.1),
                    low=0.5,high=3.0))

    if parname == 'log(R)':
        # check to see if user wants to use Boyajian2012
        if (priorinfo[0] == 'Boyajian2012'):
            teff = args[0]
            R_pred = -10.8828 + 7.18727 * 1e-3 * teff - 1.50957 * 1e-6 * teff**2 + 1.07572 * 1e-10 * teff**3
            if priorinfo[1] == 'fixed':
                return numpyro.deterministic(parname,jnp.log10(R_pred))
            if priorinfo[1] == 'normal':
                return numpyro.sample(parname,distfn.TruncatedDistribution(
                    distfn.Normal(loc=jnp.log10(R_pred),scale=0.04),
                    low=-3.0,high=3.0))

    if parname == 'log(g)':
        # check to see if user wants to use Boyajian2012
        if (priorinfo[0] == 'Boyajian2012'):
            teff = args[0]
            R_pred = -10.8828 + 7.18727 * 1e-3 * teff - 1.50957 * 1e-6 * teff**2 + 1.07572 * 1e-10 * teff**3
            M_pred = (-0.6063 + jnp.sqrt(1.28*R_pred + 0.2516))/0.64
            g_pred = 6.67430e-8 * M_pred * 1.989e33 / (R_pred*6.955e10)**2
            if priorinfo[1] == 'fixed':
                return numpyro.deterministic(parname,jnp.log10(g_pred))
            if priorinfo[1] == 'normal':
                return numpyro.sample(parname,distfn.TruncatedDistribution(
                    distfn.Normal(loc=jnp.log10(g_pred),scale=0.05),
                    low=0.0,high=5.5))
            
    # handle lsf properly
    if parname == "lsf_array":
        return jnp.asarray(priorinfo[0]) * numpyro.sample(
            "lsf_scaling",distfn.Normal(*priorinfo[1]))

    # define user defined priors

    # standard prior distributions
    if priorinfo[0] == 'uniform':
        return numpyro.sample(parname,distfn.Uniform(*priorinfo[1]))
    if priorinfo[0] == 'normal':
        return numpyro.sample(parname,distfn.Normal(*priorinfo[1]))
    if priorinfo[0] == 'halfnormal':
        return numpyro.sample(parname,distfn.HalfNormal(priorinfo[1]))
    if priorinfo[0] == 'tnormal':
        return numpyro.sample(parname,distfn.TruncatedDistribution(
            distfn.Normal(loc=priorinfo[1][0],scale=priorinfo[1][1]),
            low=priorinfo[1][2],high=priorinfo[1][3]))
    if priorinfo[0] == 'sigmoid':
        return numpyro.sample(parname,Sigmoid_Prior(
            a=priorinfo[1][0],b=priorinfo[1][1],
            low=priorinfo[1][2],high=priorinfo[1][3]))
    if priorinfo[0] == 'dsigmoid':
        return numpyro.sample(parname,DSigmoid_Prior(
            a=priorinfo[1][0],b=priorinfo[1][1],
            c=priorinfo[1][2],d=priorinfo[1][3],
            low=priorinfo[1][4],high=priorinfo[1][5]))
        
    if priorinfo[0] == 'fixed':
        return numpyro.deterministic(parname,priorinfo[1])


LATENT_PRIOR_KINDS = ('uniform', 'normal', 'tnormal', 'sigmoid', 'dsigmoid')

def latent_prior_dist(parname, priorinfo):
    """
    numpyro distribution for a user prior on a parameter that the MIST
    models predict (Teff, log(g), [Fe/H], [a/Fe], log(R), Age, log(Age)).
    The models add its log_prob to the posterior with numpyro.factor.
    """
    kind, args = priorinfo[0], priorinfo[1]
    if kind == 'uniform':
        return distfn.Uniform(low=args[0], high=args[1], validate_args=True)
    if kind == 'normal':
        return distfn.Normal(loc=args[0], scale=args[1])
    if kind == 'tnormal':
        return distfn.TruncatedDistribution(
            distfn.Normal(loc=args[0], scale=args[1]),
            low=args[2], high=args[3], validate_args=True)
    if kind == 'sigmoid':
        return Sigmoid_Prior(a=args[0], b=args[1], low=args[2], high=args[3],
                             validate_args=True)
    if kind == 'dsigmoid':
        return DSigmoid_Prior(a=args[0], b=args[1], c=args[2], d=args[3],
                              low=args[4], high=args[5], validate_args=True)
    raise ValueError(
        f"[uberMS] prior '{kind}' on {parname} is not supported for parameters "
        f"predicted by MIST, use one of {LATENT_PRIOR_KINDS}")


def photjitprior(pjprior):
    # if pjprior is a single list, treat as a global jitter term
    if isinstance(pjprior,list):
        return {'photjitter':determineprior('photjitter',pjprior)}
    elif isinstance(pjprior,dict):
        # user input in dictionary, must mean they want system/band
        # specific jitter terms
        
        # start with the jitter term for all bands
        # not included in prior dict
        outdict = {}
        pjpriorkeys = list(pjprior.keys())
        if 'global' in pjpriorkeys:
            outdict['photjitter'] = determineprior('photjitter',pjprior['global'])
        else:
            outdict['photjitter'] = numpyro.deterministic('photjitter',0.0)
        # remove global from list
        if 'global' in pjpriorkeys:
            pjpriorkeys.remove('global')
        
        # now set prior on filters or systems
        for kk in pjpriorkeys:
            outdict[f'photjitter_{kk}'] = determineprior(f'photjitter_{kk}',pjprior[kk])

        # return the draws
        return outdict


# ------------------------------------------------------------------------
# log(R) sampled as log(R/dist) = log(R) - log10(dist), with dist

# fixed sampling range of log(R/dist), covers log(R) in [-3,5] for any
# distance between 1 pc and 1E10 pc
LOGRDIST_RANGE = (-15.0,5.0)

def logR_prior_dist(priorinfo, teff=None):
    """
    numpyro distribution of the prior on log(R), or None if log(R) is fixed.
    Supports the default prior and the 'uniform', 'normal', 'tnormal',
    'sigmoid', 'dsigmoid', and 'Boyajian2012' priors.
    """
    if priorinfo is None:
        # same as defaultprior('log(R)')
        return distfn.Uniform(-2.0,3.0)
    kind = priorinfo[0]
    if kind == 'fixed':
        return None
    if kind == 'Boyajian2012':
        if priorinfo[1] == 'fixed':
            return None
        R_pred = -10.8828 + 7.18727 * 1e-3 * teff - 1.50957 * 1e-6 * teff**2 + 1.07572 * 1e-10 * teff**3
        return distfn.TruncatedDistribution(
            distfn.Normal(loc=jnp.log10(R_pred),scale=0.04),low=-3.0,high=3.0)
    if kind == 'uniform':
        return distfn.Uniform(*priorinfo[1])
    if kind == 'normal':
        return distfn.Normal(*priorinfo[1])
    if kind == 'tnormal':
        return distfn.TruncatedDistribution(
            distfn.Normal(loc=priorinfo[1][0],scale=priorinfo[1][1]),
            low=priorinfo[1][2],high=priorinfo[1][3])
    if kind == 'sigmoid':
        return Sigmoid_Prior(a=priorinfo[1][0],b=priorinfo[1][1],low=priorinfo[1][2],high=priorinfo[1][3])
    if kind == 'dsigmoid':
        return DSigmoid_Prior(
            a=priorinfo[1][0],b=priorinfo[1][1],c=priorinfo[1][2],d=priorinfo[1][3],
            low=priorinfo[1][4],high=priorinfo[1][5])
    raise ValueError(
        f"log(R) prior {kind} is not supported when sampling log(R/dist), set "
        "indict['reparam'] = False to sample log(R) directly")

def softedge_logprob(prior, x, edge_width=0.01):
    """
    log_prob of prior at x, with the hard edges of a bounded support replaced
    by Gaussian fall-offs of width edge_width: inside the support it is the
    log_prob of prior, outside it is the log_prob at the nearest edge minus
    0.5 ((x - edge)/edge_width)^2. Continuous and finite everywhere.
    """
    support = prior.support
    lo = getattr(support,'lower_bound',None)
    hi = getattr(support,'upper_bound',None)
    if (lo is None) and (hi is None):
        return prior.log_prob(x)
    xc = jnp.clip(x,
                  lo if lo is not None else -jnp.inf,
                  hi if hi is not None else jnp.inf)
    return prior.log_prob(xc) - 0.5*((x-xc)/edge_width)**2

def sample_logR_reparam(priors, sample_i, edge_width=0.01):
    """
    Sample log(R) as log(R/dist) = log(R) - log10(dist), with dist already
    in sample_i.

    The photometry constrains R/dist and the parallax constrains dist, so
    (dist, log(R/dist)) are much less correlated than (dist, log(R)), which
    otherwise form a narrow ridge that variational guides cannot follow.
    The change of variables (dist, log(R)) -> (dist, log(R/dist)) has a
    Jacobian of 1, so the prior p(dist) p(log(R)) is kept by sampling
    log(R/dist) over a fixed wide range with a flat density and adding the
    log(R) prior at log(R) = log(R/dist) + log10(dist) as a factor. The
    support of log(R/dist) does not depend on dist, otherwise the transform
    to the unconstrained space would undo the reparameterization. Hard
    edges of the log(R) prior become Gaussian fall-offs of width edge_width
    (see softedge_logprob). log(R) is recorded as a deterministic.
    """
    prior = logR_prior_dist(priors.get('log(R)',None),sample_i.get('Teff',None))
    if prior is None:
        # fixed log(R), nothing to reparameterize
        if 'log(R)' in priors.keys():
            return determineprior('log(R)',priors['log(R)'],sample_i.get('Teff',None))
        return defaultprior('log(R)')
    logRdist = numpyro.sample('log(R/dist)',distfn.Uniform(*LOGRDIST_RANGE))
    logR = numpyro.deterministic('log(R)',logRdist + jnp.log10(sample_i['dist']))
    numpyro.factor('log(R)_prior',softedge_logprob(prior,logR,edge_width=edge_width))
    return logR

def logR_reparam_initpars(initpars, priors, verbose=True):
    """
    Initial value of log(R/dist) from the initial log(R) and dist, returns a
    copy of initpars. Without both of them, log(R/dist) starts at a random
    value.
    """
    initpars = dict(initpars)
    if logR_prior_dist(priors.get('log(R)',None),initpars.get('Teff',5000.0)) is None:
        return initpars
    if 'log(R/dist)' in initpars.keys():
        initpars.pop('log(R)',None)
        return initpars
    if ('log(R)' in initpars.keys()) and ('dist' in initpars.keys()):
        initpars['log(R/dist)'] = float(initpars.pop('log(R)')) - float(jnp.log10(initpars['dist']))
    elif verbose:
        print('[uberMS] WARNING: give initial values for both log(R) and dist, otherwise '
              'log(R/dist) starts at a random value')
    return initpars
