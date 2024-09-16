from .advancedpriors import IMF_Prior,Gal_Prior,Sigmoid_Prior,DSigmoid_Prior

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
        return numpyro.sample("specjitter", distfn.HalfNormal(0.001))
    if parname == "photjitter":
        return numpyro.sample("photjitter", distfn.HalfNormal(0.001))


def determineprior(parname,priorinfo,*args):
    # advanced priors
    if (priorinfo[0] is 'IMF'):
        mass_le,mass_ue = priorinfo[1]['mass_le'],priorinfo[1]['mass_ue']
        return numpyro.sample("initial_Mass",IMF_Prior(low=mass_le,high=mass_ue))

    if (priorinfo[0] is 'GAL'):
        dist_le,dist_ue = priorinfo[1]['dist_ll'],priorinfo[1]['dist_ul']
        GP = Gal_Prior(l=priorinfo[1]['l'],b=priorinfo[1]['b'],low=dist_le,high=dist_ue)
        return numpyro.sample("dist",GP)

    if (priorinfo[0] is 'GALAGE'):
        GP = Gal_Prior(l=priorinfo[1]['l'],b=priorinfo[1]['b'])
        return numpyro.sample("dist",GP)

    if parname is 'vmic':
        # check to see if user wants to use relationship for vmic
        if (priorinfo[0] is 'Bruntt2012'):
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
            "lsf_scaling",distfn.Uniform(*priorinfo[1]))

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
        pjpriorkeys.remove('global')
        
        # now set prior on filters or systems
        for kk in pjpriorkeys:
            outdict[f'photjitter_{kk}'] = determineprior(f'photjitter_{kk}',pjprior[kk])

        # return the draws
        return outdict
