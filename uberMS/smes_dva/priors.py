
from .advancedpriors import IMF_Prior,Gal_Prior,Sigmoid_Prior,DSigmoid_Prior

import numpyro
import numpyro.distributions as distfn
import jax.numpy as jnp

def defaultprior(parname):
    # define defaults for sampled parameters
    if "EEP" in parname:
        return numpyro.sample(parname, distfn.Uniform(300,800))
    if "initial_Mass" in parname:
        return numpyro.sample(parname, IMF_Prior())
    if "initial_[Fe/H]" in parname:
        return numpyro.sample(parname, distfn.Uniform(-3.5,0.49))
    if "initial_[a/Fe]" in parname:
        return numpyro.sample(parname, distfn.Uniform(-0.19,0.59))
    if "vmic" in parname:
        return numpyro.sample(parname, distfn.Uniform(0.5, 3.0))
    if "vstar" in parname:
        return numpyro.sample(parname, distfn.Uniform(0.0, 25.0))
    if "Teff" in parname:
        return numpyro.sample(parname, distfn.Uniform(2500.0, 10000.0))
    if "log(g)" in parname:
        return numpyro.sample(parname, distfn.Uniform(0.0, 5.5))
    if "[Fe/H]" in parname:
        return numpyro.sample(parname, distfn.Uniform(-3.5,0.49))
    if "[a/Fe]" in parname:
        return numpyro.sample(parname, distfn.Uniform(-0.19,0.59))        
    if "vrad" in parname:
        return numpyro.sample(parname, distfn.Uniform(-500.0, 500.0))
    if "pc0" in parname:
        return numpyro.sample(parname, distfn.Uniform(0.5, 2.0))
    if "pc1" in parname:
        return numpyro.sample(parname, distfn.Normal(0.0, 0.25))
    if "pc2" in parname:
        return numpyro.sample(parname, distfn.Normal(0.0, 0.25))
    if "pc3" in parname:
        return numpyro.sample(parname, distfn.Normal(0.0, 0.25))
    if "lsf" in parname:
        return numpyro.sample(parname, distfn.Normal(32000.0,1000.0))
    if "specjitter" in parname:
        return numpyro.sample(parname, distfn.HalfNormal(0.001))

    if "log(R)" in parname:
        return numpyro.sample(parname, distfn.Uniform(-2,3.0))  

    if parname == "Av":
        return numpyro.sample("Av", distfn.Uniform(1E-6,5.0))
    if parname == 'dist':
        return numpyro.sample("dist", distfn.Uniform(1,200000.0))  
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

    if parname == 'MassRatio':
        # pull out mass ratio and systemic velocity priors
        mr_ll,mr_ul = priorinfo[1]['mass_ratio'][0],priorinfo[1]['mass_ratio'][1]
        sysvel_mean,sysvel_sigma = priorinfo[1]['sysvel'][0],priorinfo[1]['sysvel'][1]
        mr = numpyro.sample("mass_ratio", distfn.Uniform(mr_ll,mr_ul))
        sysvel = numpyro.sample("sysvel", distfn.Normal(sysvel_mean,sysvel_sigma))
        nspec = args[0]
        vrad_a_arr = []
        vrad_b_arr = []
        for ii in range(nspec):
            vrad_a = numpyro.sample(f"vrad_a_{ii}", distfn.Uniform(-200 + sysvel, 200 + sysvel))
            rvb = sysvel - (vrad_a - sysvel) / mr
            vrad_b = numpyro.deterministic(f"vrad_b_{ii}", rvb)
            vrad_a_arr.append(vrad_a)
            vrad_b_arr.append(vrad_b)

        return (vrad_a_arr,vrad_b_arr)

    if 'vmic' in parname:
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

    if 'log(R)' in parname:
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

    if 'log(g)' in parname:
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
    if "lsf_array" in parname:
        return jnp.asarray(priorinfo[0]) * numpyro.sample(
            "lsf_scaling",distfn.Uniform(*priorinfo[1]))


    if (priorinfo[0] == 'binchem'):
        # first option is both stars have same FeH and aFe
        if priorinfo[1][0] == 'normal':
            feh_a = numpyro.sample('[Fe/H]_a',distfn.Uniform(-4.0,0.5))
            feh_b = numpyro.sample('[Fe/H]_b',distfn.Normal(feh_p,priorinfo[1][1]))
            afe_a = numpyro.sample('[a/Fe]_a',distfn.Uniform(-0.2,0.6))
            afe_b = numpyro.sample('[a/Fe]_b',distfn.Normal(afe_p,priorinfo[1][1]))
            return (feh_a,feh_b,afe_a,afe_b)

        elif priorinfo[1][0] == 'uniform':
            feh_a = numpyro.sample('[Fe/H]_a',distfn.Uniform(-4.0,0.5))
            feh_b = numpyro.sample('[Fe/H]_b',distfn.Uniform(
                feh_a-priorinfo[1][1],feh_a+priorinfo[1][1]))
            afe_a = numpyro.sample('[a/Fe]_a',distfn.Uniform(-0.2,0.6))
            afe_b = numpyro.sample('[a/Fe]_b',distfn.Uniform(
                afe_a-priorinfo[1][1],afe_a+priorinfo[1][1]))
            return (feh_a,feh_b,afe_a,afe_b)

        else:
            feh_a = numpyro.sample('[Fe/H]_a',distfn.Uniform(-4.0,0.5))
            feh_b = numpyro.deterministic('[Fe/H]_b',feh_a)
            afe_a = numpyro.sample('[a/Fe]_a',distfn.Uniform(-0.2,0.6))
            afe_b = numpyro.deterministic('[a/Fe]_b',afe_a)
            return (feh_a,feh_b,afe_a,afe_b)
        
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
