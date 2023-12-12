
from .advancedpriors import IMF_Prior,Gal_Prior

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


def determineprior(parname,priorinfo):
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

    if (priorinfo[0] is 'binchem'):
        # first option is both stars have same FeH and aFe
        if priorinfo[1][0] == 'normal':
            feh_p = numpyro.sample('[Fe/H]_p',distfn.Uniform(-4.0,0.5))
            feh_s = numpyro.sample('[Fe/H]_s',distfn.Normal(feh_p,priorinfo[1][1]))
            afe_p = numpyro.sample('[a/Fe]_p',distfn.Uniform(-0.2,0.6))
            afe_s = numpyro.sample('[a/Fe]_s',distfn.Normal(afe_p,priorinfo[1][1]))
            return (feh_p,feh_s,afe_p,afe_s)

        elif priorinfo[1][0] == 'uniform':
            feh_p = numpyro.sample('[Fe/H]_p',distfn.Uniform(-4.0,0.5))
            feh_s = numpyro.sample('[Fe/H]_s',distfn.Uniform(
                feh_p-priorinfo[1][1],feh_p+priorinfo[1][1]))
            afe_p = numpyro.sample('[a/Fe]_p',distfn.Uniform(-0.2,0.6))
            afe_s = numpyro.sample('[a/Fe]_s',distfn.Uniform(
                afe_p-priorinfo[1][1],afe_p+priorinfo[1][1]))
            return (feh_p,feh_s,afe_p,afe_s)

        else:
            feh_p = numpyro.sample('[Fe/H]_p',distfn.Uniform(-4.0,0.5))
            feh_s = numpyro.deterministic('[Fe/H]_s',feh_p)
            afe_p = numpyro.sample('[a/Fe]_p',distfn.Uniform(-0.2,0.6))
            afe_s = numpyro.deterministic('[a/Fe]_s',afe_p)
            return (feh_p,feh_s,afe_p,afe_s)
            

    # handle lsf properly
    if "lsf_array" in parname:
        specindex = parname.split('_')[-1]
        return jnp.asarray(priorinfo[0]) * numpyro.sample(
            "lsf_scaling_{}".format(specindex),distfn.Uniform(*priorinfo[1]))

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
    if priorinfo[0] == 'fixed':
        return numpyro.deterministic(parname,priorinfo[1])

