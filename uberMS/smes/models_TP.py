import numpyro
import numpyro.distributions as distfn
from numpyro.distributions import constraints
from numpyro.contrib.control_flow import cond

import jax.numpy as jnp

from .priors import determineprior, defaultprior

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

def planck(wav, T):
    wave_i = wav*(1E-10)
    a = 2.0*h*c**2
    b = h*c/(wave_i*k*T)
    intensity = a/ ( (wave_i**5) * (jnp.exp(b) - 1.0) )
    return intensity

# define the model
def model_specphot(
    indata={},
    fitfunc={},
    priors={},
    additionalinfo={},):

    # pull out the observed data
    specwave   = indata['specwave']
    specobs    = indata['specobs']
    specobserr = indata['specobserr']
    photobs    = indata['photobs']
    photobserr = indata['photobserr']
    filtarray  = indata['filterarray']

    # pull out fitting functions
    genphotfn = fitfunc['genphotfn']
    genspecfn = fitfunc['genspecfn']

    # pull out additional info
    parallax = additionalinfo.get('parallax',[None,None])
    vmicbool = additionalinfo['vmicbool']

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "specjitter",
        "photjitter",
        # "Teff_p",
        # "Teff_s",
        # "log(g)_p",
        # "log(g)_s",
        "vrad_p",
        "vrad_s",
        "vstar_p",
        "vstar_s",
        'log(R)_p',
        'log(R)_s',
        "dist",
        "Av",
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # define the primary as the hotter of the two stars
    sample_i['Teff_p'] = numpyro.sample("Teff_p",distfn.Uniform(2500.0, 10000.0))
    sample_i['Teff_s'] = numpyro.sample("Teff_s",distfn.Uniform(2500.0, sample_i['Teff_p']))
    # sample_i['Teff_s'] = numpyro.sample("Teff_s",distfn.Uniform(2500.0, 10000.0))

    sample_i['log(g)_p'] = numpyro.sample("log(g)_p",distfn.Uniform(0.0, 5.5))
    # sample_i['log(g)_s'] = numpyro.sample("log(g)_s",distfn.Uniform(sample_i['log(g)_p'],5.5))
    sample_i['log(g)_s'] = numpyro.sample("log(g)_s",distfn.Uniform(0.0,5.5))

    # require that |vrad_p - vrad_s| > 1.0
    # mixing_dist = distfn.Categorical(probs=jnp.ones(2) / 2.)
    # component_dists = ([
    #     distfn.Uniform(sample_i['vrad_p']-100.0,sample_i['vrad_p']-1.0,),
    #     distfn.Uniform(sample_i['vrad_p']+1.0,sample_i['vrad_p']+100.0,),
    #     ])
    # sample_i['vrad_s'] = numpyro.sample('vrad_s',distfn.MixtureGeneral(mixing_dist, component_dists))
    
    # figure out if user defines extra pc terms in priors
    # or should use the default pc0-pc3 terms
    pcln = len([kk for kk in priors.keys() if 'pc' in kk])
    if pcln == 0:
        pcterms = ['pc0','pc1','pc2','pc3']
    else:
        pcterms = ['pc{0}'.format(x) for x in range(pcln)]

    # now sample from priors for pc terms
    for pp in pcterms:
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # set vmic only if included in NNs
    if vmicbool:
        if 'vmic_p' in priors.keys():
            sample_i['vmic_p'] = determineprior('vmic_p',priors['vmic_p'])
        else:
            sample_i['vmic_p'] = defaultprior('vmic_p')
        if 'vmic_s' in priors.keys():
            sample_i['vmic_s'] = determineprior('vmic_s',priors['vmic_s'])
        else:
            sample_i['vmic_s'] = defaultprior('vmic_s')
    else:
        sample_i['vmic_p'] = 1.0
        sample_i['vmic_s'] = 1.0

    # handle various lsf cases
    if 'lsf_array' in priors.keys():
        # user has defined an lsf array, so set as free parameter 
        # a scaling on the lsf
        sample_i['lsf'] = determineprior('lsf_array',priors['lsf_array'])
    else:
        # user hasn't set a lsf array, treat lsf as R
        if 'lsf' in priors.keys():
            sample_i['lsf'] = determineprior('lsf',priors['lsf'])
        else:
            sample_i['lsf'] = defaultprior('lsf')

    # handle different cases for the treatment of [Fe/H] and [a/Fe]
    if 'binchem' in priors.keys():
        (sample_i["[Fe/H]_p"],sample_i["[Fe/H]_s"],sample_i["[a/Fe]_p"],sample_i["[a/Fe]_s"]) = determineprior(None,priors['binchem'])
    else:
        for pp in ["[Fe/H]_p","[Fe/H]_s","[a/Fe]_p","[a/Fe]_s"]:  
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
            else:
                sample_i[pp] = defaultprior(pp)

    # sample in a jitter term for error in spectrum
    specsig = jnp.sqrt( (specobserr**2.0) + (sample_i['specjitter']**2.0) )

    # make the spectral prediciton
    specpars_p = ([
        sample_i['Teff_p'],sample_i['log(g)_p'],sample_i['[Fe/H]_p'],sample_i['[a/Fe]_p'],
        sample_i['vrad_p'],sample_i['vstar_p'],sample_i['vmic_p'],sample_i['lsf']])
    specpars_p += [sample_i['pc{0}'.format(x)] for x in range(len(pcterms))]
    specmod_p = genspecfn(specpars_p,outwave=specwave,modpoly=True)
    specmod_p = jnp.asarray(specmod_p[1])

    specpars_s = ([
        sample_i['Teff_s'],sample_i['log(g)_s'],sample_i['[Fe/H]_s'],sample_i['[a/Fe]_s'],
        sample_i['vrad_s'],sample_i['vstar_s'],sample_i['vmic_s'],sample_i['lsf']])
    specpars_s += [1.0,0.0]
    specmod_s = genspecfn(specpars_s,outwave=specwave,modpoly=True)
    specmod_s = jnp.asarray(specmod_s[1])

    radius_p = 10.0**sample_i['log(R)_p']
    radius_s = 10.0**sample_i['log(R)_s']

    R = (
        (planck(specwave,sample_i['Teff_p']) * radius_p**2.0) / 
        (planck(specwave,sample_i['Teff_s']) * radius_s**2.0)
         )
    specmod_est = (specmod_p + R * specmod_s) / (1.0 + R)

    # calculate likelihood for spectrum
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

    # make photometry prediction
    photpars_p = ([
        sample_i['Teff_p'],sample_i['log(g)_p'],sample_i['[Fe/H]_p'],sample_i['[a/Fe]_p'],
        sample_i['log(R)_p'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_p = genphotfn(photpars_p)
    photmod_p = [photmod_p[xx] for xx in filtarray]

    photpars_s = ([
        sample_i['Teff_s'],sample_i['log(g)_s'],sample_i['[Fe/H]_s'],sample_i['[a/Fe]_s'],
        sample_i['log(R)_s'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_s = genphotfn(photpars_s)
    photmod_s = [photmod_s[xx] for xx in filtarray]

    photmod_est = (
        [-2.5 * jnp.log10( 10.0**(-0.4 * m_p) + 10.0**(-0.4 * m_s) )
         for m_p,m_s in zip(photmod_p,photmod_s)
         ] 
    )
    photmod_est = jnp.asarray(photmod_est)

    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    
    # calcluate likelihood of parallax
    numpyro.sample("para", distfn.Normal(1000.0/sample_i['dist'],parallax[1]), obs=parallax[0])


# define the model
def model_spec(
    indata={},
    fitfunc={},
    priors={},
    additionalinfo={},):

    # pull out the observed data
    specwave   = indata['specwave']
    specobs    = indata['specobs']
    specobserr = indata['specobserr']

    # pull out fitting functions
    genspecfn = fitfunc['genspecfn']

    # pull out additional info
    vmicbool = additionalinfo['vmicbool']

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "specjitter",
        # "Teff_p",
        # "Teff_s",
        # "log(g)_p",
        # "log(g)_s",
        "vrad_p",
        # "vrad_s",
        "vstar_p",
        "vstar_s",
        "log(R)_p",
        "log(R)_s",        
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # define the primary as the hotter of the two stars
    sample_i['Teff_p'] = numpyro.sample("Teff_p",distfn.Uniform(2500.0, 10000.0))
    # sample_i['Teff_s'] = numpyro.sample("Teff_s",distfn.Uniform(2500.0, sample_i['Teff_p']))
    sample_i['Teff_s'] = numpyro.sample("Teff_s",distfn.Uniform(2500.0, 10000.0))

    sample_i['log(g)_p'] = numpyro.sample("log(g)_p",distfn.Uniform(0.0, 5.5))
    # sample_i['log(g)_s'] = numpyro.sample("log(g)_s",distfn.Uniform(sample_i['log(g)_p'],5.5))
    sample_i['log(g)_s'] = numpyro.sample("log(g)_s",distfn.Uniform(0.0,5.5))

    # require that |vrad_p - vrad_s| > 1.0
    mixing_dist = distfn.Categorical(probs=jnp.ones(2) / 2.)
    component_dists = ([
        distfn.Uniform(sample_i['vrad_p']-100.0,sample_i['vrad_p']-1.0,),
        distfn.Uniform(sample_i['vrad_p']+1.0,sample_i['vrad_p']+100.0,),
        ])
    sample_i['vrad_s'] = numpyro.sample('vrad_s',distfn.MixtureGeneral(mixing_dist, component_dists))
    
    # figure out if user defines extra pc terms in priors
    # or should use the default pc0-pc3 terms
    pcln = len([kk for kk in priors.keys() if 'pc' in kk])
    if pcln == 0:
        pcterms = ['pc0','pc1','pc2','pc3']
    else:
        pcterms = ['pc{0}'.format(x) for x in range(pcln)]

    # now sample from priors for pc terms
    for pp in pcterms:
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # set vmic only if included in NNs
    if vmicbool:
        if 'vmic_p' in priors.keys():
            sample_i['vmic_p'] = determineprior('vmic_p',priors['vmic_p'])
        else:
            sample_i['vmic_p'] = defaultprior('vmic_p')
        if 'vmic_s' in priors.keys():
            sample_i['vmic_s'] = determineprior('vmic_s',priors['vmic_s'])
        else:
            sample_i['vmic_s'] = defaultprior('vmic_s')
    else:
        sample_i['vmic_p'] = 1.0
        sample_i['vmic_s'] = 1.0

    # handle various lsf cases
    if 'lsf_array' in priors.keys():
        # user has defined an lsf array, so set as free parameter 
        # a scaling on the lsf
        sample_i['lsf'] = determineprior('lsf_array',priors['lsf_array'])
    else:
        # user hasn't set a lsf array, treat lsf as R
        if 'lsf' in priors.keys():
            sample_i['lsf'] = determineprior('lsf',priors['lsf'])
        else:
            sample_i['lsf'] = defaultprior('lsf')

    # handle different cases for the treatment of [Fe/H] and [a/Fe]
    if 'binchem' in priors.keys():
        (sample_i["[Fe/H]_p"],sample_i["[Fe/H]_s"],sample_i["[a/Fe]_p"],sample_i["[a/Fe]_s"]) = determineprior(None,priors['binchem'])
    else:
        for pp in ["[Fe/H]_p","[Fe/H]_s","[a/Fe]_p","[a/Fe]_s"]:  
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
            else:
                sample_i[pp] = defaultprior(pp)

    # sample in a jitter term for error in spectrum
    specsig = jnp.sqrt( (specobserr**2.0) + (sample_i['specjitter']**2.0) )

    # make the spectral prediciton
    specpars_p = ([
        sample_i['Teff_p'],sample_i['log(g)_p'],sample_i['[Fe/H]_p'],sample_i['[a/Fe]_p'],
        sample_i['vrad_p'],sample_i['vstar_p'],sample_i['vmic_p'],sample_i['lsf']])
    specpars_p += [sample_i['pc{0}'.format(x)] for x in range(len(pcterms))]
    specmod_p = genspecfn(specpars_p,outwave=specwave,modpoly=True)
    specmod_p = jnp.asarray(specmod_p[1])

    specpars_s = ([
        sample_i['Teff_s'],sample_i['log(g)_s'],sample_i['[Fe/H]_s'],sample_i['[a/Fe]_s'],
        sample_i['vrad_s'],sample_i['vstar_s'],sample_i['vmic_s'],sample_i['lsf']])
    specpars_s += [1.0,0.0]
    specmod_s = genspecfn(specpars_s,outwave=specwave,modpoly=True)
    specmod_s = jnp.asarray(specmod_s[1])

    radius_p = 10.0**sample_i['log(R)_p']
    radius_s = 10.0**sample_i['log(R)_s']

    R = (
        (planck(specwave,sample_i['Teff_p']) * radius_p**2.0) / 
        (planck(specwave,sample_i['Teff_s']) * radius_s**2.0)
         )
    specmod_est = (specmod_p + R * specmod_s) / (1.0 + R)

    # calculate likelihood for spectrum
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)

# define the model
def model_phot(
    indata={},
    fitfunc={},
    priors={},
    additionalinfo={},):

    # pull out the observed data
    photobs    = indata['photobs']
    photobserr = indata['photobserr']
    filtarray  = indata['filterarray']

    # pull out fitting functions
    genphotfn = fitfunc['genphotfn']

    # pull out additional info
    parallax = additionalinfo.get('parallax',[None,None])

    # define sampled parameters apply the user defined priors
    sampledpars = ([
        "photjitter",
        "Teffp",
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        'log(R)',
        "dist",
        "Av",
        "ff"
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    teffs_p = -3.58 * (10**-5) * (sample_i['Teffp']**2.0) + 0.751 * sample_i['Teffp'] + 808.0
    sample_i['Teffs'] = numpyro.sample(
        "Teffs",
        distfn.TruncatedDistribution(distfn.Normal(loc=teffs_p,scale=250.0),
                                     low=teffs_p-500.0,high=teffs_p+500.0))

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

    # make photometry prediction
    photpars_p = ([
        sample_i['Teffp'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_p = genphotfn(photpars_p)
    photmod_p = [photmod_p[xx] for xx in filtarray]

    photpars_s = ([
        sample_i['Teffs'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_s = genphotfn(photpars_s)
    photmod_s = [photmod_s[xx] for xx in filtarray]

    photmod_est = (
        [-2.5 * jnp.log10( (1.0-sample_i['ff']) * 10.0**(-0.4 * m_p) + sample_i['ff'] * 10.0**(-0.4 * m_s) )
         for m_p,m_s in zip(photmod_p,photmod_s)
         ] 
    )
    photmod_est = jnp.asarray(photmod_est)

    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    
    # calcluate likelihood of parallax
    numpyro.sample("para", distfn.Normal(1000.0/sample_i['dist'],parallax[1]), obs=parallax[0])
