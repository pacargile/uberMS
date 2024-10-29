import numpyro
import numpyro.distributions as distfn
from numpyro.distributions import constraints
from numpyro.contrib.control_flow import cond

import jax.numpy as jnp

from .priors import determineprior, defaultprior

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
        "Teff",
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        "vrad",
        "vstar",
        'log(R)',
        "dist",
        "Av",
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

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
        if 'vmic' in priors.keys():
            sample_i['vmic'] = determineprior('vmic',priors['vmic'],sample_i['Teff'],sample_i['log(g)'])
        else:
            sample_i['vmic'] = defaultprior('vmic')
    else:
        sample_i['vmic'] = 1.0

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

    # sample in a jitter term for error in spectrum
    specsig = jnp.sqrt( (specobserr**2.0) + (sample_i['specjitter']**2.0) )

    # make the spectral prediciton
    specpars = ([
        sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['vrad'],sample_i['vstar'],sample_i['vmic'],sample_i['lsf']])
    specpars += [sample_i['pc{0}'.format(x)] for x in range(len(pcterms))]
    specmod_est = genspecfn(specpars,outwave=specwave,modpoly=True)
    specmod_est = jnp.asarray(specmod_est[1])
    # calculate likelihood for spectrum
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

    # make photometry prediction
    photpars = ([
        sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_est = genphotfn(photpars)
    photmod_est = jnp.asarray([photmod_est[xx] for xx in filtarray])
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
        "Teff",
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        "vrad",
        "vstar",
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

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
        if 'vmic' in priors.keys():
            sample_i['vmic'] = determineprior('vmic',priors['vmic'],sample_i['Teff'],sample_i['log(g)'])
        else:
            sample_i['vmic'] = defaultprior('vmic')
    else:
        sample_i['vmic'] = 1.0

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

    # sample in a jitter term for error in spectrum
    specsig = jnp.sqrt( (specobserr**2.0) + (sample_i['specjitter']**2.0) )

    # make the spectral prediciton
    specpars = ([
        sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['vrad'],sample_i['vstar'],sample_i['vmic'],sample_i['lsf']])
    specpars += [sample_i['pc{0}'.format(x)] for x in range(len(pcterms))]
    specmod_est = genspecfn(specpars,outwave=specwave,modpoly=True)
    specmod_est = jnp.asarray(specmod_est[1])
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
        "Teff",
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        'log(R)',
        "dist",
        "Av",
        ])

    sample_i = {}
    for pp in sampledpars:            
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)
    
    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )
    
    # make photometry prediction
    photpars = ([
        sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_est = genphotfn(photpars)
    photmod_est = jnp.asarray([photmod_est[xx] for xx in filtarray])

    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)

    # calcluate likelihood of parallax
    numpyro.sample("para", distfn.Normal(1000.0/sample_i['dist'],parallax[1]), obs=parallax[0])
