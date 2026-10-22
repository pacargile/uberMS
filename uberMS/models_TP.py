import numpyro
import numpyro.distributions as distfn
from numpyro.distributions import constraints
from numpyro.contrib.control_flow import cond

import jax.numpy as jnp
import jax

from .priors import determineprior, defaultprior
from .fitutils import predict_spec, pcterms_from_priors

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
    parallax = additionalinfo.get('parallax', None)
    vmicbool = additionalinfo['vmicbool']
    specmode = additionalinfo.get('specmode','normalized')
    specavrv = additionalinfo.get('specavrv',False)
    fluxscale = additionalinfo.get('fluxscale',1.0)

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "specjitter",
        "photjitter",
        "Teff",
        'log(R)',
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        "vrad",
        "vstar",
        "dist",
        "Av",
        "Rv",
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            if (pp == 'log(g)') and (priors[pp][0] == 'IMF'):
                # IMF prior on log(g) is conditioned on log(R)
                sample_i[pp] = determineprior(pp,priors[pp],sample_i['log(R)'])
            else:
                sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # figure out which pc terms the user defines in priors, defaults
    # to pc0-pc3 in normalized mode and no polynomial in fluxed mode
    pcterms = pcterms_from_priors(priors,specmode)

    # now sample from priors for pc terms
    for pp in pcterms:
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        elif (pp == 'pc0') and (specmode == 'fluxed'):
            # flux scale is set by (R/dist)^2, so fix pc0 unless user sets a prior
            sample_i[pp] = numpyro.deterministic(pp,1.0)
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

    # make the spectral prediction, in fluxed mode the spectrum shares
    # Av, Rv, log(R), and dist with the photometry
    specmod_est = predict_spec(
        genspecfn,specwave,
        [sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]']],
        sample_i['vrad'],sample_i['vstar'],sample_i['vmic'],sample_i['lsf'],
        [sample_i[pp] for pp in pcterms],
        specmode=specmode,avrv=specavrv,
        Av=sample_i['Av'],Rv=sample_i['Rv'],logR=sample_i['log(R)'],dist=sample_i['dist'],
        fluxscale=fluxscale)

    # calculate likelihood for spectrum
    # sample in a jitter term for error in spectrum, in fluxed mode the
    # jitter is a fraction of the model flux
    if specmode == 'fluxed':
        specsig = jnp.sqrt( (specobserr**2.0) + ((sample_i['specjitter']*specmod_est)**2.0) )
    else:
        specsig = jnp.sqrt( (specobserr**2.0) + (sample_i['specjitter']**2.0) )
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

    # make photometry prediction
    photpars = ([
        sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],sample_i['Rv']])
    photmod_est = genphotfn(photpars)
    photmod_est = jnp.asarray([photmod_est[xx] for xx in filtarray])
    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    
    # calcluate likelihood of parallax
    if parallax is not None:
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
    parallax = additionalinfo.get('parallax', None)
    vmicbool = additionalinfo['vmicbool']
    specmode = additionalinfo.get('specmode','normalized')
    specavrv = additionalinfo.get('specavrv',False)
    fluxscale = additionalinfo.get('fluxscale',1.0)

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

    # a fluxed spectrum also constrains radius, distance, and extinction,
    # log(R) is sampled before log(g) for the IMF log(g) prior
    if specmode == 'fluxed':
        sampledpars.insert(sampledpars.index("log(g)"),"log(R)")
        sampledpars += ["dist","Av","Rv"]

    sample_i = {}
    for pp in sampledpars:
        if pp in priors.keys():
            if (pp == 'log(g)') and (priors[pp][0] == 'IMF') and ('log(R)' in sample_i.keys()):
                # IMF prior on log(g) is conditioned on log(R)
                sample_i[pp] = determineprior(pp,priors[pp],sample_i['log(R)'])
            else:
                sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # figure out which pc terms the user defines in priors, defaults
    # to pc0-pc3 in normalized mode and no polynomial in fluxed mode
    pcterms = pcterms_from_priors(priors,specmode)

    # now sample from priors for pc terms
    for pp in pcterms:
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        elif (pp == 'pc0') and (specmode == 'fluxed'):
            # flux scale is set by (R/dist)^2, so fix pc0 unless user sets a prior
            sample_i[pp] = numpyro.deterministic(pp,1.0)
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

    # make the spectral prediciton
    if specmode == 'fluxed':
        fluxpars = {kk:sample_i[kk] for kk in ['Av','Rv','dist']}
        fluxpars['logR'] = sample_i['log(R)']
    else:
        fluxpars = {}
    specmod_est = predict_spec(
        genspecfn,specwave,
        [sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]']],
        sample_i['vrad'],sample_i['vstar'],sample_i['vmic'],sample_i['lsf'],
        [sample_i[pp] for pp in pcterms],
        specmode=specmode,avrv=specavrv,fluxscale=fluxscale,**fluxpars)
    # calculate likelihood for spectrum
    # sample in a jitter term for error in spectrum, in fluxed mode the
    # jitter is a fraction of the model flux
    if specmode == 'fluxed':
        specsig = jnp.sqrt( (specobserr**2.0) + ((sample_i['specjitter']*specmod_est)**2.0) )
    else:
        specsig = jnp.sqrt( (specobserr**2.0) + (sample_i['specjitter']**2.0) )
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)

    # calcluate likelihood of parallax, only informative for a fluxed spectrum
    if (specmode == 'fluxed') and (parallax is not None):
        numpyro.sample("para", distfn.Normal(1000.0/sample_i['dist'],parallax[1]), obs=parallax[0])

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
    parallax = additionalinfo.get('parallax', None)

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "photjitter",
        "Teff",
        'log(R)',
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        "dist",
        "Av",
        "Rv",
        ])

    sample_i = {}
    for pp in sampledpars:            
        if pp in priors.keys():
            if (pp == 'log(g)') and (priors[pp][0] == 'IMF'):
                # IMF prior on log(g) is conditioned on log(R)
                sample_i[pp] = determineprior(pp,priors[pp],sample_i['log(R)'])
            else:
                sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)
    
    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )
    
    # make photometry prediction
    photpars = ([
        sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],sample_i['Rv']])
    photmod_est = genphotfn(photpars)
    photmod_est = jnp.asarray([photmod_est[xx] for xx in filtarray])

    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)

    # calcluate likelihood of parallax
    if parallax is not None:
        numpyro.sample("para", distfn.Normal(1000.0/sample_i['dist'],parallax[1]), obs=parallax[0])
