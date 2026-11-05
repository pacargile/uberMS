import numpyro
import numpyro.distributions as distfn
from numpyro.distributions import constraints
from numpyro.contrib.control_flow import cond

import jax.numpy as jnp

from .priors import determineprior, defaultprior
from ..fitutils import predict_spec, pcterms_from_priors

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

    # determine how many spectra to fit based on len of specwave
    nspec = len(specwave)

    # spectral fitting mode, F_lambda NN flag, and flux units for each spectrum
    specmode = additionalinfo.get('specmode',['normalized']*nspec)
    specavrv = additionalinfo.get('specavrv',[False]*nspec)
    fluxscale = additionalinfo.get('fluxscale',[1.0]*nspec)
    anyfluxed = ('fluxed' in specmode)
    
    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "photjitter",
        "Teff",
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        "vstar",
        'log(R)',
        "dist",
        "Av",
        "Rv",
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # for every input spectrum, figure out which pc terms the user defines in priors,
    # defaults to pc0-pc3 in normalized mode and no polynomial in fluxed mode
    pcterms = []
    for ii in range(nspec):
        pcterms_i = pcterms_from_priors(priors,specmode[ii],suffix='_{}'.format(ii))
        pcterms.append(pcterms_i)

        # now sample from priors for pc terms
        for pp in pcterms_i:
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
            elif (pp == 'pc0_{}'.format(ii)) and (specmode[ii] == 'fluxed'):
                # flux scale is set by (R/dist)^2, so fix pc0 unless user sets a prior
                sample_i[pp] = numpyro.deterministic(pp,1.0)
            else:
                sample_i[pp] = defaultprior(pp)

        # handle various lsf cases
        if 'lsf_array_{}'.format(ii) in priors.keys():
            # user has defined an lsf array, so set as free parameter 
            # a scaling on the lsf
            sample_i['lsf_{}'.format(ii)] = determineprior(
                'lsf_array_{}'.format(ii),
                priors['lsf_array_{}'.format(ii)])
        else:
            # user hasn't set a lsf array, treat lsf as R
            if 'lsf_{}'.format(ii) in priors.keys():
                sample_i['lsf_{}'.format(ii)] = determineprior(
                    'lsf_{}'.format(ii),
                    priors['lsf_{}'.format(ii)])
            else:
                sample_i['lsf_{}'.format(ii)] = defaultprior('lsf_{}'.format(ii))

        pp = 'specjitter_{}'.format(ii)
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    if 'radvel' in priors.keys():
        if priors['radvel'][0] == 'locked':
            vrad = numpyro.sample('vrad',distfn.Uniform(*priors['radvel'][1]))
            for ii in range(nspec):
                pp = 'vrad_{}'.format(ii)
                sample_i[pp] = numpyro.deterministic(pp,vrad)
        elif priors['radvel'][0] == 'normal':
            vrad = numpyro.sample('vrad',distfn.Uniform(priors['radvel'][1][1],priors['radvel'][1][2]))
            for ii in range(nspec):
                pp = 'vrad_{}'.format(ii)
                sample_i[pp] = numpyro.sample(pp,distfn.TruncatedDistribution(
                    distfn.Normal(loc=vrad,scale=priors['radvel'][1][0]),
                    low=priors['radvel'][1][1],high=priors['radvel'][1][2]))
        elif priors['radvel'][0] == 'uniform':
            vrad = numpyro.sample('vrad',distfn.Uniform(priors['radvel'][1][1],priors['radvel'][1][2]))
            for ii in range(nspec):
                pp = 'vrad_{}'.format(ii)
                sample_i[pp] = numpyro.sample(pp,distfn.Uniform(priors['radvel'][1][1],priors['radvel'][1][2]))
    else:
        for ii in range(nspec):
            pp = 'vrad_{}'.format(ii)
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
        sample_i['vmic'] = numpyro.deterministic('vmic',1.0)

    # fluxed spectra share the radius, distance, and extinction
    if anyfluxed:
        fluxpars = {kk:sample_i[kk] for kk in ['Av','Rv','dist']}
        fluxpars['logR'] = sample_i['log(R)']

    for ii in range(nspec):
        # make the spectral prediciton
        specmod_est = predict_spec(
            genspecfn[ii],specwave[ii],
            [sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]']],
            sample_i['vrad_{}'.format(ii)],sample_i['vstar'],sample_i['vmic'],
            sample_i['lsf_{}'.format(ii)],
            [sample_i[pp] for pp in pcterms[ii]],
            specmode=specmode[ii],avrv=specavrv[ii],fluxscale=fluxscale[ii],
            **(fluxpars if specmode[ii] == 'fluxed' else {}))

        # sample in a jitter term for error in spectrum, in fluxed mode the
        # jitter is a fraction of the model flux
        if specmode[ii] == 'fluxed':
            specsig = jnp.sqrt( (specobserr[ii]**2.0) + ((sample_i['specjitter_{}'.format(ii)]*specmod_est)**2.0) )
        else:
            specsig = jnp.sqrt( (specobserr[ii]**2.0) + (sample_i['specjitter_{}'.format(ii)]**2.0) )

        # calculate likelihood for spectrum
        numpyro.sample("specobs_{}".format(ii),distfn.Normal(specmod_est, specsig), obs=specobs[ii])

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

    # make photometry prediction
    photpars = ([
        sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],sample_i['Rv']])
    photmod = genphotfn(photpars)
    photmod = [photmod[xx] for xx in filtarray]
    photmod_est = jnp.asarray(photmod)

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

    # determine how many spectra to fit based on len of specwave
    nspec = len(specwave)

    # spectral fitting mode, F_lambda NN flag, and flux units for each spectrum
    specmode = additionalinfo.get('specmode',['normalized']*nspec)
    specavrv = additionalinfo.get('specavrv',[False]*nspec)
    fluxscale = additionalinfo.get('fluxscale',[1.0]*nspec)
    anyfluxed = ('fluxed' in specmode)
    
    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "Teff",
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        "vstar",
        ])

    # fluxed spectra also constrain radius, distance, and extinction
    if anyfluxed:
        sampledpars += ['log(R)','dist','Av','Rv']

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # for every input spectrum, figure out which pc terms the user defines in priors,
    # defaults to pc0-pc3 in normalized mode and no polynomial in fluxed mode
    pcterms = []
    for ii in range(nspec):
        pcterms_i = pcterms_from_priors(priors,specmode[ii],suffix='_{}'.format(ii))
        pcterms.append(pcterms_i)

        # now sample from priors for pc terms
        for pp in pcterms_i:
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
            elif (pp == 'pc0_{}'.format(ii)) and (specmode[ii] == 'fluxed'):
                # flux scale is set by (R/dist)^2, so fix pc0 unless user sets a prior
                sample_i[pp] = numpyro.deterministic(pp,1.0)
            else:
                sample_i[pp] = defaultprior(pp)

        # handle various lsf cases
        if 'lsf_array_{}'.format(ii) in priors.keys():
            # user has defined an lsf array, so set as free parameter 
            # a scaling on the lsf
            sample_i['lsf_{}'.format(ii)] = determineprior(
                'lsf_array_{}'.format(ii),
                priors['lsf_array_{}'.format(ii)])
        else:
            # user hasn't set a lsf array, treat lsf as R
            if 'lsf_{}'.format(ii) in priors.keys():
                sample_i['lsf_{}'.format(ii)] = determineprior(
                    'lsf_{}'.format(ii),
                    priors['lsf_{}'.format(ii)])
            else:
                sample_i['lsf_{}'.format(ii)] = defaultprior('lsf_{}'.format(ii))

        pp = 'vrad_{}'.format(ii)
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

        pp = 'specjitter_{}'.format(ii)
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
        sample_i['vmic'] = numpyro.deterministic('vmic',1.0)

    # fluxed spectra share the radius, distance, and extinction
    if anyfluxed:
        fluxpars = {kk:sample_i[kk] for kk in ['Av','Rv','dist']}
        fluxpars['logR'] = sample_i['log(R)']

    for ii in range(nspec):
        # make the spectral prediciton
        specmod_est = predict_spec(
            genspecfn[ii],specwave[ii],
            [sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]']],
            sample_i['vrad_{}'.format(ii)],sample_i['vstar'],sample_i['vmic'],
            sample_i['lsf_{}'.format(ii)],
            [sample_i[pp] for pp in pcterms[ii]],
            specmode=specmode[ii],avrv=specavrv[ii],fluxscale=fluxscale[ii],
            **(fluxpars if specmode[ii] == 'fluxed' else {}))

        # sample in a jitter term for error in spectrum, in fluxed mode the
        # jitter is a fraction of the model flux
        if specmode[ii] == 'fluxed':
            specsig = jnp.sqrt( (specobserr[ii]**2.0) + ((sample_i['specjitter_{}'.format(ii)]*specmod_est)**2.0) )
        else:
            specsig = jnp.sqrt( (specobserr[ii]**2.0) + (sample_i['specjitter_{}'.format(ii)]**2.0) )

        # calculate likelihood for spectrum
        numpyro.sample("specobs_{}".format(ii),distfn.Normal(specmod_est, specsig), obs=specobs[ii])

    # calcluate likelihood of parallax, only informative for fluxed spectra
    if anyfluxed and (parallax is not None):
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
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        'log(R)',
        "dist",
        "Av",
        "Rv",
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
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],sample_i['Rv']])
    photmod_est = genphotfn(photpars)
    photmod_est = jnp.asarray([photmod_est[xx] for xx in filtarray])
    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    
    # calcluate likelihood of parallax
    if parallax is not None:
        numpyro.sample("para", distfn.Normal(1000.0/sample_i['dist'],parallax[1]), obs=parallax[0])

