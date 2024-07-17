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

    # determine how many spectra to fit based on len of specwave
    nspec = len(specwave)
    
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
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # for every input spectrum, figure out if user defines extra pc terms in priors
    # or should use the default pc0-pc3 terms
    pcln = []
    pcterms = []
    for ii in range(nspec):
        # determine how many user defined pc priors there are for spec == ii
        pcln_i = len([kk for kk in priors.keys() if all(aa in kk for aa in ['pc','_{}'.format(ii)])])
        pcln.append(pcln_i)
        if pcln_i == 0:
            pcterms_i = ['pc0_{}'.format(ii),'pc1_{}'.format(ii),
                        'pc2_{}'.format(ii),'pc3_{}'.format(ii)]
        else:
            pcterms_i = ['pc{0}_{1}'.format(x,ii) for x in range(pcln_i)]

        pcterms.append(pcterms_i)

        # now sample from priors for pc terms
        for pp in pcterms_i:
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
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

    # sample in a jitter term for error in spectrum
    for ii in range(nspec):
        specsig = jnp.sqrt( (specobserr[ii]**2.0) + (sample_i['specjitter_{}'.format(ii)]**2.0) )

        # make the spectral prediciton
        specpars = ([
            sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
            sample_i['vrad_{}'.format(ii)],sample_i['vstar'],sample_i['vmic'],
            sample_i['lsf_{}'.format(ii)]])
        specpars += [sample_i['pc{0}_{1}'.format(x,ii)] for x in range(len(pcterms[ii]))]
        
        specmod = genspecfn[ii](specpars,outwave=specwave[ii],modpoly=True)
        specmod_est = jnp.asarray(specmod[1])

        # calculate likelihood for spectrum
        numpyro.sample("specobs_{}".format(ii),distfn.Normal(specmod_est, specsig), obs=specobs[ii])

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

    # make photometry prediction
    photpars = ([
        sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],3.1])
    photmod = genphotfn(photpars)
    photmod = [photmod[xx] for xx in filtarray]
    photmod_est = jnp.asarray(photmod)

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

    # determine how many spectra to fit based on len of specwave
    nspec = len(specwave)
    
    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "Teff",
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        "vstar",
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # for every input spectrum, figure out if user defines extra pc terms in priors
    # or should use the default pc0-pc3 terms
    pcln = []
    pcterms = []
    for ii in range(nspec):
        # determine how many user defined pc priors there are for spec == ii
        pcln_i = len([kk for kk in priors.keys() if all(aa in kk for aa in ['pc','_{}'.format(ii)])])
        pcln.append(pcln_i)
        if pcln_i == 0:
            pcterms_i = ['pc0_{}'.format(ii),'pc1_{}'.format(ii),
                        'pc2_{}'.format(ii),'pc3_{}'.format(ii)]
        else:
            pcterms_i = ['pc{0}_{1}'.format(x,ii) for x in range(pcln_i)]

        pcterms.append(pcterms_i)

        # now sample from priors for pc terms
        for pp in pcterms_i:
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
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

    # sample in a jitter term for error in spectrum
    for ii in range(nspec):
        specsig = jnp.sqrt( (specobserr[ii]**2.0) + (sample_i['specjitter_{}'.format(ii)]**2.0) )

        # make the spectral prediciton
        specpars = ([
            sample_i['Teff'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
            sample_i['vrad_{}'.format(ii)],sample_i['vstar'],sample_i['vmic'],
            sample_i['lsf_{}'.format(ii)]])
        specpars += [sample_i['pc{0}_{1}'.format(x,ii)] for x in range(len(pcterms[ii]))]
        
        specmod = genspecfn[ii](specpars,outwave=specwave[ii],modpoly=True)
        specmod_est = jnp.asarray(specmod[1])

        # calculate likelihood for spectrum
        numpyro.sample("specobs_{}".format(ii),distfn.Normal(specmod_est, specsig), obs=specobs[ii])

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

