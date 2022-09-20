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
    genMISTfn = fitfunc['genMISTfn']
    genphotfn = fitfunc['genphotfn']
    genspecfn = fitfunc['genspecfn']

    # pull out MIST pars
    MISTpars = fitfunc['MISTpars']

    # pull out additional info
    parallax = additionalinfo.get('parallax',[None,None])
    vmicbool = additionalinfo['vmicbool']

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "EEP",
        "initial_Mass",
        "initial_[Fe/H]",
        "initial_[a/Fe]",
        "vrad",
        "vstar",
        "pc0",
        "pc1",
        "pc2",
        "pc3",
        "Av",
        "dist",
        ])

    sample_i = {}
    for pp in sampledpars:
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # set vmic only if included in NNs
    if vmicbool:
        if 'vmic' in priors.keys():
            sample_i['vmic'] = determineprior('vmic',priors['vmic'])
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

    # predict MIST parameters
    MISTpred = genMISTfn(
        eep=sample_i["EEP"],
        mass=sample_i["initial_Mass"],
        feh=sample_i["initial_[Fe/H]"],
        afe=sample_i["initial_[a/Fe]"],
        # verbose=False
        )
    # set parameters into dictionary
    MISTdict = ({
        kk:pp for kk,pp in zip(
        MISTpars,MISTpred)
        })


    # pull out atmospheric parameters
    teff = 10.0**MISTdict['log(Teff)']
    logg = MISTdict['log(g)']
    feh  = MISTdict['[Fe/H]']
    afe  = MISTdict['[a/Fe]']
    logr = MISTdict['log(R)']
    logage = MISTdict['log(Age)']
    age  = 10.0**(logage-9.0)

    # check if user set prior on these latent variables
    for parsample,parname in zip(
        [teff,logg,feh,afe,logr,age,logage],
        ['Teff','log(g)','[Fe/H]','[a/Fe]','log(R)','Age','log(Age)']
        ):
        if parname in priors.keys():
            if priors[parname][0] == 'uniform':
                numpyro.factor(parname,
                    distfn.Uniform(
                        low=priors[parname][1][0],high=priors[parname][1][1],
                        validate_args=True).log_prob(
                        parsample
                        ))
            if priors[parname][0] == 'normal':
                numpyro.factor(parname,
                    distfn.Normal(
                        loc=priors[parname][1][0],scale=priors[parname][1][1]
                        ).log_prob(
                        parsample
                        ))
            if priors[parname][0] == 'tnormal':
                numpyro.factor(parname,
                    distfn.TruncatedDistribution(
                        distfn.Normal(
                            loc=priors[parname][1][0],scale=priors[parname][1][1]
                            ), 
                        low=priors[parname][1][2],high=priors[parname][1][3],
                        validate_args=True).log_prob(
                        parsample
                        ))

    # dlogAgedEEP = jMIST(jnp.array([eep_i,mass_i,feh_i,afe_i]))[4][0]
    # numpyro.factor("AgeWgt_log_prob", jnp.log(dlogAgedEEP))

    # sample in a jitter term for error in spectrum
    specjitter = numpyro.sample("specjitter", distfn.HalfNormal(0.001))
    specsig = jnp.sqrt( (specobserr**2.0) + (specjitter**2.0) )

    # make the spectral prediciton
    specpars = [teff,logg,feh,afe,sample_i['vrad'],sample_i['vstar'],sample_i['vmic'],sample_i['lsf']]
    specpars += [sample_i['pc0'],sample_i['pc1'],sample_i['pc2'],sample_i['pc3']]

    specmod_est = genspecfn(specpars,outwave=specwave,modpoly=True)
    specmod_est = jnp.asarray(specmod_est[1])
    # calculate likelihood for spectrum
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)

    # sample in jitter term for error in photometry
    photjitter = numpyro.sample("photjitter", distfn.HalfNormal(0.001))
    photsig = jnp.sqrt( (photobserr**2.0) + (photjitter**2.0) )

    # make photometry prediction
    photpars = [teff,logg,feh,afe,logr,sample_i['dist'],sample_i['Av'],3.1]

    photmod_est = genphotfn(photpars)
    photmod_est = jnp.asarray([photmod_est[xx] for xx in filtarray])
    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    
    # calcluate likelihood of parallax
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
    genMISTfn = fitfunc['genMISTfn']
    genphotfn = fitfunc['genphotfn']

    # pull out MIST pars
    MISTpars = fitfunc['MISTpars']

    # pull out additional info
    parallax = additionalinfo.get('parallax',[None,None])

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "EEP",
        "initial_Mass",
        "initial_[Fe/H]",
        "initial_[a/Fe]",
        "dist",
        "Av",
        ])

    sample_i = {}
    for pp in sampledpars:            
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)


    # predict MIST parameters
    MISTpred = genMISTfn(
        eep=sample_i["EEP"],
        mass=sample_i["initial_Mass"],
        feh=sample_i["initial_[Fe/H]"],
        afe=sample_i["initial_[a/Fe]"],
        verbose=False
        )
    # set parameters into dictionary
    # MISTdict = ({
    #     kk:pp for kk,pp in zip(
    #     MISTpars,MISTpred)
    #     })

    MISTdict = ({
        kk:MISTpred[kk] for kk in
        MISTpars        
    })

    # pull out atmospheric parameters
    teff = 10.0**MISTdict['log(Teff)']
    logg = MISTdict['log(g)']
    feh  = MISTdict['[Fe/H]']
    afe  = MISTdict['[a/Fe]']
    logr = MISTdict['log(R)']
    age  = 10.0**(MISTdict['log(Age)']-9.0)

    # check if user set prior on these latent variables
    for parsample,parname in zip(
        [teff,logg,feh,afe,logr,age],
        ['Teff','log(g)','[Fe/H]','[a/Fe]','log(R)','Age']
        ):
        if parname in priors.keys():
            if priors[parname][0] == 'uniform':
                numpyro.factor(parname,
                    distfn.Uniform(
                        low=priors[parname][1][0],high=priors[parname][1][1],
                        validate_args=True).log_prob(
                        parsample
                        ))
            if priors[parname][0] == 'normal':
                numpyro.factor(parname,
                    distfn.Normal(
                        loc=priors[parname][1][0],scale=priors[parname][1][1]
                        ).log_prob(
                        parsample
                        ))
            if priors[parname][0] == 'tnormal':
                numpyro.factor(parname,
                    distfn.TruncatedDistribution(
                        distfn.Normal(
                            loc=priors[parname][1][0],scale=priors[parname][1][1]
                            ), 
                        low=priors[1][2],high=priors[1][3],
                        validate_args=True).log_prob(
                        parsample
                        ))

    # dlogAgedEEP = jMIST(jnp.array([eep_i,mass_i,feh_i,afe_i]))[4][0]
    # numpyro.factor("AgeWgt_log_prob", jnp.log(dlogAgedEEP))

    # sample in jitter term for error in photometry
    photjitter = numpyro.sample("photjitter", distfn.HalfNormal(0.001))
    photsig = jnp.sqrt( (photobserr**2.0) + (photjitter**2.0) )

    # make photometry prediction
    photpars = jnp.asarray([teff,logg,feh,afe,logr,sample_i['dist'],sample_i['Av'],3.1])
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
    genMISTfn = fitfunc['genMISTfn']
    genspecfn = fitfunc['genspecfn']

    # pull out MIST pars
    MISTpars = fitfunc['MISTpars']

    # pull out additional info
    vmicbool = additionalinfo['vmicbool']

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "EEP",
        "initial_Mass",
        "initial_[Fe/H]",
        "initial_[a/Fe]",
        "vrad",
        "vstar",
        "pc0",
        "pc1",
        "pc2",
        "pc3",
        ])

    sample_i = {}
    for pp in sampledpars:            
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # set vmic only if included in NNs
    if vmicbool:
        if 'vmic' in priors.keys():
            sample_i['vmic'] = determineprior('vmic',priors['vmic'])
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

    # predict MIST parameters
    MISTpred = genMISTfn(
        eep=sample_i["EEP"],
        mass=sample_i["initial_Mass"],
        feh=sample_i["initial_[Fe/H]"],
        afe=sample_i["initial_[a/Fe]"],
        verbose=False
        )
    # set parameters into dictionary
    # MISTdict = ({
    #     kk:pp for kk,pp in zip(
    #     MISTpars,MISTpred)
    #     })
    MISTdict = ({
        kk:MISTpred[kk] for kk in
        MISTpars        
    })

    # pull out atmospheric parameters
    teff = 10.0**MISTdict['log(Teff)']
    logg = MISTdict['log(g)']
    feh  = MISTdict['[Fe/H]']
    afe  = MISTdict['[a/Fe]']
    logr = MISTdict['log(R)']
    age  = 10.0**(MISTdict['log(Age)']-9.0)

    # check if user set prior on these latent variables
    for parsample,parname in zip(
        [teff,logg,feh,afe,logr,age],
        ['Teff','log(g)','[Fe/H]','[a/Fe]','log(R)','Age']
        ):
        if parname in priors.keys():
            if priors[parname][0] == 'uniform':
                numpyro.factor(parname,
                    distfn.Uniform(
                        low=priors[parname][1][0],high=priors[parname][1][1],
                        validate_args=True).log_prob(
                        parsample
                        ))
            if priors[parname][0] == 'normal':
                numpyro.factor(parname,
                    distfn.Normal(
                        loc=priors[parname][1][0],scale=priors[parname][1][1]
                        ).log_prob(
                        parsample
                        ))
            if priors[parname][0] == 'tnormal':
                numpyro.factor(parname,
                    distfn.TruncatedDistribution(
                        distfn.Normal(
                            loc=priors[parname][1][0],scale=priors[parname][1][1]
                            ), 
                        low=priors[1][2],high=priors[1][3],
                        validate_args=True).log_prob(
                        parsample
                        ))

    # dlogAgedEEP = jMIST(jnp.array([eep_i,mass_i,feh_i,afe_i]))[4][0]
    # numpyro.factor("AgeWgt_log_prob", jnp.log(dlogAgedEEP))

    # sample in a jitter term for error in spectrum
    specjitter = numpyro.sample("specjitter", distfn.HalfNormal(0.001))
    specsig = jnp.sqrt( (specobserr**2.0) + (specjitter**2.0) )

    # make the spectral prediciton
    specpars = [teff,logg,feh,afe,sample_i['vrad'],sample_i['vstar'],sample_i['vmic'],sample_i['lsf']]
    specpars += [sample_i['pc0'],sample_i['pc1'],sample_i['pc2'],sample_i['pc3']]
    specmod_est = genspecfn(specpars,outwave=specwave,modpoly=True)
    specmod_est = jnp.asarray(specmod_est[1])
    # calculate likelihood for spectrum
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)
