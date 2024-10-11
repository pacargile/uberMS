import numpyro
from numpyro.util import enable_x64
enable_x64()
import numpyro.distributions as distfn
from numpyro.distributions import constraints
from numpyro.contrib.control_flow import cond

import jax.numpy as jnp

from .priors import determineprior, defaultprior
from .advancedpriors import Sigmoid_Prior, DSigmoid_Prior

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
    jMISTfn   = fitfunc['jMISTfn']

    # pull out MIST pars
    MISTpars = fitfunc['MISTpars']

    # pull out additional info
    parallax = additionalinfo.get('parallax',[None,None])
    vmicbool = additionalinfo['vmicbool']

    # determine how many spectra to fit based on len of specwave
    nspec = len(specwave)

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "photjitter",
        "EEP",
        "initial_Mass",
        "initial_[Fe/H]",
        "initial_[a/Fe]",
        "vstar",
        "Av",
        "dist",
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
            vrad = numpyro.sample('vrad',distfn.Uniform(priors['radvel'][1][0],priors['radvel'][1][1]))
            for ii in range(nspec):
                pp = 'vrad_{}'.format(ii)
                sample_i[pp] = numpyro.sample(pp,distfn.Uniform(vrad-priors['radvel'][1][2],vrad+priors['radvel'][1][2]))
    else:
        for ii in range(nspec):
            pp = 'vrad_{}'.format(ii)
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
        # verbose=False
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
    teff   = numpyro.deterministic('Teff',10.0**MISTdict['log(Teff)'])
    logg   = numpyro.deterministic('log(g)',MISTdict['log(g)'])
    feh    = numpyro.deterministic('[Fe/H]',MISTdict['[Fe/H]'])
    afe    = numpyro.deterministic('[a/Fe]',MISTdict['[a/Fe]'])
    logr   = numpyro.deterministic('log(R)',MISTdict['log(R)'])
    logage = numpyro.deterministic('log(Age)',MISTdict['log(Age)'])
    age    = numpyro.deterministic('Age',10.0**(logage-9.0))

    # set vmic only if included in NNs
    if vmicbool:
        if 'vmic' in priors.keys():
            sample_i['vmic'] = determineprior('vmic',priors['vmic'],teff,logg)
        else:
            sample_i['vmic'] = defaultprior('vmic')
    else:
        sample_i['vmic'] = 1.0


    # check if user set prior on these latent variables
    for parsample,parname in zip(
        [teff,logg,feh,afe,logr,age,logage],
        ['Teff','log(g)','[Fe/H]','[a/Fe]','log(R)','Age','log(Age)']
        ):
        if parname in priors.keys():
            if priors[parname][0] == 'uniform':
                logprob_i = (
                    distfn.Uniform(
                        low=priors[parname][1][0],high=priors[parname][1][1],
                        validate_args=True).log_prob(parsample)
                        )
            if priors[parname][0] == 'normal':
                logprob_i = distfn.Normal(
                    loc=priors[parname][1][0],scale=priors[parname][1][1]
                    ).log_prob(
                        parsample
                        )
            if priors[parname][0] == 'tnormal':
                logprob_i = distfn.TruncatedDistribution(
                        distfn.Normal(
                            loc=priors[parname][1][0],scale=priors[parname][1][1]
                            ), 
                        low=priors[parname][1][2],high=priors[parname][1][3],
                        validate_args=True).log_prob(parsample)
            if priors[parname][0] == 'sigmoid':
                logprob_i = Sigmoid_Prior(
                        a=priors[parname][1][0],
                        b=priors[parname][1][1],
                        low=priors[parname][1][2],
                        high=priors[parname][1][3],
                        validate_args=True).log_prob(parsample)
            if priors[parname][0] == 'dsigmoid':
                logprob_i = DSigmoid_Prior(
                        a=priors[parname][1][0],
                        b=priors[parname][1][1],
                        c=priors[parname][1][2],
                        d=priors[parname][1][3],
                        low=priors[parname][1][4],
                        high=priors[parname][1][5],
                        validate_args=True).log_prob(parsample)
                
            numpyro.factor('LatentPrior_{}'.format(parname),logprob_i)

    if jMISTfn != None:
        dlogAgedEEP = jMISTfn(jnp.array(
            [sample_i["EEP"],sample_i["initial_Mass"],
             sample_i["initial_[Fe/H]"],sample_i["initial_[a/Fe]"]]
            ))['log(Age)'][0]
        lp_AgeWgt = jnp.where(dlogAgedEEP > 0.0,jnp.log(dlogAgedEEP),-100.0)
        numpyro.factor("AgeWgt_log_prob", lp_AgeWgt)
        numpyro.deterministic("AgeWgt_LP", lp_AgeWgt)

    # sample in a jitter term for error in spectrum
    for ii in range(nspec):
        specsig = jnp.sqrt( (specobserr[ii]**2.0) + (sample_i['specjitter_{}'.format(ii)]**2.0) )

        # make the spectral prediciton
        specpars = ([
            teff,logg,feh,afe,
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
    photpars = [teff,logg,feh,afe,logr,sample_i['dist'],sample_i['Av'],3.1]

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
    jMISTfn   = fitfunc['jMISTfn']

    # pull out MIST pars
    MISTpars = fitfunc['MISTpars']

    # pull out additional info
    vmicbool = additionalinfo['vmicbool']

    # determine how many spectra to fit based on len of specwave
    nspec = len(specwave)

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "EEP",
        "initial_Mass",
        "initial_[Fe/H]",
        "initial_[a/Fe]",
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

        pp = 'specjitter_{}'.format(ii)
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    if 'radvel' in priors.keys():
        print('found radvel!!!!!!')
        if priors['radvel'][0] == 'locked':
            vrad = determineprior('radvel',priors['radvel'])
            for ii in range(nspec):
                pp = 'vrad_{}'.format(ii)
                sample_i[pp] = numpyro.deterministic(pp,vrad)
    else:
        for ii in range(nspec):
            pp = 'vrad_{}'.format(ii)
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
        # verbose=False
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
    teff   = numpyro.deterministic('Teff',10.0**MISTdict['log(Teff)'])
    logg   = numpyro.deterministic('log(g)',MISTdict['log(g)'])
    feh    = numpyro.deterministic('[Fe/H]',MISTdict['[Fe/H]'])
    afe    = numpyro.deterministic('[a/Fe]',MISTdict['[a/Fe]'])
    logr   = numpyro.deterministic('log(R)',MISTdict['log(R)'])
    logage = numpyro.deterministic('log(Age)',MISTdict['log(Age)'])
    age    = numpyro.deterministic('Age',10.0**(logage-9.0))

    # set vmic only if included in NNs
    if vmicbool:
        if 'vmic' in priors.keys():
            sample_i['vmic'] = determineprior('vmic',priors['vmic'],teff,logg)
        else:
            sample_i['vmic'] = defaultprior('vmic')
    else:
        sample_i['vmic'] = 1.0


    # check if user set prior on these latent variables
    for parsample,parname in zip(
        [teff,logg,feh,afe,logr,age,logage],
        ['Teff','log(g)','[Fe/H]','[a/Fe]','log(R)','Age','log(Age)']
        ):
        if parname in priors.keys():
            if priors[parname][0] == 'uniform':
                # logprob_i = jnp.nan_to_num(
                #     distfn.Uniform(
                #         low=priors[parname][1][0],high=priors[parname][1][1],
                #         validate_args=True).log_prob(parsample)
                #         )
                logprob_i = (
                    distfn.Uniform(
                        low=priors[parname][1][0],high=priors[parname][1][1],
                        validate_args=True).log_prob(parsample)
                        )
                # logprob_i = jnp.nan_to_num(jnp.where( 
                #                       (parsample < priors[parname][1][0]) | 
                #                       (parsample > priors[parname][1][1]),
                #                       -jnp.inf, 0.0))
            if priors[parname][0] == 'normal':
                logprob_i = distfn.Normal(
                    loc=priors[parname][1][0],scale=priors[parname][1][1]
                    ).log_prob(
                        parsample
                        )
            if priors[parname][0] == 'tnormal':
                logprob_i = distfn.TruncatedDistribution(
                        distfn.Normal(
                            loc=priors[parname][1][0],scale=priors[parname][1][1]
                            ), 
                        low=priors[parname][1][2],high=priors[parname][1][3],
                        validate_args=True).log_prob(parsample)
            numpyro.factor('LatentPrior_{}'.format(parname),logprob_i)

    if jMISTfn != None:
        dlogAgedEEP = jMISTfn(jnp.array(
            [sample_i["EEP"],sample_i["initial_Mass"],
             sample_i["initial_[Fe/H]"],sample_i["initial_[a/Fe]"]]
            ))['log(Age)'][0]
        lp_AgeWgt = jnp.where(dlogAgedEEP > 0.0,jnp.log(dlogAgedEEP),-100.0)
        numpyro.factor("AgeWgt_log_prob", lp_AgeWgt)
        numpyro.deterministic("AgeWgt_LP", lp_AgeWgt)

    # sample in a jitter term for error in spectrum
    for ii in range(nspec):
        specsig = jnp.sqrt( (specobserr[ii]**2.0) + (sample_i['specjitter_{}'.format(ii)]**2.0) )

        # make the spectral prediciton
        specpars = ([
            teff,logg,feh,afe,
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
    genMISTfn = fitfunc['genMISTfn']
    genphotfn = fitfunc['genphotfn']
    jMISTfn   = fitfunc['jMISTfn']

    # pull out MIST pars
    MISTpars = fitfunc['MISTpars']

    # pull out additional info
    parallax = additionalinfo.get('parallax',[None,None])

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "photjitter",
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

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

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
    teff_offset = numpyro.sample("teff_offset", distfn.Normal(0., 1.))
    teff   = numpyro.deterministic('Teff',(10.0**MISTdict['log(Teff)']) + teff_offset * 500.0)
    logg   = numpyro.deterministic('log(g)',MISTdict['log(g)'])
    feh    = numpyro.deterministic('[Fe/H]',MISTdict['[Fe/H]'])
    afe    = numpyro.deterministic('[a/Fe]',MISTdict['[a/Fe]'])
    logr   = numpyro.deterministic('log(R)',MISTdict['log(R)'])
    logage = numpyro.deterministic('log(Age)',MISTdict['log(Age)'])
    age    = numpyro.deterministic('Age',10.0**(logage-9.0))

    # check if user set prior on these latent variables
    for parsample,parname in zip(
        [teff,logg,feh,afe,logr,age,logage],
        ['Teff','log(g)','[Fe/H]','[a/Fe]','log(R)','Age','log(Age)']
        ):
        if parname in priors.keys():
            if priors[parname][0] == 'uniform':
                logprob_i = distfn.Uniform(
                        low=priors[parname][1][0],high=priors[parname][1][1],
                        validate_args=True).log_prob(parsample)
                        
            if priors[parname][0] == 'normal':
                logprob_i = distfn.Normal(
                    loc=priors[parname][1][0],scale=priors[parname][1][1]
                    ).log_prob(
                        parsample
                        )
            if priors[parname][0] == 'tnormal':
                logprob_i = distfn.TruncatedDistribution(
                        distfn.Normal(
                            loc=priors[parname][1][0],scale=priors[parname][1][1]
                            ), 
                        low=priors[parname][1][2],high=priors[parname][1][3],
                        validate_args=True).log_prob(parsample)
            numpyro.factor('LatentPrior_'+parname,logprob_i)

    if jMISTfn != None:
        dlogAgedEEP = jMISTfn(jnp.array(
            [sample_i["EEP"],sample_i["initial_Mass"],
             sample_i["initial_[Fe/H]"],sample_i["initial_[a/Fe]"]]
            ))['log(Age)'][0]
        lp_AgeWgt = jnp.where(dlogAgedEEP > 0.0,jnp.log(dlogAgedEEP),-100.0)
        numpyro.factor("AgeWgt_log_prob", lp_AgeWgt)
        numpyro.deterministic("AgeWgt_LP", lp_AgeWgt)

    # make photometry prediction
    photpars = jnp.asarray([teff,logg,feh,afe,logr,sample_i['dist'],sample_i['Av'],3.1])
    photmod_est = genphotfn(photpars)
    photmod_est = jnp.asarray([photmod_est[xx] for xx in filtarray])
    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    
    # calcluate likelihood of parallax
    numpyro.sample("para", distfn.Normal(1000.0/sample_i['dist'],parallax[1]), obs=parallax[0])

