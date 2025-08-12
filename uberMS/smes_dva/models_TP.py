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

    # determine how many spectra to fit based on len of specwave
    nspec = len(specwave)

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "photjitter",
        "vstar_a",
        "vstar_b",
        'log(R)_a',
        'log(R)_b',
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
    sample_i['Teff_a'] = numpyro.sample("Teff_a",distfn.Uniform(2500.0, 10000.0))
    sample_i['Teff_b'] = numpyro.sample("Teff_b",distfn.Uniform(2500.0, sample_i['Teff_a']))
    # sample_i['Teff_s'] = numpyro.sample("Teff_s",distfn.Uniform(2500.0, 10000.0))

    sample_i['log(g)_a'] = numpyro.sample("log(g)_a",distfn.Uniform(0.0, 5.5))
    # sample_i['log(g)_s'] = numpyro.sample("log(g)_s",distfn.Uniform(sample_i['log(g)_p'],5.5))
    sample_i['log(g)_b'] = numpyro.sample("log(g)_b",distfn.Uniform(0.0,5.5))

    # require that |vrad_p - vrad_s| > 1.0
    # mixing_dist = distfn.Categorical(probs=jnp.ones(2) / 2.)
    # component_dists = ([
    #     distfn.Uniform(sample_i['vrad_p']-100.0,sample_i['vrad_p']-1.0,),
    #     distfn.Uniform(sample_i['vrad_p']+1.0,sample_i['vrad_p']+100.0,),
    #     ])
    # sample_i['vrad_s'] = numpyro.sample('vrad_s',distfn.MixtureGeneral(mixing_dist, component_dists))
    
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

    if 'MassRatio' in priors.keys():
        vrad_a_arr, vrad_b_arr = determineprior('MassRatio',priors['MassRatio'],nspec)
        for ii in range(nspec):
            sample_i[f'vrad_a_{ii}'] = vrad_a_arr[ii]
            sample_i[f'vrad_b_{ii}'] = vrad_b_arr[ii]
    else:
        for ii in range(nspec):
            sample_i[f'vrad_a_{ii}'] = determineprior(f'vrad_a_{ii}',priors[f'vrad_a_{ii}'])
            sample_i[f'vrad_b_{ii}'] = determineprior(f'vrad_b_{ii}',priors[f'vrad_b_{ii}'])


    # set vmic only if included in NNs
    if vmicbool:
        if 'vmic_a' in priors.keys():
            sample_i['vmic_a'] = determineprior('vmic_a',priors['vmic_a'],sample_i['Teff_a'],sample_i['log(g)_a'])
        else:
            sample_i['vmic_p'] = defaultprior('vmic_p')
        if 'vmic_b' in priors.keys():
            sample_i['vmic_b'] = determineprior('vmic_b',priors['vmic_b'],sample_i['Teff_b'],sample_i['log(g)_b'])
        else:
            sample_i['vmic_b'] = defaultprior('vmic_b')
    else:
        sample_i['vmic_a'] = 1.0
        sample_i['vmic_b'] = 1.0

    # handle different cases for the treatment of [Fe/H] and [a/Fe]
    if 'binchem' in priors.keys():
        (sample_i["[Fe/H]_a"],sample_i["[Fe/H]_b"],sample_i["[a/Fe]_a"],sample_i["[a/Fe]_b"]) = determineprior('binchem',priors['binchem'])
    else:
        for pp in ["[Fe/H]_a","[Fe/H]_b","[a/Fe]_a","[a/Fe]_b"]:  
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
            else:
                sample_i[pp] = defaultprior(pp)

    # compute the spectrum and the likelihood of the spectrum
    # first calculate the radius ratio
    radius_a = 10.0**sample_i['log(R)_a']
    radius_b = 10.0**sample_i['log(R)_b']


    for ii in range(nspec):
        # sample in a jitter term for error in spectrum
        specsig = jnp.sqrt( (specobserr[ii]**2.0) + (sample_i[f'specjitter_{ii}']**2.0) )

        # make the spectral prediciton
        specpars_a = ([
            sample_i['Teff_a'],sample_i['log(g)_a'],sample_i['[Fe/H]_a'],sample_i['[a/Fe]_a'],
            sample_i[f'vrad_a_{ii}'],sample_i['vstar_a'],sample_i['vmic_a'],sample_i[f'lsf_{ii}']])
        specpars_a += [sample_i['pc{0}_{1}'.format(x,ii)] for x in range(len(pcterms[ii]))]
        specmod_a = genspecfn[ii](specpars_a,outwave=specwave[ii],modpoly=True)
        specmod_a = jnp.asarray(specmod_a[1])

        specpars_b = ([
            sample_i['Teff_b'],sample_i['log(g)_b'],sample_i['[Fe/H]_b'],sample_i['[a/Fe]_b'],
            sample_i[f'vrad_b_{ii}'],sample_i['vstar_b'],sample_i[f'vmic_b'],sample_i[f'lsf_{ii}']])
        specpars_b += [1.0,0.0]
        specmod_b = genspecfn[ii](specpars_b,outwave=specwave[ii],modpoly=True)
        specmod_b = jnp.asarray(specmod_b[1])

        R = (
            (planck(specwave[ii],sample_i['Teff_a']) * radius_a**2.0) / 
            (planck(specwave[ii],sample_i['Teff_b']) * radius_b**2.0)
            )

        specmod_est = (specmod_a + R * specmod_b) / (1.0 + R)

        # calculate likelihood for spectrum
        numpyro.sample(f"specobs_{ii}",distfn.Normal(specmod_est, specsig), obs=specobs[ii])

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

    # make photometry prediction
    photpars_a = ([
        sample_i['Teff_a'],sample_i['log(g)_a'],sample_i['[Fe/H]_a'],sample_i['[a/Fe]_a'],
        sample_i['log(R)_a'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_a = genphotfn(photpars_a)
    photmod_a = [photmod_a[xx] for xx in filtarray]

    photpars_b = ([
        sample_i['Teff_b'],sample_i['log(g)_b'],sample_i['[Fe/H]_b'],sample_i['[a/Fe]_b'],
        sample_i['log(R)_b'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_b = genphotfn(photpars_b)
    photmod_b = [photmod_b[xx] for xx in filtarray]

    photmod_est = (
        [-2.5 * jnp.log10( 10.0**(-0.4 * m_p) + 10.0**(-0.4 * m_s) )
         for m_p,m_s in zip(photmod_a,photmod_b)
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
        "vstar_a",
        "vstar_b",
        'log(R)_a',
        'log(R)_b',
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # define the primary as the hotter of the two stars
    sample_i['Teff_a'] = numpyro.sample("Teff_a",distfn.Uniform(2500.0, 10000.0))
    sample_i['Teff_b'] = numpyro.sample("Teff_b",distfn.Uniform(2500.0, sample_i['Teff_a']))
    # sample_i['Teff_s'] = numpyro.sample("Teff_s",distfn.Uniform(2500.0, 10000.0))

    sample_i['log(g)_a'] = numpyro.sample("log(g)_a",distfn.Uniform(0.0, 5.5))
    # sample_i['log(g)_s'] = numpyro.sample("log(g)_s",distfn.Uniform(sample_i['log(g)_p'],5.5))
    sample_i['log(g)_b'] = numpyro.sample("log(g)_b",distfn.Uniform(0.0,5.5))

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
        if 'vmic_a' in priors.keys():
            sample_i['vmic_a'] = determineprior('vmic_a',priors['vmic_a'],sample_i['Teff_a'],sample_i['log(g)_a'])
        else:
            sample_i['vmic_p'] = defaultprior('vmic_p')
        if 'vmic_b' in priors.keys():
            sample_i['vmic_b'] = determineprior('vmic_b',priors['vmic_b'],sample_i['Teff_b'],sample_i['log(g)_b'])
        else:
            sample_i['vmic_b'] = defaultprior('vmic_b')
    else:
        sample_i['vmic_a'] = 1.0
        sample_i['vmic_b'] = 1.0

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
        (sample_i["[Fe/H]_a"],sample_i["[Fe/H]_b"],sample_i["[a/Fe]_a"],sample_i["[a/Fe]_b"]) = determineprior('binchem',priors['binchem'])
    else:
        for pp in ["[Fe/H]_a","[Fe/H]_b","[a/Fe]_a","[a/Fe]_b"]:  
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
            else:
                sample_i[pp] = defaultprior(pp)

    if 'MassRatio' in priors.keys():
        sample_i['vrad_a'], sample_i['vrad_b'] = determineprior('MassRatio',priors['MassRatio'])
    else:
        sample_i['vrad_a'] = determineprior('vrad_a',priors['vrad_a'])
        sample_i['vrad_b'] = determineprior('vrad_b',priors['vrad_b'])

    # sample in a jitter term for error in spectrum
    specsig = jnp.sqrt( (specobserr**2.0) + (sample_i['specjitter']**2.0) )

    # make the spectral prediciton
    specpars_a = ([
        sample_i['Teff_a'],sample_i['log(g)_a'],sample_i['[Fe/H]_a'],sample_i['[a/Fe]_a'],
        sample_i['vrad_a'],sample_i['vstar_a'],sample_i['vmic_a'],sample_i['lsf']])
    specpars_a += [sample_i['pc{0}'.format(x)] for x in range(len(pcterms))]
    specmod_a = genspecfn(specpars_a,outwave=specwave,modpoly=True)
    specmod_a = jnp.asarray(specmod_a[1])

    specpars_b = ([
        sample_i['Teff_b'],sample_i['log(g)_b'],sample_i['[Fe/H]_b'],sample_i['[a/Fe]_b'],
        sample_i['vrad_b'],sample_i['vstar_b'],sample_i['vmic_b'],sample_i['lsf']])
    specpars_b += [1.0,0.0]
    specmod_b = genspecfn(specpars_b,outwave=specwave,modpoly=True)
    specmod_b = jnp.asarray(specmod_b[1])

    radius_a = 10.0**sample_i['log(R)_a']
    radius_b = 10.0**sample_i['log(R)_b']

    R = (
        (planck(specwave,sample_i['Teff_a']) * radius_a**2.0) / 
        (planck(specwave,sample_i['Teff_b']) * radius_b**2.0)
         )
    specmod_est = (specmod_a + R * specmod_b) / (1.0 + R)

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
        'log(R)_a',
        'log(R)_b',
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
    sample_i['Teff_a'] = numpyro.sample("Teff_a",distfn.Uniform(2500.0, 10000.0))
    sample_i['Teff_b'] = numpyro.sample("Teff_b",distfn.Uniform(2500.0, sample_i['Teff_a']))
    # sample_i['Teff_s'] = numpyro.sample("Teff_s",distfn.Uniform(2500.0, 10000.0))

    sample_i['log(g)_a'] = numpyro.sample("log(g)_a",distfn.Uniform(0.0, 5.5))
    # sample_i['log(g)_s'] = numpyro.sample("log(g)_s",distfn.Uniform(sample_i['log(g)_p'],5.5))
    sample_i['log(g)_b'] = numpyro.sample("log(g)_b",distfn.Uniform(0.0,5.5))

    # handle different cases for the treatment of [Fe/H] and [a/Fe]
    if 'binchem' in priors.keys():
        (sample_i["[Fe/H]_a"],sample_i["[Fe/H]_b"],sample_i["[a/Fe]_a"],sample_i["[a/Fe]_b"]) = determineprior(None,priors['binchem'])
    else:
        for pp in ["[Fe/H]_a","[Fe/H]_b","[a/Fe]_a","[a/Fe]_b"]:  
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
            else:
                sample_i[pp] = defaultprior(pp)

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

    # make photometry prediction
    photpars_a = ([
        sample_i['Teff_a'],sample_i['log(g)_a'],sample_i['[Fe/H]_a'],sample_i['[a/Fe]_a'],
        sample_i['log(R)_a'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_a = genphotfn(photpars_a)
    photmod_a = [photmod_a[xx] for xx in filtarray]

    photpars_b = ([
        sample_i['Teff_b'],sample_i['log(g)_b'],sample_i['[Fe/H]_b'],sample_i['[a/Fe]_b'],
        sample_i['log(R)_b'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_b = genphotfn(photpars_b)
    photmod_b = [photmod_b[xx] for xx in filtarray]

    photmod_est = (
        [-2.5 * jnp.log10( 10.0**(-0.4 * m_p) + 10.0**(-0.4 * m_s) )
         for m_p,m_s in zip(photmod_a,photmod_b)
         ] 
    )
    photmod_est = jnp.asarray(photmod_est)

    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    
    # calcluate likelihood of parallax
    numpyro.sample("para", distfn.Normal(1000.0/sample_i['dist'],parallax[1]), obs=parallax[0])
