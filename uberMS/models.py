# define the model
def model_specphot(
    indata={},
    fitfunc={},
    priors={},
    additionalinfo={},):

    # specwave=None,specobs=None,specobserr=None,
    # photobs=None,photobserr=None,
    # genspecfn=None,
    # genphotfn=None,
    # genMISTfn=None,
    # MISTpars=None,
    # jMIST=None,
    # parallax=None,
    # filtarr=None,
    # lsf=None,
    # SFD_Av=None,
    # RVest=None,
    # ):

    eep_i  = numpyro.sample("eep", distfn.Uniform(300,800))
    # mass_i = numpyro.sample("initial_Mass",   distfn.Uniform(0.5,3.0))
    mass_i = numpyro.sample('initial_Mass', IMF_Prior())
    feh_i  = numpyro.sample("initial_[Fe/H]", distfn.Uniform(-3.0,0.25))
    afe_i  = numpyro.sample("initial_[a/Fe]", distfn.Uniform(-0.15,0.55))

    MISTpred = genMISTfn(
        eep=eep_i,
        mass=mass_i,
        feh=feh_i,
        afe=afe_i,
        verbose=False
        )

    # dlogAgedEEP = jMIST(jnp.array([eep_i,mass_i,feh_i,afe_i]))[4][0]
    # numpyro.factor("AgeWgt_log_prob", jnp.log(dlogAgedEEP))

    MISTdict = ({
        kk:pp for kk,pp in zip(
        MISTpars,MISTpred)
        })

    numpyro.factor('AgePrior',
        distfn.Uniform(low=9.0,high=10.15,validate_args=True).log_prob(MISTdict['log(Age)']))

    teff = 10.0**MISTdict['log(Teff)']
    logg = MISTdict['log(g)']
    feh  = MISTdict['[Fe/H]']
    afe  = MISTdict['[a/Fe]']

    # teff = numpyro.sample("teff", distfn.Uniform(3500.0, 10000.0))
    # logg = numpyro.sample("logg", distfn.Uniform(0.0, 5.5))
    # feh  = numpyro.sample("feh",  distfn.Uniform(-3.0, 0.5))
    # afe  = numpyro.sample("afe",  distfn.Uniform(-0.2, 0.6))

    vrad = numpyro.sample("vrad", distfn.Uniform(RVest-25.0, RVest+25.0))
    vrot = numpyro.sample("vrot", distfn.Uniform(0.01, 25.0))

    # vmic = numpyro.sample("vmic", distfn.Uniform(0.5, 3.0))
    # Ramirez, Allende Prieto, Lambert (2013)
    # vmic_p = 1.163 + (7.808E-4)*(teff-5800.0)-0.494*(logg-4.30)-0.050*feh
    # vmic  = numpyro.sample('vmic', distfn.TruncatedNormal(low=0.5,loc=vmic_p,scale=0.12))    

    # vmic_p = 1.163 + (7.808E-4)*(teff-5800.0)-0.494*(logg-4.30)-0.050*feh
    # vmic  = numpyro.sample('vmic', distfn.TruncatedDistribution(
    #     distfn.Normal(loc=vmic_p,scale=0.12),low=0.6,high=2.9))

    vmic = 1.0

    pc0  = numpyro.sample('pc0', distfn.Uniform(0.5,1.5))
    pc1  = numpyro.sample('pc1', distfn.Normal(0.0,0.25))
    pc2  = numpyro.sample('pc2', distfn.Normal(0.0,0.25))
    pc3  = numpyro.sample('pc3', distfn.Normal(0.0,0.25))
        
    instr_scale = numpyro.sample('instr_scale', distfn.Uniform(0.5,3.0))
    instr = lsf * instr_scale

    specpars = [teff,logg,feh,afe,vrad,vrot,vmic,instr]
    specpars += [pc0,pc1,pc2,pc3]

    specmod_est = genspecfn(specpars,outwave=specwave,modpoly=True)
    specmod_est = jnp.asarray(specmod_est[1])

    # numpyro.sample("specobs",distfn.Normal(specmod_est, 0.01), obs=specobs)
    # numpyro.sample("specobs",distfn.Normal(specmod_est, specobserr), obs=specobs)
    specjitter = numpyro.sample("specjitter", distfn.HalfNormal(0.001))
    specsig = jnp.sqrt( (specobserr**2.0) + (specjitter**2.0) )
    # specsig = specobserr
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)

    logr = MISTdict['log(R)']
    dist = numpyro.sample("dist", distfn.Uniform(1.0, 200000.0))
    # av   = numpyro.sample("av", distfn.TruncatedNormal(low=1E-6,loc=0.0,scale=3.0*SFD_Av))
    av = numpyro.sample("av", distfn.Uniform(1E-6, 3.0*SFD_Av))

    numpyro.sample("para", distfn.Normal(1000.0/dist,parallax[1]), obs=parallax[0])

    photpars = jnp.asarray([teff,logg,feh,afe,logr,dist,av,3.1])
    photmod_est = genphotfn(photpars)
    photmod_est = jnp.asarray([photmod_est[xx] for xx in filtarr])

    # numpyro.sample("photobs",distfn.Normal(photmod_est, 0.1), obs=photobs)
    # numpyro.sample("photobs",distfn.Normal(photmod_est, photobserr), obs=photobs)
    photjitter = numpyro.sample("photjitter", distfn.HalfNormal(0.001))
    photsig = jnp.sqrt( (photobserr**2.0) + (photjitter**2.0) )
    # photsig = photobserr
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    
