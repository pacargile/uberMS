from datetime import datetime
import sys,os,tqdm

import jax
from jax import jit, jacfwd #,lax
from jax import random as jrandom
import jax.numpy as jnp

import numpyro
from numpyro.infer import SVI, autoguide, initialization, Trace_ELBO, RenyiELBO
from numpyro.diagnostics import print_summary

from optax import exponential_decay

from misty.predict import GenModJax as GenMIST
from Payne.jax.genmod import GenMod
from .fitutils import avrv_spec_layout, specsetup, check_initpars, check_numpyro_init, AutoBNAFNormalInit
from .dustprior import dustprior_setup
from .priors import logR_reparam_initpars
from functools import partial

from astropy.table import Table

# add to the user's XLA flags instead of replacing them
if "xla_cpu_use_thunk_runtime" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_use_thunk_runtime=false").strip()

class sviMS(object):
    """docstring for sviMS"""
    def __init__(self, *arg, **kwargs):
        super(sviMS, self).__init__()
        
        # set paths for NN's
        self.specNN = kwargs.get('specNN',None)
        self.contNN = kwargs.get('contNN',None)
        self.photNN = kwargs.get('photNN',None)
        self.mistNN = kwargs.get('mistNN',None)

        # set if to use dEEP/dAge grad
        self.gradbool = kwargs.get('usegrad',True)

        # set type of NN
        
        # for legacy, keep the NNtype
        self.NNtype = kwargs.get('NNtype','LinNet')

        # for the new specNN and photNN
        self.sNNtype = kwargs.get('sNNtype',None)
        self.pNNtype = kwargs.get('pNNtype',None)
        self.mNNtype = kwargs.get('mNNtype',None)

        self.cwcversion = kwargs.get('cwcversion','v1.1')
        self.nnversion = kwargs.get('nnversion','v1')

        # if user did not use the new specNN and photNN
        if self.sNNtype is None:
            self.sNNtype = self.NNtype
        if self.pNNtype is None:
            self.pNNtype = self.NNtype
        if self.mNNtype is None:
            self.mNNtype = self.NNtype

        # set if you want spot model to be applied in model call
        self.applyspot = kwargs.get('applyspot',False)

        self.rng_key = jrandom.PRNGKey(0)

        # initialize prediction classes
        GM = GenMod()

        if self.specNN is not None:
            GM._initspecnn(
                nnpath=self.specNN,
                Cnnpath=self.contNN,
                NNtype=self.sNNtype)
            self.specNN_labels = GM.PP.modpars
            self.genspecfn = jit(GM.genspec)
        else:
            self.specNN_labels = []
            self.genspecfn = None

        # check if spectral NN predicts F_lambda with Av/Rv inputs (e.g., FlxEm),
        # and grab its metadata (resolution, label ranges)
        self.specavrv = (self.specNN is not None) and avrv_spec_layout(self.sNNtype)
        if self.specavrv:
            self.specmeta = getattr(GM.PP.anns.anns,'flux_meta',{})
        else:
            self.specmeta = {}
            
        if self.photNN is not None:
            GM._initphotnn(
                None,
                nnpath=self.photNN,
                NNtype=self.pNNtype,
                cwcversion=self.cwcversion,
                nnversion=self.nnversion,
                )
            # rvfree=True so the photometry uses the sampled Rv (default 3.1)
            self.genphotfn = jit(partial(GM.genphot,rvfree=True))
        else:
            self.genphotfn = None
            
        if self.mistNN is not None:
            GMIST = GenMIST.modpred(
                nnpath=self.mistNN,
                nntype=self.mNNtype,
                normed=True,
                applyspot=self.applyspot)
            self.MISTpars = GMIST.modpararr
            self.genMISTfn = jit(GMIST.getMIST)
        else:
            print('[uberMS] DID NOT READ IN MIST NN, DO YOU WANT TO RUN THE PAYNE?')
            raise IOError

        if self.gradbool:
            def gMIST(pars):
                eep,mass,feh,afe = pars
                return self.genMISTfn(eep=eep,mass=mass,feh=feh,afe=afe)
            jitgMIST = jit(gMIST)
            self.jMISTfn = jacfwd(jitgMIST)
        else:
            self.jMISTfn = None

        self.verbose = kwargs.get('verbose',True)

        if self.verbose:
            print('[uberMS] --------')
            print('[uberMS] MODELS:')
            print('[uberMS] --------')
            print('[uberMS] Spec NN: {}'.format(self.specNN))
            print('[uberMS] Cont NN: {}'.format(self.contNN))
            print('[uberMS] Spec NN-type: {}'.format(self.sNNtype))
            print('[uberMS] Phot NN: {}'.format(self.photNN))
            print('[uberMS] Phot NN-type: {}'.format(self.pNNtype))
            print('[uberMS] MIST NN: {}'.format(self.mistNN))
            print('[uberMS] MIST NN-type: {}'.format(self.mNNtype))

    def run(self,indict,dryrun=False):

        starttime = datetime.now()

        # random seed: indict['seed'] if given, otherwise the fitter's key
        # (seed 0). The fit and the posterior resampling get separate keys.
        rng_key = jrandom.PRNGKey(indict['seed']) if 'seed' in indict else self.rng_key
        fit_key, post_key = jrandom.split(rng_key)

        # break out parts on input dict
        data = indict['data']
        initpars = indict['initpars']
        priors = indict.get('priors',{})
        settings = indict.get('svi',{})

        # default specbool and photbool, user can overwrite it
        self.specbool = indict.get('specbool',True)
        self.photbool = indict.get('photbool',True)

        # determine if spectrum is input
        if 'spec' in data.keys():
            if isinstance(data['spec'],dict):
                specwave_in  = data['spec']['obs_wave']
                specflux_in  = data['spec']['obs_flux']
                speceflux_in = data['spec']['obs_eflux']
            else:               
                specwave_in,specflux_in,speceflux_in = data['spec']
            specwave_in  = jnp.asarray(specwave_in,dtype=float)
            specflux_in  = jnp.asarray(specflux_in,dtype=float)
            speceflux_in = jnp.asarray(speceflux_in,dtype=float)
        else:
            specwave_in  = None
            specflux_in  = None
            speceflux_in = None
            self.specbool = False

        # determine if photometry is input
        if 'phot' in data.keys():
            filterarray = data.get('filtarr',list(data['phot'].keys()))
            phot_in    = jnp.asarray([data['phot'][xx][0] for xx in filterarray],dtype=float)
            photerr_in = jnp.asarray([data['phot'][xx][1] for xx in filterarray],dtype=float)
        else:
            filterarray = []
            phot_in    = None
            photerr_in = None
            self.photbool = False

        # check that the NNs needed for the input data were loaded
        if self.specbool and (self.genspecfn is None):
            raise ValueError('[uberMS] spectrum given in data but no specNN was set')
        if self.photbool and (self.genphotfn is None):
            raise ValueError('[uberMS] photometry given in data but no photNN was set')

        # spectral fitting mode: 'normalized' (default) or 'fluxed', fluxscale
        # gives the units of a fluxed spectrum relative to erg/s/cm^2/A
        specmode = indict.get('specmode','normalized')
        fluxscale = indict.get('fluxscale',1.0)
        if self.specbool:
            priors = specsetup(
                priors,specmode=specmode,fluxscale=fluxscale,
                avrv=self.specavrv,specmeta=self.specmeta,
                specwave=specwave_in,specflux=specflux_in,
                isochrone=True,verbose=self.verbose)

        # build the line-of-sight table for a 3D dust map prior on Av, only
        # needed if the model samples Av
        if self.photbool or (self.specbool and (specmode == 'fluxed')):
            priors = dustprior_setup(priors,verbose=self.verbose)

        # determine which model to use based on 
        if self.specbool and self.photbool:
            from .models_MS import model_specphot as model
        if self.specbool and not self.photbool:
            from .models_MS import model_spec as model
        if self.photbool and not self.specbool:
            from .models_MS import model_phot as model

        # define input dictionary for models
        modelkw = ({
            'indata':{
                'specobs':specflux_in,
                'specobserr':speceflux_in, 
                'specwave':specwave_in,
                'photobs':phot_in,
                'photobserr':photerr_in,
                'filterarray':filterarray,
                },
            'fitfunc':{
                'genspecfn':self.genspecfn,
                'genphotfn':self.genphotfn,
                'genMISTfn':self.genMISTfn,
                'MISTpars': self.MISTpars,
                'jMISTfn':  self.jMISTfn,
                },
            'priors':priors,
            'additionalinfo':{
                }
            })

        # cycle through possible additional parameters
        if 'parallax' in data.keys():
            modelkw['additionalinfo']['parallax'] = data['parallax']
        else:
            modelkw['additionalinfo']['parallax'] = None

        # pass info about if vmic is included in NN labels
        if ('vturb' in self.specNN_labels) | ('vmic' in self.specNN_labels):
            modelkw['additionalinfo']['vmicbool'] = True
        else:
            modelkw['additionalinfo']['vmicbool'] = False

        # check to see if user wants to turn off difusion
        if 'diffbool' in indict.keys():
            modelkw['additionalinfo']['diffbool'] = indict['diffbool']
        else:
            modelkw['additionalinfo']['diffbool'] = True

        # pass info about the spectral fitting mode
        modelkw['additionalinfo']['specmode'] = specmode
        modelkw['additionalinfo']['specavrv'] = self.specavrv
        modelkw['additionalinfo']['fluxscale'] = fluxscale
        
        # define the optimizer
        # optimizer = numpyro.optim.ClippedAdam(settings.get('opt_tol',0.001))
        optimizer = numpyro.optim.ClippedAdam(exponential_decay(settings.get('start_tol',1E-3),3000,0.5, end_value=settings.get('opt_tol',1E-5)))

        # check that the initial values are inside the prior volume, and
        # report every parameter that is not
        check_initpars(model,modelkw,initpars,rng_key=rng_key,verbose=self.verbose)

        # check that numpyro can start the fit from the initial values
        check_numpyro_init(model,modelkw,initpars,rng_key=rng_key,verbose=self.verbose)

        # define the guide
        guide_str = settings.get('guide','Normaling Flow')
        # define the guide
        if guide_str == 'Normal':
            if self.verbose:
                print('[uberMS] ... Using N-D Normal as Guide')
            guide = autoguide.AutoLowRankMultivariateNormal(
                model,
                init_loc_fn=initialization.init_to_value(values=initpars))
        else:
            if self.verbose:
                print('[uberMS] ... Using NF as Guide')
            guide = AutoBNAFNormalInit(
                model,num_flows=settings.get('numflows',2),
                init_loc_fn=initialization.init_to_value(values=initpars),
                init_scale=settings.get('init_scale',0.1))
        
        # # Try the GUIDE
        # try:
        #     numpyro.infer.util.initialize_model(
        #         self.rng_key, 
        #         guide,
        #         dynamic_args=True,
        #         init_strategy=initialization.init_to_value(values=initpars),
        #         model_kwargs=modelkw,
        #     )
        # except Exception as e:
        #     print("[uberMS]Guide init FAILED:", repr(e))

        loss = RenyiELBO(alpha=1.25)

        # build SVI object
        svi = SVI(model, guide, optimizer, loss=loss)
        
        # if dry run, just return useful things
        if dryrun:
            return (svi,model,guide,modelkw)
        
        # run the SVI
        svi_result = svi.run(
            fit_key, 
            settings.get('steps',30000),
            **modelkw,
            progress_bar=settings.get('progress_bar',True),
            stable_update=settings.get('stable_update',True),
            )

        # reconstruct the posterior
        params = svi.get_params(svi_result.state)
        posterior = guide.sample_posterior(
            post_key, 
            params, 
            sample_shape=(settings.get('post_resample',int(settings.get('steps',30000)/3)),)
            )
        if self.verbose:
            print_summary({k: v for k, v in posterior.items() if k != "mu"}, 0.89, False)

        # write posterior samples to an astropy table
        outtable = Table(posterior)

        # determine extra parameter from MIST
        extrapars = [x for x in self.MISTpars if x not in outtable.keys()] 
        for kk in extrapars:
            outtable[kk] = jnp.nan * jnp.ones(len(outtable),dtype=float)

        for t_i in outtable:
            MISTpred = self.genMISTfn(
                eep=t_i['EEP'],
                mass=t_i['initial_Mass'],
                feh=t_i['initial_[Fe/H]'],
                afe=t_i['initial_[a/Fe]'],
                verbose=False
                )
            MISTdict = ({
                kk:MISTpred[kk] for kk in
                self.MISTpars        
            })

            for kk in extrapars:
                t_i[kk] = MISTdict[kk]

        # if self.verbose:
        #     for kk in ['Teff','log(g)','[Fe/H]','[a/Fe]','Age']:
        #         pars = [jnp.median(outtable[kk]),jnp.std(outtable[kk])]
        #         print('{0} = {1:f} +/-{2:f}'.format(kk,pars[0],pars[1]))

        # write out the samples to a file
        outfile = indict['outfile']
        outtable.write(outfile,format='fits',overwrite=True)

        if self.verbose:
            print('[uberMS]... writing samples to {}'.format(outfile))
            print('[uberMS]... Finished: {0}'.format(datetime.now()-starttime))
        sys.stdout.flush()

        jax.clear_caches()

        return (svi,guide,svi_result)


class sviTP(object):
    """ 
    SVI class for running ThePayne
    """
    def __init__(self, *arg, **kwargs):
        super(sviTP, self).__init__()
        
        # set paths for NN's
        self.specNN = kwargs.get('specNN',None)
        self.contNN = kwargs.get('contNN',None)
        self.photNN = kwargs.get('photNN',None)

        # for legacy, keep the NNtype
        self.NNtype = kwargs.get('NNtype','LinNet')

        # for the new specNN and photNN
        self.sNNtype = kwargs.get('sNNtype',None)
        self.pNNtype = kwargs.get('pNNtype',None)

        self.cwcversion = kwargs.get('cwcversion','v1.1')
        self.nnversion = kwargs.get('nnversion','v1')

        # if user did not use the new specNN and photNN
        if self.sNNtype is None:
            self.sNNtype = self.NNtype
        if self.pNNtype is None:
            self.pNNtype = self.NNtype

        self.rng_key = jrandom.PRNGKey(0)

        # initialize prediction classes
        GM = GenMod()

        if self.specNN is not None:
            GM._initspecnn(
                nnpath=self.specNN,
                Cnnpath=self.contNN,
                NNtype=self.sNNtype)
            self.specNN_labels = GM.PP.modpars
            self.genspecfn = jit(GM.genspec)
        else:
            self.specNN_labels = []
            self.genspecfn = None

        # check if spectral NN predicts F_lambda with Av/Rv inputs (e.g., FlxEm),
        # and grab its metadata (resolution, label ranges)
        self.specavrv = (self.specNN is not None) and avrv_spec_layout(self.sNNtype)
        if self.specavrv:
            self.specmeta = getattr(GM.PP.anns.anns,'flux_meta',{})
        else:
            self.specmeta = {}
                
        if self.photNN is not None:
            GM._initphotnn(
                None,
                nnpath=self.photNN,
                NNtype=self.pNNtype,
                cwcversion=self.cwcversion,
                nnversion=self.nnversion,)
            # rvfree=True so the photometry uses the sampled Rv (default 3.1)
            self.genphotfn = jit(partial(GM.genphot,rvfree=True))
        else:
            self.genphotfn = None

        self.verbose = kwargs.get('verbose',True)

        if self.verbose:
            print('[uberMS] --------')
            print('[uberMS] MODELS:')
            print('[uberMS] --------')
            print('[uberMS] Spec NN: {}'.format(self.specNN))
            print(f'[uberMS] Spec NN labels: {self.specNN_labels}')
            print('[uberMS] Cont NN: {}'.format(self.contNN))
            print('[uberMS] Spec NN-type: {}'.format(self.sNNtype))
            print('[uberMS] Phot NN: {}'.format(self.photNN))
            print('[uberMS] Phot NN-type: {}'.format(self.pNNtype))

    def run(self,indict,dryrun=False):

        starttime = datetime.now()

        # random seed: indict['seed'] if given, otherwise the fitter's key
        # (seed 0). The fit and the posterior resampling get separate keys.
        rng_key = jrandom.PRNGKey(indict['seed']) if 'seed' in indict else self.rng_key
        fit_key, post_key = jrandom.split(rng_key)

        # break out parts on input dict
        data = indict['data']
        initpars = indict['initpars']
        priors = indict.get('priors',{})
        settings = indict.get('svi',{})

        # default specbool and photbool, user can overwrite it
        self.specbool = indict.get('specbool',True)
        self.photbool = indict.get('photbool',True)

        # determine if spectrum is input
        if 'spec' in data.keys():
            if isinstance(data['spec'],dict):
                specwave_in  = data['spec']['obs_wave']
                specflux_in  = data['spec']['obs_flux']
                speceflux_in = data['spec']['obs_eflux']
            else:               
                specwave_in,specflux_in,speceflux_in = data['spec']
            specwave_in  = jnp.asarray(specwave_in,dtype=float)
            specflux_in  = jnp.asarray(specflux_in,dtype=float)
            speceflux_in = jnp.asarray(speceflux_in,dtype=float)
        else:
            specwave_in  = None
            specflux_in  = None
            speceflux_in = None
            self.specbool = False

        # determine if photometry is input
        if 'phot' in data.keys():
            filterarray = data.get('filtarr',list(data['phot'].keys()))
            phot_in    = jnp.asarray([data['phot'][xx][0] for xx in filterarray],dtype=float)
            photerr_in = jnp.asarray([data['phot'][xx][1] for xx in filterarray],dtype=float)
        else:
            filterarray = []
            phot_in    = None
            photerr_in = None
            self.photbool = False

        # check that the NNs needed for the input data were loaded
        if self.specbool and (self.genspecfn is None):
            raise ValueError('[uberMS] spectrum given in data but no specNN was set')
        if self.photbool and (self.genphotfn is None):
            raise ValueError('[uberMS] photometry given in data but no photNN was set')

        # spectral fitting mode: 'normalized' (default) or 'fluxed', fluxscale
        # gives the units of a fluxed spectrum relative to erg/s/cm^2/A
        specmode = indict.get('specmode','normalized')
        fluxscale = indict.get('fluxscale',1.0)
        if self.specbool:
            priors = specsetup(
                priors,specmode=specmode,fluxscale=fluxscale,
                avrv=self.specavrv,specmeta=self.specmeta,
                specwave=specwave_in,specflux=specflux_in,verbose=self.verbose)

        # build the line-of-sight table for a 3D dust map prior on Av, only
        # needed if the model samples Av
        if self.photbool or (self.specbool and (specmode == 'fluxed')):
            priors = dustprior_setup(priors,verbose=self.verbose)

        # determine which model to use based on 
        if self.specbool and self.photbool:
            from .models_TP import model_specphot as model
        if self.specbool and not self.photbool:
            from .models_TP import model_spec as model
        if self.photbool and not self.specbool:
            from .models_TP import model_phot as model

        # define input dictionary for models
        modelkw = ({
            'indata':{
                'specobs':specflux_in,
                'specobserr':speceflux_in, 
                'specwave':specwave_in,
                'photobs':phot_in,
                'photobserr':photerr_in,
                'filterarray':filterarray,
                },
            'fitfunc':{
                'genspecfn':self.genspecfn,
                'genphotfn':self.genphotfn,
                },
            'priors':priors,
            'additionalinfo':{
                }
            })

        # cycle through possible additional parameters
        if 'parallax' in data.keys():
            modelkw['additionalinfo']['parallax'] = data['parallax']
        else:
            modelkw['additionalinfo']['parallax'] = None
            
        # pass info about if vmic is included in NN labels
        if ('vturb' in self.specNN_labels) | ('vmic' in self.specNN_labels):
            modelkw['additionalinfo']['vmicbool'] = True
        else:
            modelkw['additionalinfo']['vmicbool'] = False

        # pass info about the spectral fitting mode
        modelkw['additionalinfo']['specmode'] = specmode
        modelkw['additionalinfo']['specavrv'] = self.specavrv
        modelkw['additionalinfo']['fluxscale'] = fluxscale

        # sample log(R) as log(R/dist) with dist, see
        # priors.sample_logR_reparam. On by default for SVI: without it the
        # guide collapses along the narrow R-dist ridge. Set
        # indict['reparam'] = False to turn it off (e.g., when log(R) is
        # set mostly by its prior).
        reparam = indict.get('reparam',True)
        modelkw['additionalinfo']['reparam'] = reparam
        if reparam and (self.photbool or (self.specbool and (specmode == 'fluxed'))):
            initpars = logR_reparam_initpars(initpars,priors,verbose=self.verbose)

        # define the optimizer
        # optimizer = numpyro.optim.ClippedAdam(settings.get('opt_tol',1E-4))
        optimizer = numpyro.optim.ClippedAdam(exponential_decay(settings.get('start_tol',1E-3),3000,0.5, end_value=settings.get('opt_tol',1E-5)))

        # check that the initial values are inside the prior volume, and
        # report every parameter that is not
        check_initpars(model,modelkw,initpars,rng_key=rng_key,verbose=self.verbose)

        # check that numpyro can start the fit from the initial values
        check_numpyro_init(model,modelkw,initpars,rng_key=rng_key,verbose=self.verbose)

        guide_str = settings.get('guide','Normaling Flow')
        # define the guide
        if guide_str == 'Normal':
            if self.verbose:
                print('[uberMS] ... Using N-D Normal as Guide')
            guide = autoguide.AutoLowRankMultivariateNormal(
                model,
                init_loc_fn=initialization.init_to_value(values=initpars))
        else:
            if self.verbose:
                print('[uberMS] ... Using NF as Guide')
            guide = AutoBNAFNormalInit(
                model,num_flows=settings.get('numflows',2),
                init_loc_fn=initialization.init_to_value(values=initpars),
                init_scale=settings.get('init_scale',0.1))

        # loss = Trace_ELBO()
        loss = RenyiELBO()
        
        # build SVI object
        svi = SVI(model, guide, optimizer, loss=loss)

        # if dry run, just return useful things
        if dryrun:
            return (svi,model,guide,modelkw)

        # run the SVI
        svi_result = svi.run(
            fit_key, 
            settings.get('steps',30000),
            **modelkw,
            progress_bar=settings.get('progress_bar',True),
            stable_update=settings.get('stable_update',True),
            )

        # reconstruct the posterior
        params = svi.get_params(svi_result.state)
        posterior = guide.sample_posterior(
            post_key, 
            params, 
            sample_shape=(settings.get('post_resample',int(settings.get('steps',30000)/3)),)
            )
        if self.verbose:
            print_summary({k: v for k, v in posterior.items() if k != "mu"}, 0.89, False)

        # write posterior samples to an astropy table
        outtable = Table(posterior)

        # write out the samples to a file
        outfile = indict['outfile']
        outtable.write(outfile,format='fits',overwrite=True)

        if self.verbose:
            print('[uberMS] ... writing samples to {}'.format(outfile))
            print('[uberMS] ... Finished: {0}'.format(datetime.now()-starttime))
        sys.stdout.flush()

        return (svi,guide,svi_result)
        