import numpyro
from numpyro.infer import MCMC, NUTS,initialization
from numpyro.diagnostics import print_summary

import jax
from jax import jit,lax,jacfwd
from jax import random as jrandom
import jax.numpy as jnp

from misty.predict import GenModJax as GenMIST
from Payne.jax.genmod import GenMod
from .fitutils import avrv_spec_layout, specsetup, check_initpars, check_numpyro_init

from datetime import datetime
import sys,os
from astropy.table import Table

try:
    os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
except:
    pass

def readdata(data, specbool=True, photbool=True):
    """
    Pull the spectrum and photometry out of the input data dict, returns
    (specwave, specflux, speceflux, filterarray, phot, photerr, specbool, photbool)
    """
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
        specbool = False

    # determine if photometry is input
    if 'phot' in data.keys():
        filterarray = data.get('filtarr',list(data['phot'].keys()))
        phot_in    = jnp.asarray([data['phot'][xx][0] for xx in filterarray],dtype=float)
        photerr_in = jnp.asarray([data['phot'][xx][1] for xx in filterarray],dtype=float)
    else:
        filterarray = []
        phot_in    = None
        photerr_in = None
        photbool = False

    return (specwave_in,specflux_in,speceflux_in,
            filterarray,phot_in,photerr_in,specbool,photbool)

def buildmcmc(model, initpars, settings):
    """
    Build the NUTS kernel and MCMC object from the user settings
    """
    nuts_kernel = NUTS(model,
        dense_mass=settings.get('dense_mass',True),
        target_accept_prob=settings.get('target_accept_prob',0.8),
        max_tree_depth=settings.get('max_tree_depth',10),
        init_strategy=initialization.init_to_value(values=initpars),
        )
    mcmc = MCMC(nuts_kernel,
        num_samples=settings.get('steps',500),
        num_warmup=settings.get('warmup',150),
        progress_bar=settings.get('progress_bar',True))
    return nuts_kernel,mcmc

class nutsMS(object):
    """
    NUTS class for running MIST + ThePayne
    """
    def __init__(self, *arg, **kwargs):
        super(nutsMS, self).__init__()

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
            self.genphotfn = jit(GM.genphot)
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

        # break out parts on input dict
        data = indict['data']
        initpars = indict['initpars']
        priors = indict.get('priors',{})
        settings = indict.get('nuts',{})

        # default specbool and photbool, user can overwrite it, and
        # pull out the input data
        (specwave_in,specflux_in,speceflux_in,
         filterarray,phot_in,photerr_in,
         self.specbool,self.photbool) = readdata(
            data,specbool=indict.get('specbool',True),photbool=indict.get('photbool',True))

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

        # check that the initial values are inside the prior volume, and
        # report every parameter that is not
        check_initpars(model,modelkw,initpars,rng_key=self.rng_key,verbose=self.verbose)

        # check that numpyro can start the fit from the initial values
        check_numpyro_init(model,modelkw,initpars,rng_key=self.rng_key,verbose=self.verbose)

        # build the sampler
        nuts_kernel,mcmc = buildmcmc(model,initpars,settings)

        # if dry run, just return useful things
        if dryrun:
            return (nuts_kernel,mcmc,model,modelkw)

        mcmc.run(
            self.rng_key,
            **modelkw,
            )

        # write posterior samples to an astropy table
        posterior = mcmc.get_samples()
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

        # write out the samples to a file
        outfile = indict['outfile']
        outtable.write(outfile,format='fits',overwrite=True)

        if self.verbose:
            print('[uberMS]... writing samples to {}'.format(outfile))
            print('[uberMS]... Finished: {0}'.format(datetime.now()-starttime))
        sys.stdout.flush()

        jax.clear_caches()

        return (nuts_kernel,mcmc)


class nutsTP(object):
    """
    NUTS class for running ThePayne
    """
    def __init__(self, *arg, **kwargs):
        super(nutsTP, self).__init__()

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
            self.genphotfn = jit(GM.genphot)
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

        # break out parts on input dict
        data = indict['data']
        initpars = indict['initpars']
        priors = indict.get('priors',{})
        if 'nuts' in indict.keys():
            settings = indict['nuts']
        else:
            # older versions of nutsTP read the NUTS settings from indict['svi']
            settings = indict.get('svi',{})
            if 'svi' in indict.keys():
                print("[uberMS] WARNING: reading NUTS settings from indict['svi'], "
                      "please move them to indict['nuts']")

        # default specbool and photbool, user can overwrite it, and
        # pull out the input data
        (specwave_in,specflux_in,speceflux_in,
         filterarray,phot_in,photerr_in,
         self.specbool,self.photbool) = readdata(
            data,specbool=indict.get('specbool',True),photbool=indict.get('photbool',True))

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

        # check that the initial values are inside the prior volume, and
        # report every parameter that is not
        check_initpars(model,modelkw,initpars,rng_key=self.rng_key,verbose=self.verbose)

        # check that numpyro can start the fit from the initial values
        check_numpyro_init(model,modelkw,initpars,rng_key=self.rng_key,verbose=self.verbose)

        # build the sampler
        nuts_kernel,mcmc = buildmcmc(model,initpars,settings)

        # if dry run, just return useful things
        if dryrun:
            return (nuts_kernel,mcmc,model,modelkw)

        mcmc.run(
            self.rng_key,
            **modelkw
            )

        if self.verbose:
            mcmc.print_summary()

        # write posterior samples to an astropy table
        posterior = mcmc.get_samples()
        outtable = Table(posterior)

        # write out the samples to a file
        outfile = indict['outfile']
        outtable.write(outfile,format='fits',overwrite=True)

        if self.verbose:
            print('[uberMS] ... writing samples to {}'.format(outfile))
            print('[uberMS] ... Finished: {0}'.format(datetime.now()-starttime))
        sys.stdout.flush()

        return (nuts_kernel,mcmc)
