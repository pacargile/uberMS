import numpyro
from numpyro.infer import SVI, autoguide, initialization, Trace_ELBO, RenyiELBO
from numpyro.diagnostics import print_summary

import jax
from jax import jit, jacfwd #,lax
from jax import random as jrandom
import jax.numpy as jnp

from optax import exponential_decay

from misty.predict import GenModJax as GenMIST
from Payne.jax.genmod import GenMod

from datetime import datetime
import sys,os
from astropy.table import Table

os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"

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

        # if user did not use the new specNN and photNN
        if self.sNNtype is None:
            self.sNNtype = self.NNtype
        if self.pNNtype is None:
            self.pNNtype = self.NNtype

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
        else:
            self.specNN_labels = []
            
        if self.photNN is not None:
            GM._initphotnn(
                None,
                nnpath=self.photNN,
                NNtype=self.pNNtype)
            
        if self.mistNN is not None:
            GMIST = GenMIST.modpred(
                nnpath=self.mistNN,
                nntype='LinNet',
                normed=True,
                applyspot=self.applyspot)
            self.MISTpars = GMIST.modpararr
        else:
            print('DID NOT READ IN MIST NN, DO YOU WANT TO RUN THE PAYNE?')
            raise IOError


        # jit a couple of functions
        self.genspecfn = jit(GM.genspec)
        self.genphotfn = jit(GM.genphot)
        self.genMISTfn = jit(GMIST.getMIST)

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
            print('--------')
            print('MODELS:')
            print('--------')
            print('Spec NN: {}'.format(self.specNN))
            print('Cont NN: {}'.format(self.contNN))
            print('NN-type: {}'.format(self.NNtype))
            print('Phot NN: {}'.format(self.photNN))
            print('MIST NN: {}'.format(self.mistNN))


    def run(self,indict,dryrun=False):

        starttime = datetime.now()

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
        # pass info about if vmic is included in NN labels
        if 'vturb' in self.specNN_labels:
            modelkw['additionalinfo']['vmicbool'] = True
        else:
            modelkw['additionalinfo']['vmicbool'] = False

        # check to see if user wants to turn off difusion
        if 'diffbool' in indict.keys():
            modelkw['additionalinfo']['diffbool'] = indict['diffbool']
        else:
            modelkw['additionalinfo']['diffbool'] = True
        
        # define the optimizer
        # optimizer = numpyro.optim.ClippedAdam(settings.get('opt_tol',0.001))
        optimizer = numpyro.optim.ClippedAdam(exponential_decay(settings.get('start_tol',1E-3),3000,0.5, end_value=settings.get('opt_tol',1E-5)))

        # check if initial values are valid
        initpars_test = numpyro.infer.util.find_valid_initial_params(
            self.rng_key,
            model,
            init_strategy=initialization.init_to_value(values=initpars),
            model_kwargs=modelkw,
        )

        # This is not good, init par outside of prior
        # figure out which one and print to stdout
        if initpars_test[1] == False:
            inpdict = initpars_test[0][0]
            inpdict_grad = initpars_test[0][1]
            
            print(inpdict)
            print(inpdict_grad)

            for kk in inpdict.keys():
                if jnp.isnan(inpdict[kk]):
                    raise IOError(f"Found following parameter outside prior volume: {kk}")
                # if jnp.isnan(inpdict_grad[kk]):
                #     raise IOError(f"Found following parameter has a NaN grad: {kk}")

        # define the guide
        guide_str = settings.get('guide','Normalizing Flow')
        # define the guide
        if guide_str == 'Normal':
            guide = autoguide.AutoLowRankMultivariateNormal(
                model,init_loc_fn=initialization.init_to_value(values=initpars))
        else:
            guide = autoguide.AutoBNAFNormal(
                model,num_flows=settings.get('numflows',2),
                init_loc_fn=initialization.init_to_value(values=initpars))

        loss = RenyiELBO(alpha=1.25)

        # build SVI object
        svi = SVI(model, guide, optimizer, loss=loss)
        
        # if dry run, just return useful things
        if dryrun:
            return (svi,model,guide,modelkw)
        
        # run the SVI
        svi_result = svi.run(
            self.rng_key, 
            settings.get('steps',30000),
            **modelkw,
            progress_bar=settings.get('progress_bar',True),
            )

        # reconstruct the posterior
        params = svi.get_params(svi_result.state)
        posterior = guide.sample_posterior(
            self.rng_key, 
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
            print('... writing samples to {}'.format(outfile))
            print('... Finished: {0}'.format(datetime.now()-starttime))
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
        else:
            self.specNN_labels = []
            
        if self.photNN is not None:
            GM._initphotnn(
                None,
                nnpath=self.photNN,
                NNtype=self.pNNtype)

        # jit a couple of functions
        self.genspecfn = jit(GM.genspec)
        self.genphotfn = jit(GM.genphot)

        self.verbose = kwargs.get('verbose',True)

        if self.verbose:
            print('--------')
            print('MODELS:')
            print('--------')
            print('Spec NN: {}'.format(self.specNN))
            print('Cont NN: {}'.format(self.contNN))
            print('Phot NN: {}'.format(self.photNN))
            print('NN-type: {}'.format(self.NNtype))

    def run(self,indict,dryrun=False):

        starttime = datetime.now()

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
        # pass info about if vmic is included in NN labels
        if 'vturb' in self.specNN_labels:
            modelkw['additionalinfo']['vmicbool'] = True
        else:
            modelkw['additionalinfo']['vmicbool'] = False

        # define the optimizer
        # optimizer = numpyro.optim.ClippedAdam(settings.get('opt_tol',1E-4))
        optimizer = numpyro.optim.ClippedAdam(exponential_decay(settings.get('start_tol',1E-3),3000,0.5, end_value=settings.get('opt_tol',1E-5)))

        # check if initial values are valid
        initpars_test = numpyro.infer.util.find_valid_initial_params(
            self.rng_key,
            model,
            init_strategy=initialization.init_to_value(values=initpars),
            model_kwargs=modelkw,
        )

        # # This is not good, init par outside of prior
        # # figure out which one and print to stdout
        # if initpars_test[1] == False:
        #     inpdict = initpars_test[0][0]
        #     for kk in inpdict.keys():
        #         if jnp.isnan(inpdict[kk]):
        #             raise IOError(f"Found following parameter outside prior volume: {kk}")
        
        guide_str = settings.get('guide','Normaling Flow')
        # define the guide
        if guide_str == 'Normal':
            if self.verbose:
                print('... Using N-D Normal as Guide')
            guide = autoguide.AutoLowRankMultivariateNormal(
                model,
                init_loc_fn=initialization.init_to_value(values=initpars))
        else:
            if self.verbose:
                print('... Using NF as Guide')
            guide = autoguide.AutoBNAFNormal(
                model,num_flows=settings.get('numflows',2),
                init_loc_fn=initialization.init_to_value(values=initpars))

        # loss = Trace_ELBO()
        loss = RenyiELBO()
        
        # build SVI object
        svi = SVI(model, guide, optimizer, loss=loss)

        # if dry run, just return useful things
        if dryrun:
            return (svi,model,guide,modelkw)

        # run the SVI
        svi_result = svi.run(
            self.rng_key, 
            settings.get('steps',30000),
            **modelkw,
            progress_bar=settings.get('progress_bar',True),
            )

        # reconstruct the posterior
        params = svi.get_params(svi_result.state)
        posterior = guide.sample_posterior(
            self.rng_key, 
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
            print('... writing samples to {}'.format(outfile))
            print('... Finished: {0}'.format(datetime.now()-starttime))
        sys.stdout.flush()

        return (svi,guide,svi_result)
        