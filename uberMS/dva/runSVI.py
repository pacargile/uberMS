import numpyro
from numpyro.util import enable_x64
enable_x64()
from numpyro.infer import SVI, autoguide, initialization, Trace_ELBO, RenyiELBO
from numpyro.diagnostics import print_summary

from jax import jit, jacfwd #,lax
from jax import random as jrandom
import jax.numpy as jnp

from optax import exponential_decay

from misty.predict import GenModJax as GenMIST
from Payne.jax.genmod import GenMod

from datetime import datetime
import sys,os
from astropy.table import Table

try:
    os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
except:
    pass

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
        self.gradbool = kwargs.get('usegrad',False)

        # set type of NN
        self.NNtype = kwargs.get('NNtype','LinNet')

        self.rng_key = jrandom.PRNGKey(0)

        if isinstance(self.specNN,str):
            print('User input only one spectrum, please use standard uberMS (not dva)')
            print(self.specNN)
            raise IOError

        # determine how many spec user inputed
        if self.specNN is not None:
            self.nspec = len(self.specNN)
        else:
            self.nspec = 0            

        if self.nspec == 1:
            print('User input only one spectrum, please use standard uberMS (not dva)')
            raise IOError

        self.genspecfn = []
        self.vmic_bool = []        
        for ii in range(self.nspec):
            # initialize prediction classes
            GM = GenMod()
            
            if self.contNN is not None:
                contNN_i = self.contNN[ii]
            else:
                contNN_i = None
            
            GM._initspecnn(
                nnpath=self.specNN[ii],
                Cnnpath=contNN_i,
                NNtype=self.NNtype)
            genspecfn_i = jit(GM.genspec)
            self.genspecfn.append(genspecfn_i)
            specNN_labels = GM.PP.modpars
            if 'vturb' in specNN_labels:
                self.vmic_bool.append(True)
            else:
                self.vmic_bool.append(False)

        if self.photNN is not None:
            GM._initphotnn(
                None,
                nnpath=self.photNN)
            self.genphotfn = jit(GM.genphot)
        else:
            self.genphotfn = None

        if self.mistNN is not None:
            GMIST = GenMIST.modpred(
                nnpath=self.mistNN,
                nntype='LinNet',
                normed=True)
            self.genMISTfn = jit(GMIST.getMIST)
            self.MISTpars = GMIST.modpararr
        else:
            print('DID NOT READ IN MIST NN, DO YOU WANT TO RUN THE PAYNE?')
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
            print('--------')
            print('MODELS:')
            print('--------')
            print('Found {0} specNN models'.format(self.nspec))
            for ii in range(self.nspec):
                print('Spec NN {0}: {1}'.format(ii,self.specNN[ii]))
                if self.contNN is None:
                    print('Cont NN: {}'.format(self.contNN))
                else:
                    print('Cont NN {0}: {1}'.format(ii,self.contNN[ii]))
            print('Phot NN: {}'.format(self.photNN))
            print('NN-type: {}'.format(self.NNtype))
            print('MIST NN: {}'.format(self.mistNN))

    def run(self,indict):

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
            specwave_in  = []
            specflux_in  = []
            speceflux_in = []
            for ii in range(self.nspec):
                if isinstance(data['spec'][ii],dict):
                    specwave_i  = data['spec'][ii]['obs_wave']
                    specflux_i  = data['spec'][ii]['obs_flux']
                    speceflux_i = data['spec'][ii]['obs_eflux']
                else:               
                    specwave_i,specflux_i,speceflux_i = data['spec'][ii]
                specwave_in.append(jnp.asarray(specwave_i,dtype=float))
                specflux_in.append(jnp.asarray(specflux_i,dtype=float))
                speceflux_in.append(jnp.asarray(speceflux_i,dtype=float))
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
        if any(self.vmic_bool):
            modelkw['additionalinfo']['vmicbool'] = True
        else:
            modelkw['additionalinfo']['vmicbool'] = False

        # define the optimizer
        # optimizer = numpyro.optim.ClippedAdam(settings.get('opt_tol',0.001))
        optimizer = numpyro.optim.ClippedAdam(exponential_decay(settings.get('start_tol',1E-3),3000,0.5, end_value=settings.get('opt_tol',1E-5)))

        # define the guide
        guide_str = settings.get('guide','Normalizing Flow')
        # define the guide
        if guide_str == 'Normal':
            print('--- Using Normal Guide ---')
            guide = autoguide.AutoLowRankMultivariateNormal(
                model,init_loc_fn=initialization.init_to_value(values=initpars))
        else:
            guide = autoguide.AutoBNAFNormal(
                model,num_flows=settings.get('numflows',2),
                init_loc_fn=initialization.init_to_value(values=initpars))

        loss = RenyiELBO(alpha=1.25)

        # build SVI object
        svi = SVI(model, guide, optimizer, loss=loss)
        
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

        # set type of NN
        self.NNtype = kwargs.get('NNtype','LinNet')

        self.rng_key = jrandom.PRNGKey(0)

        if isinstance(self.specNN,str):
            print('User input only one spectrum, please use standard uberMS (not dva)')
            return IOError

        # determine how many spec user inputed
        if self.specNN is not None:
            self.nspec = len(self.specNN)
        else:
            self.nspec = 0            

        if self.nspec == 1:
            print('User input only one spectrum, please use standard uberMS (not dva)')
            return IOError
        
        self.genspecfn = []
        self.vmic_bool = []        
        for ii in range(self.nspec):
            # initialize prediction classes
            GM = GenMod()
            
            if self.contNN is not None:
                contNN_i = self.contNN[ii]
            else:
                contNN_i = None
            
            GM._initspecnn(
                nnpath=self.specNN[ii],
                Cnnpath=contNN_i,
                NNtype=self.NNtype)
            genspecfn_i = jit(GM.genspec)
            self.genspecfn.append(genspecfn_i)
            specNN_labels = GM.PP.modpars
            if 'vturb' in specNN_labels:
                self.vmic_bool.append(True)
            else:
                self.vmic_bool.append(False)

        if self.photNN is not None:
            GM._initphotnn(
                None,
                nnpath=self.photNN)
            self.genphotfn = jit(GM.genphot)
        else:
            self.genphotfn = None

        self.verbose = kwargs.get('verbose',True)

        if self.verbose:
            print('--------')
            print('MODELS:')
            print('--------')
            print('Found {0} specNN models'.format(self.nspec))
            for ii in range(self.nspec):
                print('Spec NN {0}: {1}'.format(ii,self.specNN[ii]))
                if self.contNN is None:
                    print('Cont NN: {}'.format(self.contNN))
                else:
                    print('Cont NN {0}: {1}'.format(ii,self.contNN[ii]))
            print('Phot NN: {}'.format(self.photNN))
            print('NN-type: {}'.format(self.NNtype))

    def run(self,indict):

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
            specwave_in  = []
            specflux_in  = []
            speceflux_in = []
            for ii in range(self.nspec):
                if isinstance(data['spec'][ii],dict):
                    specwave_i  = data['spec'][ii]['obs_wave']
                    specflux_i  = data['spec'][ii]['obs_flux']
                    speceflux_i = data['spec'][ii]['obs_eflux']
                else:               
                    specwave_i,specflux_i,speceflux_i = data['spec'][ii]
                specwave_in.append(jnp.asarray(specwave_i,dtype=float))
                specflux_in.append(jnp.asarray(specflux_i,dtype=float))
                speceflux_in.append(jnp.asarray(speceflux_i,dtype=float))
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
        if any(self.vmic_bool):
            modelkw['additionalinfo']['vmicbool'] = True
        else:
            modelkw['additionalinfo']['vmicbool'] = False

        # define the optimizer
        # optimizer = numpyro.optim.ClippedAdam(settings.get('opt_tol',1E-4))
        optimizer = numpyro.optim.ClippedAdam(exponential_decay(settings.get('start_tol',1E-3),3000,0.5, end_value=settings.get('opt_tol',1E-5)))
        
        guide_str = settings.get('guide','Normalizing Flow')
        # define the guide
        if guide_str == 'Normal':
            print('--- Using Normal Guide ---')
            guide = autoguide.AutoLowRankMultivariateNormal(
                model,init_loc_fn=initialization.init_to_value(values=initpars))
        else:
            guide = autoguide.AutoBNAFNormal(
                model,num_flows=2,
                init_loc_fn=initialization.init_to_value(values=initpars))

        # loss = Trace_ELBO()
        loss = RenyiELBO()
        
        # build SVI object
        svi = SVI(model, guide, optimizer, loss=loss)

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
        