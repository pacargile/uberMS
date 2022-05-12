import numpyro
from numpyro.infer import SVI, autoguide, initialization
from numpyro.diagnostics import print_summary

from jax import jit,lax,jacfwd
from jax import random as jrandom
import jax.numpy as jnp

from misty.predict import GenModJax as GenMIST
from Payne.jax.genmod import GenMod

class sviMS(object):
    """docstring for sviMS"""
    def __init__(self, *arg, **kwargs):
        super(sviMS, self).__init__()
        
        # set paths for NN's
        self.specNN = kwargs.get('specNN',None)
        self.contNN = kwargs.get('contNN',None)
        self.photNN = kwargs.get('photNN',None)
        self.mistNN = kwargs.get('mistNN',None)

        # set type of NN
        self.NNtype = kwargs.get('NNtype','LinNet')

        self.rng_key = jrandom.PRNGKey(0)

        # initialize prediction classes
        GM = GenMod()
        GM._initspecnn(
            nnpath=self.specNN,
            Cnnpath=self.contNN,
            NNtype=self.NNtype)
        GM._initphotnn(
            None,
            nnpath=self.photNN)
        GMIST = GenMIST.modpred(
            nnpath=self.mistNN,
            nntype='LinNet',
            normed=True)
        self.MISTpars = GMIST.modpararr

        # jit a couple of functions
        self.genspecfn = jit(GM.genspec)
        self.genphotfn = jit(GM.genphot)
        self.genMISTfn = jit(GMIST.getMIST)

        self.verbose = kwargs.get('verbose',True)

        if self.verbose:
            print('--------')
            print('MODELS:')
            print('--------')
            print('Spec NN: {}'.format(specNN))
            print('Cont NN: {}'.format(contNN))
            print('NN-type: {}'.format(NNtype))
            print('Phot NN: {}'.format(photNN))
            print('MIST NN: {}'.format(mistNN))


    def run(self,indict):

        # break out parts on input dict
        data = indict['data']
        initpars = indict['initpars']
        prior = indict['prior']
        settings = indict['svi']

        # determine if spectrum is input
        if 'spec' in data.keys():
            specwave_in,specflux_in,speceflux_in = data['spec']
            specwave_in  = jnp.asarray(specwave_in,dtype=float)
            specflux_in  = jnp.asarray(specflux_in,dtype=float)
            speceflux_in = jnp.asarray(speceflux_in,dtype=float)
            specbool = True
        else:
            specwave_in  = None
            specflux_in  = None
            speceflux_in = None
            specbool = False

        # determine if photometry is input
        if 'phot' in data.keys():
            phot_in    = jnp.asarray([data['phot'][xx][0] for xx in data['filtarr']],dtype=float)
            photerr_in = jnp.asarray([data['phot'][xx][1] for xx in data['filtarr']],dtype=float)
            photbool = True
        else:
            phot_in    = None
            photerr_in = None
            photbool = False

        # determine which model to use based on 
        if specbool and photbool:
            from .models import model_specphot as model
        if specbool and not photbool:
            from .models import model_spec as model
        if photbool and not specbool:
            from .models import model_phot as model

        # define input dictionary for models
        modelkw = ({
            'indata':{
                'specobs':specflux_in,
                'specobserr':speceflux_in, 
                'specwave':specwave_in,
                'photobs':phot_in,
                'photobserr':photerr_in,
                },
            'fitfunc':{
                'genspecfn':self.genspecfn,
                'genphotfn':self.genphotfn,
                'genMISTfn':self.genMISTfn,
                'MISTpars': self.MISTpars,
                },
            'priors':{
                },
            'additionalinfo':{
                }
            })

        # cycle through possible additional parameters
        if 'parallax' in data.keys():
            modelkw['additionalinfo']['parallax'] = data['parallax']
        if 'filtarr' in data.keys():
            modelkw['additionalinfo']['filtarr'] = data['filtarr']
        if 'lsf' in data.keys():
            modelkw['additionalinfo']['lsf'] = data['lsf']
        if 'instr' in data.keys():
            modelkw['additionalinfo']['instr'] = data['instr']
        if 'RVest' in data.keys():
            modelkw['additionalinfo']['RVest'] = data['RVest']
        if 'SFD_Av' in data.keys():
            modelkw['additionalinfo']['SFD_Av'] = data['SFD_Av']

        # define the optimizer
        optimizer = numpyro.optim.ClippedAdam(0.001)

        # define the guide
        # guide = autoguide.AutoLaplaceApproximation(model,init_loc_fn=initialization.init_to_value(values=initpars))
        # guide = autoguide.AutoNormal(model,init_loc_fn=initialization.init_to_value(values=initpars))
        guide = autoguide.AutoLowRankMultivariateNormal(model,init_loc_fn=initialization.init_to_value(values=initpars))

        # build SVI object
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        # run the SVI
        svi_result = svi.run(
            self.rng_key, 
            settings.get('steps',30000),
            **modelkw
            )

        # reconstruct the posterior
        params = svi_result.params
        posterior = guide.sample_posterior(rng_key, params, (5000,))
        if self.verbose:
            print_summary({k: v for k, v in posterior.items() if k != "mu"}, 0.89, False)

        # write posterior samples to an astropy table
        outtable = Table(posterior)

        # determine extra parameter from MIST
        extrapars = [x for x in self.MISTpars if x not in t.keys()] 
        for kk in extrapars + ['Teff','Age']:
            t[kk] = np.nan * np.ones(len(t),dtype=float)

        for t_i in t:
            MISTpred = self.genMISTfn(
                eep=t_i['eep'],
                mass=t_i['initial_Mass'],
                feh=t_i['initial_[Fe/H]'],
                afe=t_i['initial_[a/Fe]'],
                verbose=False
                )
            MISTdict = ({
                kk:pp for kk,pp in zip(
                self.MISTpars,MISTpred)
                })

            for kk in extrapars:
                t_i[kk] = MISTdict[kk]

            t_i['Teff'] = 10.0**(t_i['log(Teff)'])
            t_i['Age']  = 10.0**(t_i['log(Age)']-9.0)

        if self.verbose:
            for kk in ['Teff','log(g)','[Fe/H]','[a/Fe]','Age']:
                pars = [np.median(t[kk]),np.std(t[kk])]
                print('{0} = {1:f} +/-{2:f}'.format(kk,pars[0],pars[1]))

        # write out the samples to a file
        outfile = indict['outfile']
        t.write(outfile,format='fits',overwrite=True)

        if self.verbose:
            print('... writing samples to {}'.format(outfile))
            print('... Finished: {0}'.format(datetime.now()-starttime))
        sys.stdout.flush()

        return (svi,guide,svi_result)

# def gMIST(pars):
#     eep,mass,feh,afe = pars
#     return genMISTfn(eep=eep,mass=mass,feh=feh,afe=afe)
# jgMIST = jit(gMIST)
# Jac_genMISTfn = jacfwd(jgMIST)
