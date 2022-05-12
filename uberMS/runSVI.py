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
        
        self.specNN = kwargs.get('specNN',None)
        self.contNN = kwargs.get('contNN',None)
        self.photNN = kwargs.get('photNN',None)
        self.mistNN = kwargs.get('mistNN',None)

        self.NNtype = kwargs.get('NNtype','LinNet')

        self.rng_key = jrandom.PRNGKey(0)

        self.defaultfilterarr = ([
            'GaiaEDR3_G',
            'GaiaEDR3_BP',
            'GaiaEDR3_RP',
            'PS_g',
            'PS_r',
            'PS_i',
            'PS_z',
            'PS_y',
            'SDSS_u',
            'SDSS_g',
            'SDSS_r',
            'SDSS_i',
            'SDSS_z',
            '2MASS_J',
            '2MASS_H',
            '2MASS_Ks',
            'WISE_W1',
            'WISE_W2',
            ])

        GM = GenMod()
        GM._initspecnn(
            nnpath=self.specNN,
            Cnnpath=self.contNN,
            NNtype=self.NNtype)
        GM._initphotnn(
            self.defaultfilterarr,
            nnpath=self.photNN)
        GMIST = GenMIST.modpred(
            nnpath=self.mistNN,
            nntype='LinNet',
            normed=True)

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
        data = indict['data']
        initpars = indict['initpars']
        prior = indict['prior']
        settings = indict['svi']



    sys.stdout.flush()
    starttime = datetime.now()
    print('... Working on {0} @ {1}...'.format(data['starinfo']['starname'],starttime))

    print('--------')
    print('PHOT:')
    print('--------')
    for ii,ff in enumerate(data['filtarr']):
        print('{0} = {1} +/- {2}'.format(ff,phot_in[ii],photerr_in[ii]))
    print('--------')
    print('SPEC:')
    print('--------')
    print('number of pixels: {0}'.format(len(specwave_in)))
    print('min/max wavelengths: {0} -- {1}'.format(specwave_in.min(),specwave_in.max()))
    print('median flux: {0}'.format(np.median(specflux_in)))
    print('median flux error: {0}'.format(np.median(speceflux_in)))
    print('SNR: {0}'.format(np.median(specflux_in/speceflux_in)))
    print('--------')
    sys.stdout.flush()

    initpars = ({
        'eep':400,
        'initial_Mass':1.00,
        # 'initial_[Fe/H]':0.0,
        # 'initial_[a/Fe]':0.0,
        'dist':1000.0/data['parallax'][0],
        # 'av':0.01,
        # 'vrad':0.0,
        'vmic':1.0,
        'vrot':5.0,
        'pc0':1.0,
        'pc1':0.0,
        'pc2':0.0,
        'pc3':0.0,
        'instr_scale':1.0,
        'photjitter':1E-5,
        'specjitter':1E-5,            
        })

    modelkw = ({
        'specobs':specflux_in,
        'specobserr':speceflux_in, 
        'specwave':specwave_in,
        'parallax':data['parallax'],
        'photobs':phot_in,
        'photobserr':photerr_in,
        'filtarr':data['filtarr'],
        'genspecfn':genspecfn,
        'genphotfn':genphotfn,
        'genMISTfn':genMISTfn,
        'MISTpars':GMIST.modpararr,
        'jMIST':Jac_genMISTfn,
        'lsf':data['lsf'],
        'RVest':data['RVest'],
        'SFD_Av':data['SFD_Av'],
        })

    # optimizer = numpyro.optim.Adam(0.1)
    # optimizer = numpyro.optim.Adagrad(1.0)
    # optimizer = numpyro.optim.Minimize()
    optimizer = numpyro.optim.ClippedAdam(0.001)
    # optimizer = numpyro.optim.SM3(0.1)    
    # guide = autoguide.AutoLaplaceApproximation(model,init_loc_fn=initialization.init_to_value(values=initpars))
    # guide = autoguide.AutoNormal(model,init_loc_fn=initialization.init_to_value(values=initpars))
    guide = autoguide.AutoLowRankMultivariateNormal(model,init_loc_fn=initialization.init_to_value(values=initpars))
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(
        rng_key, 
        30000,
        **modelkw
        )

    params = svi_result.params
    posterior = guide.sample_posterior(rng_key, params, (5000,))
    print_summary({k: v for k, v in posterior.items() if k != "mu"}, 0.89, False)

    t = Table(posterior)

    # determine extra parameter from MIST
    extrapars = [x for x in GMIST.modpararr if x not in t.keys()] 
    for kk in extrapars + ['Teff','Age']:
        t[kk] = np.nan * np.ones(len(t),dtype=float)

    for t_i in t:
        MISTpred = genMISTfn(
            eep=t_i['eep'],
            mass=t_i['initial_Mass'],
            feh=t_i['initial_[Fe/H]'],
            afe=t_i['initial_[a/Fe]'],
            verbose=False
            )
        MISTdict = ({
            kk:pp for kk,pp in zip(
            GMIST.modpararr,MISTpred)
            })

        for kk in extrapars:
            t_i[kk] = MISTdict[kk]

        t_i['Teff'] = 10.0**(t_i['log(Teff)'])
        t_i['Age']  = 10.0**(t_i['log(Age)']-9.0)

    for kk in ['Teff','log(g)','[Fe/H]','[a/Fe]','Age']:
        pars = quantile(t[kk],[0.5,0.16,0.84])
        print('{0} = {1:f} +{2:f}/-{3:f}'.format(kk,pars[0],pars[2]-pars[0],pars[0]-pars[1]))


    outfile = './output/samp_fibID_{FIBID}_gaiaID_{GAIAID}_plate_{PLATEID}_mjd_{MJD}_{VER}.fits'.format(
        FIBID=data['starinfo']['FIBERID'],
        GAIAID=data['starinfo']['GAIAEDR3_ID'],
        PLATEID=data['starinfo']['PLATE'],
        MJD=data['starinfo']['MJD'],
        VER=version)
    print('... writing samples to {}'.format(outfile))
    t.write(outfile,format='fits',overwrite=True)
    print('... Finished {0} @ {1}...'.format(data['starinfo']['starname'],datetime.now()-starttime))
    sys.stdout.flush()

    return (svi,guide,svi_result)

# def gMIST(pars):
#     eep,mass,feh,afe = pars
#     return genMISTfn(eep=eep,mass=mass,feh=feh,afe=afe)
# jgMIST = jit(gMIST)
# Jac_genMISTfn = jacfwd(jgMIST)
