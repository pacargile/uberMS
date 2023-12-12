import jax
import jax.numpy as jnp
from jax import jit, jacfwd #,lax
from jax import random as jrandom
import optax
from scipy import constants
speedoflight = constants.c / 1000.0

class SEDopt(object):
    def __init__(self, **kwargs):
        super(SEDopt, self).__init__()

        self.photobs    = kwargs.get('photobs',{})
        self.filtarray  = kwargs.get('filterarray',{})

        self.priors  = kwargs.get('priors',{})
        self.additionalinfo = kwargs.get('additionalinfo',{})

        # pull out fitting functions
        self.genphotfn = jax.jit(kwargs.get('genphotfn',None))

        self.returnsed = kwargs.get('returnsed',False)

        self.init_p0 = kwargs.get('initpars',
            {'Teff':6000.0,'log(g)':4.0,'[Fe/H]':0.0,'[a/Fe]':0.2,'log(R)':0.0,'dist':1000.0,'Av':0.0}
            )

        # figure out fixed parameters
        self.fixedpars = {}
        for kk in self.priors.keys():
            if self.priors[kk][0] == 'fixed':
                self.fixedpars[kk] = self.priors[kk][1]

        # all possible fit parameters
        fitpars_i = ['Teff','log(g)','[Fe/H]','[a/Fe]','log(R)','dist','Av']

        # fit parameters without fixedpars
        self.fitpars =  [x for x in fitpars_i if x not in self.fixedpars.keys()]
        
        if len(self.fitpars) + len(list(self.fixedpars.keys())) != len(fitpars_i):
            print('ALL PARS MUST BE EITHER FIT OR FIXED')
            print('FIT PARS:',self.fitpars)
            print('FIX PARS:',list(self.fixedpars.keys()))
            raise(IOError)
        
        self.verbose = kwargs.get('verbose',False)
        
    def __call__(self):
        p0 = jnp.array([self.init_p0[x] for x in self.fitpars])

        # init the optimizer        
        start_learning_rate = 1e-3
        optimizer = optax.adam(start_learning_rate)

        # set initial pars
        opt_state = optimizer.init(p0)
        
        # define step func
        @jax.jit
        def step(params,opt_state):
            loss_value,grads = jax.value_and_grad(self.chisq_sed)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params,opt_state,loss_value            
        
        for ii in range(10000):
            if ii == 0:
                params = p0
            params,opt_state,loss_value = step(params,opt_state)
            if self.verbose:
                if ii % 100 == 0:
                    print(f'step - {ii}, loss - {loss_value}')

        params_dict = {x:y for x,y in zip(self.fitpars,params)}
        return (params_dict,loss_value)

    # define the model
    def chisq_sed(self,pars):
        fitpars_dict = {}
        for ii,x in enumerate(self.fitpars):
            fitpars_dict[x] = pars[ii]

        parsdict = dict(**fitpars_dict,**self.fixedpars)

        # make photometry prediction
        photpars = ([
            parsdict['Teff'],parsdict['log(g)'],parsdict['[Fe/H]'],parsdict['[a/Fe]'],
            parsdict['log(R)'],parsdict['dist'],parsdict['Av'],3.1])
        photmod = self.genphotfn(photpars)

        chisq = jnp.sum(
            jnp.array([((photmod[kk]-self.photobs[kk][0])**2.0)/(self.photobs[kk][1]**2.0) 
            for kk in self.filtarray])
            )
        
        return chisq 

class RVopt(object):
    def __init__(self, **kwargs):
        super(RVopt, self).__init__()

        self.wave = jnp.asarray(kwargs.get('inwave',[]))
        self.flux = jnp.asarray(kwargs.get('influx',[]))
        self.eflux = jnp.asarray(kwargs.get('ineflux',[]))
        self.modflux = jnp.asarray(kwargs.get('modflux',[]))
        self.modwave = jnp.asarray(kwargs.get('modwave',[]))

        self.init_p0 = kwargs.get('initvel',jnp.arange(-500,500,25))
        
        self.verbose = kwargs.get('verbose',False)
        
    def __call__(self):
        if type(self.init_p0) is float:
            p0 = [jnp.array([self.init_p0])]
        else:
            p0 = [jnp.array([float(x)]) for x in self.init_p0]
        

        # init the optimizer        
        start_learning_rate = 1e-1
        optimizer = optax.adam(start_learning_rate)

        outval = []
        outlos = []
        for p0_i in p0:
            # set initial pars
            opt_state = optimizer.init(p0_i)
            
            # define step func
            @jax.jit
            def step(params,opt_state):
                loss_value,grads = jax.value_and_grad(self.chisq_rv)(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params,opt_state,loss_value            
            
            for ii in range(1000):
                if ii == 0:
                    params = p0_i
                params,opt_state,loss_value = step(params,opt_state)
                if self.verbose:
                    if ii % 100 == 0:
                        print(f'step - {ii}, loss - {loss_value}, p0_i - {p0_i[0]}, p0_f - {params[0]}')

            outval.append(params)
            outlos.append(loss_value)
        outloss_ind = jnp.argmin(jnp.array(outlos))
        return (outval[outloss_ind],outlos[outloss_ind])

    def chisq_rv(self,pars):
        
        rv = pars[0]
        
        wave = self.wave
        flux = self.flux
        eflux = self.eflux
        modflux = self.modflux
        modwave = self.modwave

        # adjust model to new rv
        modwave_i = modwave*(1.0+(rv/speedoflight))

        # interpolate it back to wave so that chi-sq can be computed
        modflux_i = jnp.interp(wave,modwave_i,modflux,left=jnp.nan,right=jnp.nan)

        chisq = jnp.nansum( ((modflux_i-flux)**2.0) / (eflux**2.0) ) / jnp.isfinite(modflux_i).sum()

        return chisq