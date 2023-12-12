import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

import numpyro.distributions as distfn
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes,validate_sample

class Sigmoid_Prior(distfn.Distribution):
    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    reparametrized_params = ["low", "high"]

    def __init__(self,a=1.0, b=0.0, low=-10.0,high=10.0,validate_args=None):
        """
        Apply a Sigmoid-like prior
        Parameters
        ----------

        a : float, optional
            scale parameter. 
            Default is '1.0'.
        b : float, optional
            Location parameter.
            Default is `0.0`.
        """
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

        self.a = a
        self.b = b
        
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
       
    @validate_sample      
    def log_prob(self, x):
        """
            x :a `~numpy.ndarray` of shape (Ngrid)
            
        Returns
        -------
        lnprior : `~numpy.ndarray` of shape (Ngrid)
            The corresponding unnormalized ln(prior).
        """

        lnprior = jnp.log(
            1.0 / (1.0 + jnp.exp(-1.0*self.a*(x-self.b)))
            )

        return lnprior 

class DSigmoid_Prior(distfn.Distribution):
    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    reparametrized_params = ["low", "high"]

    def __init__(self,a=1.0, b=0.0, c=-1.0, d=0.0, low=-10.0,high=10.0,validate_args=None):
        """
        Apply a Sigmoid-like prior
        Parameters
        ----------

        a : float, optional
            scale parameter. 
            Default is '1.0'.
        b : float, optional
            Location parameter.
            Default is `0.0`.
        c : float, optional
            scale parameter. 
            Default is '-1.0'.
        d : float, optional
            Location parameter.
            Default is `0.0`.

        """
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
       
    @validate_sample      
    def log_prob(self, x):
        """
            x :a `~numpy.ndarray` of shape (Ngrid)
            
        Returns
        -------
        lnprior : `~numpy.ndarray` of shape (Ngrid)
            The corresponding unnormalized ln(prior).
        """

        prior1 = 1.0 / (1.0 + jnp.exp(-1.0*self.a*(x-self.b)))
        prior2 = 1.0 / (1.0 + jnp.exp(-1.0*self.c*(x-self.d)))

        lnprior = jnp.log(
            prior1*prior2
            )

        return lnprior 

class IMF_Prior(distfn.Distribution):
    # support = constraints.interval(0.25,3.0)

    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    reparametrized_params = ["low", "high"]


    def __init__(self,low=0.25, high=3.0, alpha_low=1.3, alpha_high=2.3, mass_break=0.5, validate_args=None):
        """
        Apply a Kroupa-like broken IMF prior over the provided initial mass grid.
        Parameters
        ----------

        alpha_low : float, optional
            Power-law slope for the low-mass component of the IMF.
            Default is `1.3`.
        alpha_high : float, optional
            Power-law slope for the high-mass component of the IMF.
            Default is `2.3`.
        mass_break : float, optional
            The mass where we transition from `alpha_low` to `alpha_high`.
            Default is `0.5`.
        """
        # self.mass = mass
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.interval(low, high)
        # super().__init__(batch_shape = (), event_shape=())
        super().__init__(batch_shape, validate_args=validate_args)
        
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.mass_break = mass_break

        # Compute normalization.
        norm_low = mass_break ** (1. - alpha_low) / (alpha_high - 1.)
        norm_high = 0.08 ** (1. - alpha_low) / (alpha_low - 1.)  # H-burning limit
        norm_high -= mass_break ** (1. - alpha_low) / (alpha_low - 1.)
        norm = norm_low + norm_high
        self.lognorm = jnp.log(norm)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
       
    @validate_sample      
    def log_prob(self, mass):
        """
        mgrid : `~numpy.ndarray` of shape (Ngrid)
            Grid of initial mass (solar units) the IMF will be evaluated over.
        Returns
        -------
        lnprior : `~numpy.ndarray` of shape (Ngrid)
            The corresponding unnormalized ln(prior).
        """

        def lnprior_high(mass):
            return (-self.alpha_high * jnp.log(mass) 
                + (self.alpha_high - self.alpha_low) * jnp.log(self.mass_break))
        def lnprior_low(mass):
            return -self.alpha_low * jnp.log(mass)

        lnprior = lax.cond(mass > self.mass_break,lnprior_high,lnprior_low,mass)

        return lnprior - self.lognorm

class Gal_Prior(distfn.Distribution):
    # support = constraints.interval(1.0,500000.0)

    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    reparametrized_params = ["low", "high"]

    def __init__(self,l,b,
                low=1.0, high=500000.0,
                R_solar=8.2, Z_solar=0.025,
                R_thin=2.6, Z_thin=0.3,
                R_thick=2.0, Z_thick=0.9, f_thick=0.04,
                Rs_halo=0.5, q_halo_ctr=0.2, q_halo_inf=0.8, r_q_halo=6.0,
                eta_halo=4.2, f_halo=0.005,
                feh_thin=-0.2, feh_thin_sigma=0.3,
                feh_thick=-0.7, feh_thick_sigma=0.4,
                feh_halo=-1.6, feh_halo_sigma=0.5,
                max_age=13.8, min_age=0., feh_age_ctr=-0.5, feh_age_scale=0.5,
                nsigma_from_max_age=2., max_sigma=4., min_sigma=1., validate_args=None):
        # super().__init__(batch_shape = (), event_shape=())
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

        self.l = l
        self.b = b

        lp = jnp.deg2rad(self.l)
        bp = jnp.deg2rad(self.b)

        self.Xp = jnp.cos(lp) * jnp.cos(bp)
        self.Yp = jnp.sin(lp) * jnp.cos(bp)
        self.Zp = jnp.sin(bp)

        self.sol_X = 8.3
        self.sol_Z = -27.0/1000.0

        self.R_solar    = R_solar   
        self.Z_solar    = Z_solar   
        self.R_thin     = R_thin    
        self.Z_thin     = Z_thin    
        self.R_thick    = R_thick   
        self.Z_thick    = Z_thick   
        self.f_thick    = f_thick   
        self.Rs_halo    = Rs_halo   
        self.q_halo_ctr = q_halo_ctr
        self.q_halo_inf = q_halo_inf
        self.r_q_halo   = r_q_halo  
        self.eta_halo   = eta_halo  
        self.f_halo     = f_halo    

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def logn_disk(self,R, Z, R_solar=8.2, Z_solar=0.025, R_scale=2.6, Z_scale=0.3):
        """
        Log-number density of stars in the disk component of the galaxy.
        Parameters
        ----------
        R : `~numpy.ndarray` of shape (N)
            The distance from the center of the galaxy.
        Z : `~numpy.ndarray` of shape (N)
            The height above the galactic midplane.
        R_solar : float, optional
            The solar distance from the center of the galaxy in kpc.
            Default is `8.2`.
        Z_solar : float, optional
            The solar height above the galactic midplane in kpc.
            Default is `0.025`.
        R_scale : float, optional
            The scale radius of the disk in kpc. Default is `2.6`.
        Z_scale : float, optional
            The scale height of the disk in kpc. Default is `0.3`.
        Returns
        -------
        logn : `~numpy.ndarray` of shape (N)
            The corresponding normalized ln(number density).
        """

        rterm = (R - R_solar) / R_scale  # radius term
        zterm = (jnp.abs(Z) - jnp.abs(Z_solar)) / Z_scale  # height term

        return -(rterm + zterm)

    def logn_halo(self,R, Z, R_solar=8.2, Z_solar=0.025, R_smooth=0.5,
                  eta=4.2, q_ctr=0.2, q_inf=0.8, r_q=6.):
        """
        Log-number density of stars in the halo component of the galaxy.
        Parameters
        ----------
        R : `~numpy.ndarray` of shape (N)
            The distance from the center of the galaxy.
        Z : `~numpy.ndarray` of shape (N)
            The height above the galactic midplane.
        R_solar : float, optional
            The solar distance from the center of the galaxy in kpc.
            Default is `8.2`.
        Z_solar : float, optional
            The solar height above the galactic midplane in kpc.
            Default is `0.025`.
        R_smooth : float, optional
            The smoothing radius in kpc used to avoid singularities
            around the Galactic center. Default is `0.5`.
        eta : float, optional
            The (negative) power law index describing the number density.
            Default is `4.2`.
        q_ctr : float, optional
            The nominal oblateness of the halo at Galactic center.
            Default is `0.2`.
        q_inf : float, optional
            The nominal oblateness of the halo infinitely far away.
            Default is `0.8`.
        r_q : float, optional
            The scale radius over which the oblateness changes in kpc.
            Default is `6.`.
        Returns
        -------
        logn : `~numpy.ndarray` of shape (N)
            The corresponding normalized ln(number density).
        """

        # Compute distance from Galactic center.
        r = jnp.sqrt(R**2 + Z**2)

        # Compute oblateness.
        rp = jnp.sqrt(r**2 + r_q**2)
        q = q_inf - (q_inf - q_ctr) * jnp.exp(1. - rp / r_q)

        # Compute effective radius.
        Reff = jnp.sqrt(R**2 + (Z / q)**2 + R_smooth**2)

        # Compute solar value for normalization.
        rp_solar = jnp.sqrt(R_solar**2 + Z_solar**2 + r_q**2)
        q_solar = q_inf - (q_inf - q_ctr) * jnp.exp(1. - rp_solar / r_q)
        Reff_solar = jnp.sqrt(R_solar**2 + (Z_solar / q_solar) + R_smooth**2)

        # Compute inner component.
        logn = -eta * jnp.log(Reff / Reff_solar)

        return logn

    @validate_sample      
    def log_prob(self,dist):
        dist = dist / 1000.0

        # Compute volume factor.
        vol_factor = 2. * jnp.log(dist + 1e-300)  # dV = r**2 factor

        X = dist * self.Xp - self.sol_X
        Y = dist * self.Yp

        Z = dist * self.Zp - self.sol_Z
        # R = np.sqrt((X**2.0) + (Y**2.0))
        R = jnp.hypot(X, Y)

        # Get thin disk component.
        logp_thin = self.logn_disk(R, Z, R_solar=self.R_solar, Z_solar=self.Z_solar,
                              R_scale=self.R_thin, Z_scale=self.Z_thin)
        logp_thin += vol_factor

        # Get thick disk component.
        logp_thick = self.logn_disk(R, Z, R_solar=self.R_solar, Z_solar=self.Z_solar,
                               R_scale=self.R_thick, Z_scale=self.Z_thick)
        logp_thick += vol_factor + jnp.log(self.f_thick)

        # Get halo component.
        logp_halo = self.logn_halo(R, Z, R_solar=self.R_solar, Z_solar=self.Z_solar,
                              R_smooth=self.Rs_halo, eta=self.eta_halo,
                              q_ctr=self.q_halo_ctr, q_inf=self.q_halo_inf, r_q=self.r_q_halo)
        logp_halo += vol_factor + jnp.log(self.f_halo)

        # Compute log-probability.
        lnprior = logsumexp(
            jnp.asarray([logp_thin, logp_thick, logp_halo]), axis=0)

        # if return_comp:
        #     # Collect components.
        #     components = {}
        #     components['number_density'] = [logp_thin, logp_thick, logp_halo]
        #     components['lnprior'] = ([
        #         logp_thin  - lnprior,
        #         logp_thick - lnprior,
        #         logp_halo  - lnprior,])
        #     return lnprior,components
        # else:
        return lnprior