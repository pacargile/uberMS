class IMF_Prior(distfn.Distribution):
    support = constraints.interval(0.5,3.0)
    def __init__(self,alpha_low=1.3, alpha_high=2.3, mass_break=0.5):
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
        super().__init__(batch_shape = (), event_shape=())

        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.mass_break = mass_break

        # Compute normalization.
        norm_low = mass_break ** (1. - alpha_low) / (alpha_high - 1.)
        norm_high = 0.08 ** (1. - alpha_low) / (alpha_low - 1.)  # H-burning limit
        norm_high -= mass_break ** (1. - alpha_low) / (alpha_low - 1.)
        norm = norm_low + norm_high
        self.lognorm = jnp.log(norm)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
         
    def log_prob(self, mass):
        """
        mgrid : `~numpy.ndarray` of shape (Ngrid)
            Grid of initial mass (solar units) the IMF will be evaluated over.
        Returns
        -------
        lnprior : `~numpy.ndarray` of shape (Ngrid)
            The corresponding unnormalized ln(prior).
        """

        # # make sure mgrid is not a single float
        # if not isinstance(mgrid,Iterable):
        #     mgrid = jnp.array([mgrid])

        # # Initialize log-prior.
        # lnprior = jnp.zeros_like(mgrid) - jnp.inf

        # # Low mass.
        # low_mass = (mgrid <= mass_break) & (mgrid > 0.08)
        # lnprior[low_mass] = -alpha_low * jnp.log(mgrid[low_mass])

        # # High mass.
        # high_mass = mgrid > mass_break
        # lnprior[high_mass] = (-alpha_high * jnp.log(mgrid[high_mass])
        #                       + (alpha_high - alpha_low) * jnp.log(mass_break))

        # lnprior = self.lnprior_high(mass)

        def lnprior_high(mass):
            return (-self.alpha_high * jnp.log(mass) 
                + (self.alpha_high - self.alpha_low) * jnp.log(self.mass_break))
        def lnprior_low(mass):
            return -self.alpha_low * jnp.log(mass)

        lnprior = lax.cond(mass > self.mass_break,lnprior_high,lnprior_low,mass)

        return lnprior - self.lognorm
