try:
    from ._version import __version__
except(ImportError):
    pass

from jax.config import config
config.update('jax_enable_x64', True)

from . import spots
from . import dva
from . import utils
__abspath__ = '/Users/pcargile/Astro/gitrepos/uberMS/'
