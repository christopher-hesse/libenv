from pkg_resources import get_distribution, DistributionNotFound

from .libenv import CVecEnv, scalar_adapter

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

__all__ = ["CVecEnv", "scalar_adapter"]
