"""
C API for reinforcement learning environments

Environment libraries are normal C shared libraries, providing
the interface described here.  Each library must implement all
functions.
"""

from .libenv import CVecEnv, scalar_adapter

__all__ = ["CVecEnv", "scalar_adapter"]
