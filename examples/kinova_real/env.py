
"""Thin alias mapping to kinova_real_env.RealEnv."""
from examples.kinova_real import real_env as _real_env

class KinovaRealEnvironment(_real_env.RealEnv):
    pass
