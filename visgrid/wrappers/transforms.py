from typing import Optional

import numpy as np

from .base import BaseObservationWrapper
from visgrid.wrappers.grayscale import GrayscaleWrapper
from visgrid.envs import GridworldEnv

class TransformWrapper(BaseObservationWrapper):
    def __init__(self, env, f, name: Optional[str] = None) -> None:
        super().__init__(env)
        self.f = f
        self.name = name if name is not None else 'TransformWrapper'

    def __str__(self):
        return f'<{self.name}>'

    def observation(self, observation):
        """Returns a modified observation."""
        return self.f(observation)

def wrap(env, f):
    return TransformWrapper(env, f, f.__qualname__.split('.')[0])

def NoiseWrapper(env, sigma=0.01, truncation=None):
    def fn(obs):
        noise = env.np_random.normal(0, sigma, obs.shape)
        if truncation is not None:
            noise = np.clip(noise, -truncation, truncation)
        return obs + noise

    return wrap(env, fn)

def ClipWrapper(env, low=0.0, high=1.0):
    return wrap(env, lambda obs: np.clip(obs, low, high))

def GaussianBlurWrapper(env, sigma=0.6, truncate=1.0):
    def fn(obs):
        import scipy.ndimage
        return scipy.ndimage.gaussian_filter(obs, sigma, truncate=truncate, mode='nearest')

    return wrap(env, fn)

def InvertWrapper(env, max_value=1):
    return wrap(env, lambda obs: max_value - obs)

def wrap_gridworld(env):
    assert isinstance(env.unwrapped, GridworldEnv)

    env.unwrapped.image_observations = True

    env = GrayscaleWrapper(env)
    env = GaussianBlurWrapper(env)
    env = NoiseWrapper(env, sigma=0.01)
    env = ClipWrapper(env)
    env = InvertWrapper(env)
    return env
