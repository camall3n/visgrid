from typing import Optional

import gym
import numpy as np

from visgrid.wrappers.grayscale import GrayscaleWrapper
from visgrid.envs import GridworldEnv

class TransformWrapper(gym.ObservationWrapper):
    def __init__(self, env, f) -> None:
        super().__init__(env)
        self.f = f

    def observation(self, observation):
        """Returns a modified observation."""
        return self.f(observation)

class NoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, sigma=0.01, truncation=None):
        super().__init__(env)
        self.sigma = sigma
        self.truncation = truncation

    def observation(self, obs):
        noise = self.env.np_random.normal(0, self.sigma, obs.shape)
        if self.truncation is not None:
            noise = np.clip(noise, -self.truncation, self.truncation)
        return obs + noise

class ClipWrapper(gym.ObservationWrapper):
    def __init__(self, env, low=0.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def observation(self, obs):
        return np.clip(obs, self.low, self.high)

class GaussianBlurWrapper(gym.ObservationWrapper):
    def __init__(self, env, sigma=0.6, truncate=1.0):
        super().__init__(env)
        self.sigma = sigma
        self.truncate = truncate

    def observation(self, obs):
        import scipy.ndimage
        return scipy.ndimage.gaussian_filter(obs,
                                             self.sigma,
                                             truncate=self.truncate,
                                             mode='nearest')

class InvertWrapper(gym.ObservationWrapper):
    def __init__(self, env, max_value=1):
        super().__init__(env)
        self.max_value = max_value

    def observation(self, obs):
        return self.max_value - obs

def wrap_gridworld(env):
    assert isinstance(env.unwrapped, GridworldEnv)
    env.unwrapped.set_rendering(enabled=True)

    env = GrayscaleWrapper(env)
    env = GaussianBlurWrapper(env)
    env = NoiseWrapper(env, sigma=0.01)
    env = ClipWrapper(env)
    env = InvertWrapper(env)
    return env
