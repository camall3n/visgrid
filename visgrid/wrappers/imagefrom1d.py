import copy

import numpy as np
import gym

class ImageFrom1DWrapper(gym.ObservationWrapper):
    """Convert vector observations to image observations by adding two dummy dimensions
    """
    def __init__(self, env: gym.Env):
        """Convert vector observations to image observations by adding two dummy dimensions

        Args:
            env (Env): The environment to apply the wrapper
        """
        super().__init__(env)
        ndims = len(env.observation_space.shape)
        if ndims != 1:
            raise ValueError(f'Expected a 1D env, but env has {ndims} dims')

        obs_shape = self.observation_space.shape + (1, 1)
        self.observation_space = copy.deepcopy(self.observation_space)
        self.observation_space._shape = obs_shape

    def observation(self, observation):
        """Converts the vector observation to a H,W,C image observation
        """
        observation = np.expand_dims(observation, -1)
        observation = np.expand_dims(observation, -1)
        return observation
