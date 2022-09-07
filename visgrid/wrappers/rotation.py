import gym
import numpy as np
from scipy.stats import special_ortho_group

from .base import BaseObservationWrapper

class RotationWrapper(BaseObservationWrapper):
    """
    Applies a random (fixed) rotation matrix to all observations
    """
    def __init__(self, env: gym.Env, axes=None):
        super().__init__(env)
        if axes is None:
            n_dims = env.observation_space.shape[0]
            self.axes = np.arange(n_dims)
        else:
            self.axes = np.asarray(axes)
            n_dims = len(self.axes)

        self.rotation_matrix = special_ortho_group.rvs(n_dims, random_state=self.np_random)

    def observation(self, obs):
        factors = obs[self.axes]
        r_factors = self.rotation_matrix @ factors
        obs[self.axes] = r_factors
        return obs
