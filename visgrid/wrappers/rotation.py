import gym
from scipy.stats import special_ortho_group

from .base import BaseObservationWrapper

class RotationWrapper(BaseObservationWrapper):
    """
    Applies a random (fixed) rotation matrix to all observations
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        n_dims = env.observation_space.shape[0]
        self.rotation_matrix = special_ortho_group.rvs(n_dims)

    def observation(self, obs):
        return self.rotation_matrix @ obs
