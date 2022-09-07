import gym
from gym import spaces
import numpy as np

from .base import BaseObservationWrapper

class PermuteFactorsWrapper(BaseObservationWrapper):
    """
    Randomly permutes the list of values for each factor

    This transformation destroys distances, but preserves "sameness" of factor values
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        ob_space = env.observation_space
        if isinstance(ob_space, spaces.Discrete):
            self.factor_sizes = (ob_space.n, )
        elif isinstance(ob_space, spaces.MultiDiscrete):
            self.factor_sizes = ob_space.nvec
        else:
            raise TypeError('Observation space must be Discrete or MultiDiscrete')
        self.n_factors = len(self.factor_sizes)
        self._generate_permutations()

    def _generate_permutations(self):
        self.permutations = [self.np_random.permutation(fac_sz) for fac_sz in self.factor_sizes]

    def observation(self, obs):
        return np.array([p[obs[i]] for i, p in enumerate(self.permutations)])

class PermuteStatesWrapper(BaseObservationWrapper):
    """
    Randomly permutes the list of states in the factor-product space

    This transformation destroys both distances and "sameness" of factor values,
    and only preserves "sameness" of states.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        ob_space = env.observation_space
        if isinstance(ob_space, spaces.Discrete):
            self.factor_sizes = (ob_space.n, )
        elif isinstance(ob_space, spaces.MultiDiscrete):
            self.factor_sizes = ob_space.nvec
        elif isinstance(ob_space, spaces.MultiBinary):
            self.factor_sizes = (2, ) * ob_space.n
        else:
            raise TypeError('Observation space must be Discrete, MultiDiscrete, or MultiBinary')

    def observation(self, obs):
        raise NotImplementedError  #TODO
