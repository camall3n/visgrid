import gym
from gym import spaces
import numpy as np

from .base import BaseObservationWrapper

class FactorPermutationWrapper(BaseObservationWrapper):
    """
    Randomly permutes the list of values for each factor

    This transformation destroys distances, but preserves "sameness" of factor values
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        ob_space = env.observation_space
        if isinstance(ob_space, spaces.Discrete):
            self._factor_sizes = (ob_space.n, )
        elif isinstance(ob_space, spaces.MultiDiscrete):
            self._factor_sizes = ob_space.nvec
        else:
            raise TypeError('Observation space must be Discrete or MultiDiscrete')
        self.n_factors = len(self._factor_sizes)
        self._generate_permutations()

    def _generate_permutations(self):
        self.permutations = [self.np_random.permutation(fac_sz) for fac_sz in self._factor_sizes]

    def observation(self, obs):
        return np.array([p[obs[i]] for i, p in enumerate(self.permutations)])

class ObservationPermutationWrapper(BaseObservationWrapper):
    """
    Randomly permutes the list of states in the factor-product space

    This transformation destroys both distances and "sameness" of factor values,
    and only preserves "sameness" of states.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        ob_space = env.observation_space
        if isinstance(ob_space, spaces.Discrete):
            self._factor_sizes = (ob_space.n, )
        elif isinstance(ob_space, spaces.MultiDiscrete):
            self._factor_sizes = ob_space.nvec
        elif isinstance(ob_space, spaces.MultiBinary):
            self._factor_sizes = (2, ) * ob_space.n
        else:
            raise TypeError('Observation space must be Discrete, MultiDiscrete, or MultiBinary')

        n_observations = int(np.prod(self._factor_sizes))
        self.permutation = self.np_random.permutation(n_observations)

    def observation(self, obs):
        idx = self.pos_to_idx(obs)
        p_idx = self.permutation[idx]
        p_obs = self.idx_to_pos(p_idx)
        return p_obs

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Coordinate Transforms - any dim array, only last axis counts!
    #
    # Adapted from https://github.com/nmichlo/disent
    #
    #
    # MIT License
    #
    # Copyright (c) 2021 Nathan Juraj Michlo
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

    def pos_to_idx(self, positions) -> np.ndarray:
        """
        Convert a position to an index (or convert a list of positions to a list of indices)
        - positions are lists of integers, with each element < their corresponding factor size
        - indices are integers < size

        TODO: can factor_multipliers be used to speed this up?
        """
        positions = np.moveaxis(positions, source=-1, destination=0)
        return np.ravel_multi_index(positions, self._factor_sizes)

    def idx_to_pos(self, indices) -> np.ndarray:
        """
        Convert an index to a position (or convert a list of indices to a list of positions)
        - indices are integers < size
        - positions are lists of integers, with each element < their corresponding factor size

        TODO: can factor_multipliers be used to speed this up?
        """
        positions = np.array(np.unravel_index(indices, self._factor_sizes))
        return np.moveaxis(positions, source=0, destination=-1)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
