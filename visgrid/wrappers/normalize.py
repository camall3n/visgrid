from typing import Union, Tuple, SupportsFloat

import gym
from gym import spaces
import numpy as np

class NormalizeWrapper(gym.ObservationWrapper):
    """Normalize float observations to a specific range (default = [0,1])
    """
    def __init__(self,
                 env: gym.Env,
                 low: Union[SupportsFloat, np.ndarray] = 0.0,
                 high: Union[SupportsFloat, np.ndarray] = 1.0):
        """
        Normalize float observations to a specific range (default = [0,1])

        Args:
            env (Env): The environment to apply the wrapper
        """
        super().__init__(env)

        ob_space = env.observation_space
        assert isinstance(ob_space, spaces.Box)
        assert np.issubdtype(ob_space.dtype, np.floating)
        assert ob_space.is_bounded('both')

        self.orig_low = ob_space.low
        self.orig_high = ob_space.high
        self.low = low
        self.high = high
        self.observation_space = spaces.Box(low=self.low,
                                            high=self.high,
                                            shape=ob_space.shape,
                                            dtype=np.float32)

    def observation(self, observation):
        obs_0_to_1 = (observation - self.orig_low) / (self.orig_high - self.orig_low)
        return obs_0_to_1 * (self.high - self.low) + self.low
