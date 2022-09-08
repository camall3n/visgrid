import numpy as np

import gym
from gym import spaces

class NormalizedFloatWrapper(gym.ObservationWrapper):
    """Convert discrete observations to floating point, normalized to interval [0,1]
    """
    def __init__(self, env: gym.Env):
        """
        Convert discrete observations to floating point, normalized to interval [0,1]

        Args:
            env (Env): The environment to apply the wrapper
        """
        super().__init__(env)

        ob_space = env.observation_space
        assert isinstance(ob_space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary))

        if isinstance(ob_space, spaces.Discrete):
            factor_sizes = (ob_space.n, )
        elif isinstance(ob_space, spaces.MultiDiscrete):
            factor_sizes = ob_space.nvec
        elif isinstance(ob_space, spaces.MultiBinary):
            factor_sizes = (2, ) * ob_space.n
            raise TypeError('Observation space must be Discrete or MultiDiscrete')

        self.max_values = np.asarray(factor_sizes) - 1
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(len(factor_sizes), ),
                                            dtype=np.float32)

    def observation(self, observation):
        return (observation / self.max_values).astype(np.float32)
