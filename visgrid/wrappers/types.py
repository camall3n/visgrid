import gym
from gym import spaces
import numpy as np

class ToFloatWrapper(gym.ObservationWrapper):
    """Convert discrete observations to floating point
    """
    def __init__(self, env: gym.Env, dtype=np.float32):
        """
        Convert discrete observations to floating point

        Args:
            env (Env): The environment to apply the wrapper
        """
        super().__init__(env)

        ob_space = env.observation_space
        if isinstance(ob_space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
            if isinstance(ob_space, spaces.Discrete):
                factor_sizes = (ob_space.n, )
            elif isinstance(ob_space, spaces.MultiDiscrete):
                factor_sizes = ob_space.nvec
            elif isinstance(ob_space, spaces.MultiBinary):
                factor_sizes = (2, ) * ob_space.n
                raise TypeError('Observation space must be Discrete or MultiDiscrete')

            max_values = (np.asarray(factor_sizes) - 1).astype(dtype)
            min_values = np.zeros_like(max_values, dtype=dtype)
            self.observation_space = spaces.Box(low=min_values,
                                                high=max_values,
                                                shape=(len(factor_sizes), ),
                                                dtype=dtype)
        elif isinstance(ob_space, spaces.Box):
            self.observation_space = spaces.Box(low=ob_space.low,
                                                high=ob_space.high,
                                                shape=ob_space.shape,
                                                dtype=dtype)

    def observation(self, observation):
        return observation.astype(self.observation_space.dtype)
