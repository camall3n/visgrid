import numpy as np

import gym
from gym.spaces import Box

class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert the image observation from RGB to grayscale.
    """
    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. observations are of the shape AxBx1.
                Otherwise, they are of shape AxB.
        """
        super().__init__(env)
        self.keep_dim = keep_dim

        assert (isinstance(self.observation_space, Box) and len(self.observation_space.shape) == 3
                and self.observation_space.shape[-1] == 3)

        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            obs_shape = obs_shape + (1, )
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=obs_shape,
                                     dtype=self.observation_space.dtype)

    def observation(self, observation):
        """Converts the colour observation to greyscale.

        Args:
            observation: Color observations

        Returns:
            Grayscale observations
        """
        import cv2

        if observation.dtype == np.float64:
            observation = observation.astype(np.float32)
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation
