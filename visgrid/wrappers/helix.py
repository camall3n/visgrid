from typing import Tuple

import gym
import numpy as np

from .base import BaseObservationWrapper

class HelixWrapper(BaseObservationWrapper):
    """
    Applies a helix transformation to all observations

    For inputs (x, y, z), the x-y plane is rotated by an amount depending on z:
            +x
             |
             |__           __
             /   `.    . `   /|
            /       `v      / |
           /          `-._ /  |
          |`              |   |___________+z
          |  ` _          |  /
          |     _--__     | /
          |__--       --__|/
         /
        /
      +y
    """
    def __init__(self,
                 env: gym.Env,
                 axes_xy: Tuple[int] = (0, 1),
                 axis_z: int = 2,
                 rotations_per_unit_z: float = 1.0):
        super().__init__(env)
        self.rotations_per_unit_z = rotations_per_unit_z
        self.axes_xy = list(axes_xy)
        self.axis_z = axis_z

    def observation(self, obs):
        xy = obs[self.axes_xy]
        z = obs[self.axis_z]
        theta = 2 * np.pi * self.rotations_per_unit_z * z
        xy = self._rotation_matrix(theta) @ xy
        obs[self.axes_xy] = xy
        return obs

    def _rotation_matrix(self, radians: float):
        return np.array([
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ])
