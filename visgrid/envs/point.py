import gym
import gym.spaces
import numpy as np

class PointEnv(gym.Env):
    """
    A simple n-dimensional continuous state space with continuous actions,
    where the actions define the exact change in state at each time step.
    """
    def __init__(self, ndim: int = 5) -> None:
        super().__init__()
        self.ndim = ndim
        self.x = np.zeros(ndim)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.ndim, ))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.ndim, ))

    def reset(self, x=None, seed: int = None):
        super().reset(seed=seed)
        if x is not None:
            self.x = np.array(x)
        else:
            self.x = np.zeros(self.ndim)
        return self.x.copy(), {}

    def step(self, action):
        self.x += action
        return self.x.copy(), 0, False, False, {}
