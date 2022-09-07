import gym

class BaseObservationWrapper(gym.ObservationWrapper):
    """ObservationWrapper stand-in for new-style gym API"""
    def __init__(self, env) -> None:
        super().__init__(env, new_step_api=True)

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        observation, info = super().reset(**kwargs)
        return self.observation(observation), info

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        step_returns = self.env.step(action)
        observation, reward, terminated, truncated, info = step_returns
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation):
        """Returns a modified observation."""
        raise NotImplementedError
