import gym

class BaseObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env, new_step_api=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def get_observation(self):
        obs = self.env.get_observation()
        return self.observation(obs)
