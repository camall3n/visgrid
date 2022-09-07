import matplotlib.pyplot as plt
import numpy as np

from visgrid.envs import GridworldEnv
from visgrid.wrappers.permutation import ObservationPermutationWrapper, FactorPermutationWrapper

#%% ------------------ Define MDP ------------------
env = GridworldEnv(rows=6, cols=6, image_observations=False)
env = FactorPermutationWrapper(env)

#%% ------------------ Generate experiences ------------------
n_samples = 500
fig = plt.figure()
fig.show()

def render(s):
    return env.unwrapped.get_observation(s)

x = env.reset(seed=1)[0]
for t in range(n_samples):
    s = env.get_state()
    o = np.concatenate((render(s), render(x)), axis=1)
    plt.imshow(o)
    fig.canvas.draw()
    fig.canvas.flush_events()
    while True:
        a = {
            'w': 2,
            's': 3,
            'a': 0,
            'd': 1,
        }[input('Move with WASD: ')]
        if env.can_run(a):
            break
    x = env.step(a)[0]
