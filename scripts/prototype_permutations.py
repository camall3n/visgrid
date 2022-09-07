import matplotlib.pyplot as plt
import numpy as np

from visgrid.envs import GridworldEnv
from visgrid.wrappers.permutation import ObservationPermutationWrapper, FactorPermutationWrapper

#%% ------------------ Define MDP ------------------
env = GridworldEnv(rows=6, cols=6, image_observations=False)
env = FactorPermutationWrapper(env)
x = env.reset(seed=1)[0]

#%% ------------------ Generate experiences ------------------
n_samples = 500
fig = plt.figure()
fig.show()

for t in range(n_samples):
    x = env.get_observation()
    s = env.get_state()
    o = np.concatenate((env.render(s), env.render(x)), axis=1)
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
