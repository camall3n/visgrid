import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.ndimage
import seeding
from tqdm import tqdm

from visgrid.envs import GridworldEnv, TaxiEnv
from visgrid.agents.expert import TaxiExpert
from visgrid.wrappers.transforms import NoiseWrapper, ClipWrapper

env = TaxiEnv(exploring_starts=False,
              terminate_on_goal=True,
              depot_dropoff_only=False,
              should_render=True)
env = NoiseWrapper(env, 0.05)
env = ClipWrapper(env)

ob, _ = env.reset()
env.agent.position = (0, 4)
env.passengers[0].position = (0, 0)
env.passengers[0].color = 'yellow'

images = []
images.append(ob)
for action in [0, 0, 3, 3, 0, 0, 2, 2, 4, 3, 3, 3, 3, 4, 2, 2, 1, 1, 1, 1, 2, 2]:
    ob = env.step(action)[0]
    images.append(ob)

def convert_to_uint8(images):
    # convert to uint8
    images = np.stack(images)
    images -= np.min(images)
    images /= np.max(images)
    images *= 255
    images = images.astype(np.uint8)
    return images

imageio.mimwrite('taxi-v5-5x5-to-84x84.gif', convert_to_uint8(images), fps=2)

#%%
images = []
env = TaxiEnv(size=10,
              n_passengers=7,
              exploring_starts=False,
              terminate_on_goal=True,
              depot_dropoff_only=False,
              should_render=True)
env = NoiseWrapper(env, 0.05)
env = ClipWrapper(env)

ob, _ = env.reset()
expert = TaxiExpert(env)
images.append(ob)
n_steps = 0
while n_steps < 1000:
    action = expert.act()
    ob, reward, terminal, _, _ = env.step(action)
    images.append(ob)
    n_steps += 1
    if terminal:
        break
assert terminal == True
assert reward == 1

imageio.mimwrite('taxi-v5-10x10-to-64x64.mp4', convert_to_uint8(images), fps=16)
