import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.ndimage
import seeding
from tqdm import tqdm

from visgrid.envs import GridworldEnv, TaxiEnv
from visgrid.sensors import *

# for seed in tqdm(range(100)):
seeding.seed(94, np, random)
env = TaxiEnv(terminate_on_goal=False, exploring_starts=False)
s = env.reset()
# taxi_top_right = np.array_equal(env.agent.position, np.array([0, 4]))
# passenger_top_left = np.array_equal(env.passengers[0].position, np.array([0, 0]))
# passenger_yellow = env.passengers[0].color == 'yellow'
# if taxi_top_right and passenger_top_left and passenger_yellow:
#     break

sensor_list = [
    # MultiplySensor(scale=1 / 255),
    NoisySensor(sigma=0.05),
]
sensor = SensorChain(sensor_list)

images = []
images.append(sensor(s))
for action in [0, 0, 3, 3, 0, 0, 2, 2, 4, 3, 3, 3, 3, 4, 2, 2, 1, 1, 1, 1, 2, 2]:
    s, _, _, _ = env.step(action)
    images.append(sensor(s))

# convert to uint8
images = np.stack(images)
images -= np.min(images)
images /= np.max(images)
images *= 255
images = images.astype(np.uint8)

imageio.mimwrite('taxi-rgb-v5-84x84.gif', images, fps=2)

#%%
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for i, ax in zip([7, 8, 9, 10], axes):
    ax.imshow(images[i], interpolation='nearest', cmap='gray')
    ax.axis('off')
plt.show()
