import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.ndimage
import seeding
from tqdm import tqdm

from visgrid.gridworld import GridWorld
from visgrid.taxi import VisTaxi5x5
from visgrid.sensors import *

# for seed in tqdm(range(100)):
seeding.seed(9, np, random)
env = VisTaxi5x5(wall_width=2,
                 cell_width=13,
                 passenger_width=9,
                 depot_width=3,
                 banner_widths=(4, 3),
                 dash_widths=(6, 6),
                 grayscale=False)
# env = VisTaxi5x5(wall_width=1,
#                  cell_width=11,
#                  passenger_width=7,
#                  depot_width=2,
#                  banner_widths=(2, 1),
#                  dash_widths=(4, 4),
#                  grayscale=False)
s = env.reset(goal=False)
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
images.append(sensor.observe(s))
for action in [0, 0, 3, 3, 0, 0, 2, 2, 4, 3, 3, 3, 3, 4, 2, 2, 1, 1, 1, 1, 2, 2]:
    s, _, _ = env.step(action)
    images.append(sensor.observe(s))

# convert to uint8
images -= np.min(images)
images /= np.max(images)
images *= 255
images = images.astype(np.uint8)

color_str = 'grayscale' if env.grayscale else 'rgb'
imageio.mimwrite('taxi-{}-v4-84x84.gif'.format(color_str), images, fps=2)

#%%
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for i, ax in zip([7, 8, 9, 10], axes):
    ax.imshow(images[i], interpolation='nearest', cmap='gray')
    ax.axis('off')
plt.show()
