import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.ndimage
import seeding

from visgrid.gridworld import GridWorld
from visgrid.taxi import VisTaxi5x5
from visgrid.sensors import *

seeding.seed(2, np, random)

env = VisTaxi5x5()
s = env.reset(goal=False)

sensor_list = [
    MultiplySensor(scale=1 / 255),
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

imageio.mimwrite('taxi.mp4', images, fps=2)
