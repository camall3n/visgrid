import random
import os

import matplotlib.pyplot as plt
import numpy as np
import seeding
from tqdm import tqdm

from visgrid.envs.components import Grid
from visgrid.envs.gridworld import GridworldEnv

grid = Grid.generate_four_rooms()
env = GridworldEnv.from_grid(grid, dimensions=GridworldEnv.dimensions_13x13_to_84x84)
env.reset()
env.plot()
