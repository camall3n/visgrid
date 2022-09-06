import random
import os

import matplotlib.pyplot as plt
import numpy as np
import seeding
from tqdm import tqdm

from visgrid.envs.components import Grid
from visgrid.envs.gridworld import GridworldEnv

rows = 6
cols = 6

for seed in tqdm(range(1, 301)):
    seeding.seed(seed, np, random)
    grid = Grid.generate_maze(rows=rows, cols=cols)
    maze_dir = 'visgrid/envs/gridworld/saved/mazes_{}x{}/seed-{:03d}/'.format(rows, cols, seed)
    os.makedirs(maze_dir, exist_ok=True)

    txt_file = 'maze-{}.txt'.format(seed)
    grid.save(maze_dir + txt_file)

    png_file = 'maze-{}.png'.format(seed)
    env = GridworldEnv.from_grid(grid)
    env.plot()
    plt.savefig(maze_dir + png_file, facecolor='white', edgecolor='none')
    plt.close()
