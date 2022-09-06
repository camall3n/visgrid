#%%
import imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seeding
import sys
import time
from tqdm import tqdm

from visgrid.envs import GridworldEnv
from visgrid.utils import get_parser
from visgrid.wrappers.sensors import *

parser = get_parser()
if 'ipykernel' in sys.argv[0]:
    sys.argv = []
# yapf: disable
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
parser.add_argument('-n', '--n_updates', type=int, default=10000, help='Number of training updates')
parser.add_argument('-r', '--rows', type=int, default=7, help='Number of gridworld rows')
parser.add_argument('-c', '--cols', type=int, default=4, help='Number of gridworld columns')
parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
# yapf: enable
args = parser.parse_args()

seeding.seed(args.seed, np)

#%% ------------------ Define MDP ------------------
env = GridworldEnv(rows=args.rows, cols=args.cols)
cmap = None

sensor = SensorChain([
    RearrangeXYPositionsSensor((env.rows, env.cols))
    # OffsetSensor(offset=(0.5, 0.5)),
    # NoisySensor(sigma=0.05),
    # ImageSensor(range=((0, env.rows), (0, env.cols)), pixel_density=3),
    # BlurSensor(sigma=0.6, truncate=1.),
    # NoisySensor(sigma=0.01)
])
image_sensor = ImageSensor(range=((0, env.rows), (0, env.cols)), pixel_density=1)

#%% ------------------ Generate experiences ------------------
n_samples = 50000
# states = [env.get_state()]
# actions = []
fig = plt.figure()
fig.show()

for t in range(n_samples):
    s = env.get_state()
    x = sensor([s])[0]
    o = np.concatenate((image_sensor(s), image_sensor(x)), axis=1)
    plt.imshow(o)
    fig.canvas.draw()
    fig.canvas.flush_events()
    while True:
        # a = env.action_space.sample()
        a = {
            'w': 2,
            's': 3,
            'a': 0,
            'd': 1,
        }[input('Move with WASD: ')]
        if env.can_run(a):
            break
    s, _, _ = env.step(a)
    # states.append(s)
    # actions.append(a)
# states = np.stack(states)
# s0 = np.asarray(states[:-1, :])
# c0 = s0[:, 0] * env.cols + s0[:, 1]
# s1 = np.asarray(states[1:, :])
# a = np.asarray(actions)

ds = list(map(np.linalg.norm, s1 - s0))

#%% ------------------ Define sensor ------------------
x0 = sensor(s0)
x1 = sensor(s1)
dx = list(map(np.linalg.norm, x1 - x0))

#%% ------------------ Plot ds vs dx ------------------
# import matplotlib.pyplot as plt
# plt.plot(x0[:, 0], x1[:, 1])
# plt.plot(s0[:, 0], s0[:, 1])
# plt.show()

#%%

for s, x in zip(s0[:20], x0):
    o = np.concatenate((image_sensor(s), image_sensor(x)), axis=1)
    plt.imshow(o)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)

# %%
