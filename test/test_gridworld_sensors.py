import numpy as np
import matplotlib.pyplot as plt

from visgrid.envs import GridworldEnv
from visgrid.sensors import *

rows, cols = 6, 6

sensor = SensorChain([
    # OffsetSensor(offset=(0.5, 0.5)),
    # NoisySensor(sigma=0.05),
    # ImageSensor(range=((0, rows), (0, cols)), pixel_density=3),
    # ResampleSensor(scale=2.0),
    # BlurSensor(sigma=0.6, truncate=1.),
    # NoisySensor(sigma=0.01)
])
env = GridworldEnv(rows,
                   cols,
                   fixed_goal=False,
                   terminate_on_goal=False,
                   image_observations=False,
                   sensor=sensor)
env.reset()

# #%%
# obs = []
# for _ in range(100):
#     ob, reward, done, info = env.step(np.random.randint(4))
#     obs.append(ob)
# obs = np.stack(obs)

# #%%
# s = env.get_state()
# obs = sensor(s)

# plt.figure()
# plt.imshow(obs)
# plt.xticks([])
# plt.yticks([])
# plt.show()

#%%
ob = env.render()
plt.imshow(ob)
plt.show()

#%%
for a in [0, 3]:
    env.step(a)
    ob = env.render()
    plt.imshow(ob)
