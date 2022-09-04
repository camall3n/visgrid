import numpy as np
import matplotlib.pyplot as plt

from visgrid.envs import GridworldEnv
from visgrid.sensors import *

rows, cols = 6, 6

sensor = SensorChain([
    OffsetSensor(offset=(0.5, 0.5)),
    NoisySensor(sigma=0.05),
    ImageSensor(range=((0, rows), (0, cols)), pixel_density=3),
    BlurSensor(sigma=0.6, truncate=1.),
    NoisySensor(sigma=0.01)
])
env = GridworldEnv(rows,
                   cols,
                   fixed_goal=False,
                   hidden_goal=True,
                   terminate_on_goal=False,
                   image_observations=False,
                   sensor=sensor)
env.reset().shape

#%%
obs = []
for _ in range(100):
    ob, reward, done, info = env.step(np.random.randint(4))
    obs.append(ob)
obs = np.stack(obs)

#%%
obs = env.get_observation()

plt.figure()
plt.imshow(obs)
plt.xticks([])
plt.yticks([])
plt.show()

#%%
env.plot()

#%%
env.sensor = Sensor()
env.hidden_goal = False
env.image_observations = True
env.plot()

# env = GridworldEnv(rows,
#                    cols,
#                    fixed_goal=False,
#                    terminate_on_goal=False,
#                    image_observations=True)
# env.reset().shape

#%%
obs = []
for _ in range(100):
    ob, reward, done, info = env.step(np.random.randint(4))
    obs.append(ob)
obs = np.stack(obs)

#%%
dir(env)
obs = env.get_observation()

plt.figure()
plt.imshow(obs)
plt.xticks([])
plt.yticks([])
plt.show()
