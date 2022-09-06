import numpy as np

from visgrid.envs import GridworldEnv
from visgrid.wrappers.sensors import *

from time import time

N = 1000  # steps per trial

#%%
start = time()
env = GridworldEnv(
    rows=6,
    cols=6,
    exploring_starts=True,
    terminate_on_goal=False,
    fixed_goal=True,
    hidden_goal=True,
    agent_position=(5, 3),
    goal_position=(4, 0),
    image_observations=False,
)
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Base steps/sec: {N / (time() - start)}')

#%%
start = time()
env = GridworldEnv(rows=6,
                   cols=6,
                   exploring_starts=True,
                   terminate_on_goal=False,
                   fixed_goal=True,
                   hidden_goal=True,
                   agent_position=(5, 3),
                   goal_position=(4, 0),
                   image_observations=False,
                   sensor=SensorChain([
                       OffsetSensor(offset=(0.5, 0.5)),
                       ImageSensor(range=((0, 6), (0, 6)), pixel_density=3),
                       BlurSensor(sigma=0.6, truncate=1.),
                       NoiseSensor(sigma=0.01)
                   ]))
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Sensor steps/sec: {N / (time() - start)}')

#%%
start = time()
env = GridworldEnv(
    rows=6,
    cols=6,
    exploring_starts=True,
    terminate_on_goal=False,
    fixed_goal=True,
    hidden_goal=True,
    agent_position=(5, 3),
    goal_position=(4, 0),
    image_observations=True,
)
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Rendered steps/sec: {N / (time() - start)}')
