from visgrid.envs import TaxiEnv
from visgrid.sensors import *

from time import time

N = 1000  # steps per trial

#%%
start = time()
env = TaxiEnv(
    size=5,
    n_passengers=1,
    exploring_starts=True,
    terminate_on_goal=False,
    depot_dropoff_only=False,
    image_observations=False,
)
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Base 5x5 steps/sec: {N / (time() - start)}')

#%%
start = time()
env = TaxiEnv(
    size=10,
    n_passengers=1,
    exploring_starts=True,
    terminate_on_goal=False,
    depot_dropoff_only=False,
    image_observations=False,
)
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Base 10x10 steps/sec: {N / (time() - start)}')

#%%
start = time()
env = TaxiEnv(
    size=5,
    n_passengers=1,
    exploring_starts=True,
    terminate_on_goal=False,
    depot_dropoff_only=False,
    image_observations=True,
    dimensions = TaxiEnv.dimensions_5x5_to_64x64,
)
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Rendered (5x5 @ 64x64x3) steps/sec: {N / (time() - start)}')

#%%
start = time()
env = TaxiEnv(
    size=5,
    n_passengers=1,
    exploring_starts=True,
    terminate_on_goal=False,
    depot_dropoff_only=False,
    image_observations=True,
)
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Rendered (5x5 @ 84x84x3) steps/sec: {N / (time() - start)}')

#%%
start = time()
env = TaxiEnv(
    size=10,
    n_passengers=1,
    exploring_starts=True,
    terminate_on_goal=False,
    depot_dropoff_only=False,
    image_observations=True,
)
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Rendered (10x10 @ 64x64x3) steps/sec: {N / (time() - start)}')
