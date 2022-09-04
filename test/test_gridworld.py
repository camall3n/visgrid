import random

import matplotlib.pyplot as plt
import numpy as np
import seeding

from visgrid.envs import GridworldEnv
from visgrid.sensors import *

#%% Test initial positions
env = GridworldEnv(rows=6,
                   cols=6,
                   exploring_starts=False,
                   terminate_on_goal=False,
                   fixed_goal=True,
                   hidden_goal=False,
                   agent_position=(5, 3),
                   goal_position=(4, 0),
                   image_observations=False)
assert env.reset().shape == (4, )
initial_agent_position = tuple(env.get_state()[:2])
assert initial_agent_position == (5, 3)
initial_goal_position = tuple(env.get_state()[2:])
assert initial_goal_position == (4, 0)

#%% Test that hidden_goal changes observation size
env.hidden_goal = True
assert env.get_observation().shape == (2, )

#%% Test that sensor chain produces noisy images
sensor = SensorChain([
    OffsetSensor(offset=(0.5, 0.5)),
    ImageSensor(range=((0, env.rows), (0, env.cols)), pixel_density=3),
    BlurSensor(sigma=0.6, truncate=1.),
    NoiseSensor(sigma=0.01)
])
env.sensor = sensor
env.plot()
assert env.get_observation().shape == (18, 18)

#%% Test a deterministic action sequence
obs, rewards, dones, infos = [], [], [], []
for action in [0, 0, 1, 1, 2, 2, 0, 3, 3, 0]:
    assert env.can_run(action)
    ob, reward, done, info = env.step(action)
    assert not done
    obs.append(ob)
    rewards.append(reward)
    dones.append(done)
    infos.append(info)
obs = np.stack(obs)
rewards = np.stack(rewards)
dones = np.stack(dones)
assert obs.shape == (10, 18, 18)
assert rewards.shape == dones.shape == (10, )
assert all(rewards == 0) and all(dones == False)
assert tuple(env.get_state()[:2]) != initial_agent_position
assert tuple(env.get_state()[2:]) == initial_goal_position

#%% Test image observations with hidden goal
env.image_observations = True
env.sensor = Sensor()
hidden_goal_image = env.get_observation()
assert hidden_goal_image.shape == (64, 64, 3)
env.plot()

#%% Test image observations with visible goal
env.hidden_goal = False
visible_goal_image = env.get_observation()
assert (hidden_goal_image != visible_goal_image).any()
env.plot()

#%% Test reaching goal when terminate_on_goal is False
for action in [2, 0]:
    ob, reward, done, info = env.step(action)
env.plot()
plt.show()
print(f'r = {reward}, done = {done}, info = {info}')
assert env._check_goal() == True
assert reward == 0 and not done

#%% Test noop action reaching goal when terminate_on_goal is True
env.terminate_on_goal = True
for action in [0]:
    assert not env.can_run(action)
    ob, reward, done, info = env.step(action)
env.plot()
print(f'r = {reward}, done = {done}, info = {info}')
assert env._check_goal() == True
assert reward == 1 and done

#%% Test reset uses initial positions
env.reset()
env.plot()
env.get_state()
assert tuple(env.get_state()[:2]) == initial_agent_position
assert tuple(env.get_state()[2:]) == initial_goal_position

#%% Test reset still uses initial agent position when fixed_goal is False
env.fixed_goal = False
env.reset()
env.plot()
assert tuple(env.get_state()[:2]) == initial_agent_position

#%% Test reset changes agent & goal positions, but never sets them equal
env.exploring_starts = True
reset_agent_positions = []
reset_goal_positions = []
for _ in range(100):
    env.reset()
    assert not env._check_goal()
    state = env.get_state()
    assert 0 <= state.min() and state.max() < 6
    reset_agent_positions.append(tuple(state[:2]))
    reset_goal_positions.append(tuple(state[2:]))
assert not all([agent_pos == initial_agent_position for agent_pos in reset_agent_positions])
assert not all([goal_pos == initial_goal_position for goal_pos in reset_goal_positions])
