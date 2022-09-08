import pytest

import matplotlib.pyplot as plt
import numpy as np

from visgrid.envs import GridworldEnv
from visgrid.wrappers.transforms import wrap_gridworld

@pytest.fixture
def initial_agent_position():
    return (5, 3)

@pytest.fixture
def initial_goal_position():
    return (4, 0)

@pytest.fixture
def env(initial_agent_position, initial_goal_position):
    env = GridworldEnv(rows=6,
                       cols=6,
                       exploring_starts=False,
                       terminate_on_goal=False,
                       fixed_goal=True,
                       hidden_goal=True,
                       agent_position=initial_agent_position,
                       goal_position=initial_goal_position,
                       should_render=True,
                       dimensions=GridworldEnv.dimensions_6x6_to_18x18)
    env = wrap_gridworld(env)
    env.reset()
    return env

def test_sensor_chain_produces_images(env):
    ob, _ = env.reset()
    assert ob.shape == (18, 18)

def test_sensor_chain_produces_images(env):
    ob, _ = env.unwrapped.reset()
    assert ob.shape == (18, 18, 3)

def test_sensor_chain_produces_noisy_images(env):
    start_ob = env.reset()[0]
    start_state = env.get_state()
    for _ in range(10):
        ob = env.reset()[0]
        state = env.get_state()
        assert np.all(state == start_state)
        assert not np.all(ob == start_ob)
