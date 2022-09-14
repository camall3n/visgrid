import pytest

import matplotlib.pyplot as plt
import numpy as np

from visgrid.envs import GridworldEnv
from visgrid.agents.expert import GridworldExpert
from visgrid.envs.components import Grid

@pytest.fixture
def initial_agent_position():
    return (5, 3)

@pytest.fixture
def initial_goal_position():
    return (4, 0)

def test_initial_positions(initial_agent_position, initial_goal_position):
    env = GridworldEnv(rows=6,
                       cols=6,
                       exploring_starts=False,
                       terminate_on_goal=False,
                       fixed_goal=True,
                       hidden_goal=False,
                       agent_position=initial_agent_position,
                       goal_position=initial_goal_position,
                       should_render=False)

    assert env.reset()[0].shape == (4, )
    assert tuple(env.get_state()[:2]) == initial_agent_position
    assert tuple(env.get_state()[2:]) == initial_goal_position

def test_hidden_goal_changes_obs_size(initial_agent_position, initial_goal_position):
    env = GridworldEnv(rows=6,
                       cols=6,
                       exploring_starts=False,
                       terminate_on_goal=False,
                       fixed_goal=True,
                       hidden_goal=True,
                       agent_position=initial_agent_position,
                       goal_position=initial_goal_position,
                       should_render=False)
    ob, _ = env.reset()
    assert ob.shape == (2, )

@pytest.fixture
def sensor_env(initial_agent_position, initial_goal_position):
    env = GridworldEnv(rows=6,
                       cols=6,
                       exploring_starts=False,
                       terminate_on_goal=False,
                       fixed_goal=True,
                       hidden_goal=True,
                       agent_position=initial_agent_position,
                       goal_position=initial_goal_position,
                       should_render=False)
    env.reset()
    return env

def test_deterministic_action_sequence(sensor_env, initial_agent_position):
    obs, rewards, terminals, truncateds, infos = [], [], [], [], []
    for action in [0, 0, 1, 1, 2, 2, 0, 3, 3, 0]:
        assert sensor_env.can_run(action)
        ob, reward, terminal, truncated, info = sensor_env.step(action)
        assert not terminal
        obs.append(ob)
        rewards.append(reward)
        terminals.append(terminal)
        truncateds.append(truncated)
        infos.append(info)
    obs = np.stack(obs)
    rewards = np.stack(rewards)
    terminals = np.stack(terminals)
    assert rewards.shape == terminals.shape == (10, )
    assert all(rewards == 0) and all(terminals == False)
    assert tuple(sensor_env.get_state()[:2]) != initial_agent_position

@pytest.fixture
def env4(initial_agent_position, initial_goal_position):
    env = GridworldEnv(rows=6,
                       cols=6,
                       exploring_starts=False,
                       terminate_on_goal=False,
                       fixed_goal=True,
                       hidden_goal=True,
                       agent_position=initial_agent_position,
                       goal_position=initial_goal_position,
                       should_render=True)
    env.reset()
    return env

@pytest.fixture
def hidden_goal_image(env4):
    return env4.reset()[0]

def test_rendering_with_hidden_goal(hidden_goal_image):
    assert hidden_goal_image.shape == (64, 64, 3)

@pytest.fixture
def env5(initial_agent_position, initial_goal_position):
    env = GridworldEnv(rows=6,
                       cols=6,
                       exploring_starts=False,
                       terminate_on_goal=False,
                       fixed_goal=True,
                       hidden_goal=False,
                       agent_position=initial_agent_position,
                       goal_position=initial_goal_position,
                       should_render=True)
    env.reset()
    return env

def test_rendering_with_visible_goal(env5, hidden_goal_image):
    visible_goal_image, _ = env5.reset()
    assert not np.all(hidden_goal_image == visible_goal_image)

def test_reaching_goal_when_terminate_on_goal_is_false(env5):
    for action in [0, 0, 2, 0]:
        ob, reward, terminal, truncated, info = env5.step(action)
    assert env5._check_goal() == True
    assert reward == 0 and not terminal

@pytest.fixture
def env6(initial_agent_position, initial_goal_position):
    env = GridworldEnv(rows=6,
                       cols=6,
                       exploring_starts=False,
                       terminate_on_goal=True,
                       fixed_goal=True,
                       hidden_goal=False,
                       agent_position=initial_agent_position,
                       goal_position=initial_goal_position,
                       should_render=True)
    env.reset()
    return env

def test_reaching_goal_when_terminate_on_goal_is_true(env6):
    for action in [0, 0, 2, 0]:
        assert env6.can_run(action)
        ob, reward, terminal, truncated, info = env6.step(action)
    assert env6._check_goal() == True
    assert not env6.can_run(0)
    assert reward == 1 and terminal

def test_reset_uses_initial_positions(env6, initial_agent_position, initial_goal_position):
    env6.reset()
    env6.get_state()
    assert tuple(env6.get_state()[:2]) == initial_agent_position
    assert tuple(env6.get_state()[2:]) == initial_goal_position

@pytest.fixture
def env7(initial_agent_position):
    env = GridworldEnv(rows=6,
                       cols=6,
                       exploring_starts=False,
                       terminate_on_goal=True,
                       fixed_goal=False,
                       hidden_goal=False,
                       agent_position=initial_agent_position,
                       should_render=True)
    env.reset()
    return env

def test_reset_goal_but_not_agent(env7: GridworldEnv, initial_agent_position,
                                  initial_goal_position):
    reset_agent_positions = []
    reset_goal_positions = []
    for _ in range(100):
        env7.reset()
        state = env7.get_state()
        assert 0 <= state.min() and state.max() < 6
        reset_agent_positions.append(tuple(state[:2]))
        reset_goal_positions.append(tuple(state[2:]))
    assert all([agent_pos == initial_agent_position for agent_pos in reset_agent_positions])
    assert not all([goal_pos == initial_goal_position for goal_pos in reset_goal_positions])

@pytest.fixture
def env8():
    env = GridworldEnv(rows=6,
                       cols=6,
                       exploring_starts=True,
                       terminate_on_goal=True,
                       fixed_goal=False,
                       hidden_goal=False,
                       should_render=True)
    env.reset()
    return env

def test_exploring_resets_unique_positions(env8, initial_agent_position, initial_goal_position):
    reset_agent_positions = []
    reset_goal_positions = []
    for _ in range(100):
        env8.reset()
        assert not env8._check_goal()
        state = env8.get_state()
        assert 0 <= state.min() and state.max() < 6
        reset_agent_positions.append(tuple(state[:2]))
        reset_goal_positions.append(tuple(state[2:]))
    assert not all([agent_pos == initial_agent_position for agent_pos in reset_agent_positions])
    assert not all([goal_pos == initial_goal_position for goal_pos in reset_goal_positions])

def test_loading_from_saved_maze():
    env = GridworldEnv.from_saved_maze(6, 6, 7, agent_position=(0, 0), goal_position=(5, 5))
    expert = GridworldExpert(env)
    ob, info = env.reset()
    n_steps = 0
    while n_steps < 50:
        action = expert.act(ob)
        ob, reward, terminal, truncated, info = env.step(action)
        n_steps += 1
        if terminal:
            break
    assert terminal == True
    assert reward == 1

def test_loading_from_file():
    env = GridworldEnv.from_file('visgrid/envs/saved/test_3x4.txt')
    with pytest.warns(RuntimeWarning):
        env.reset()
    assert env.rows == 3 and env.cols == 4

def test_constructing_gridworld_directly_from_grid():
    grid = Grid.generate_spiral(6, 6)
    env = GridworldEnv.from_grid(grid, agent_position=(0, 0), goal_position=(2, 3))
    expert = GridworldExpert(env)
    ob, info = env.reset()
    assert env.rows == 6 and env.cols == 6
    assert tuple(env.agent.position) == (0, 0)
    assert tuple(env.goal.position) == (2, 3)
    n_steps = 0
    while n_steps < 50:
        action = expert.act(ob)
        ob, reward, terminal, truncated, info = env.step(action)
        n_steps += 1
        if terminal:
            break
    assert terminal == True
    assert reward == 1

def test_saving_and_loading_grid(tmp_path):
    filepath = tmp_path / 'grid.txt'
    grid = Grid(6, 6)
    grid[1:5, 4:9] = 1
    grid[8:, 4:9] = 1
    grid.save(filepath)
    env1 = GridworldEnv.from_grid(grid, goal_position=(0, 5), agent_position=(5, 0))
    assert np.all(env1.grid._grid == grid._grid)
    env2 = GridworldEnv.from_file(filepath, goal_position=(0, 5), agent_position=(5, 0))
    assert np.all(env1.grid._grid == env2.grid._grid)

@pytest.fixture
def ring_grid():
    return Grid.generate_ring(6, 6, width=2)

def test_generating_valid_states(ring_grid):
    grid = ring_grid
    invalid_positions = [(2, 2), (2, 3), (3, 2), (3, 3)]
    for _ in range(10000):
        position = tuple(grid.get_random_position())
        assert position not in invalid_positions

def test_generating_ring_mazes(ring_grid):
    grid = ring_grid
    env = GridworldEnv.from_grid(grid)
    invalid_positions = [(2, 2), (2, 3), (3, 2), (3, 3)]
    for _ in range(100):
        env.reset()
        assert tuple(env.agent.position) not in invalid_positions
        assert tuple(env.goal.position) not in invalid_positions
