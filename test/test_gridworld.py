import matplotlib.pyplot as plt
import numpy as np

from visgrid.envs import GridworldEnv
from visgrid.agents.expert.gridworld_expert import GridworldExpert
from visgrid.envs.components.grid import Grid
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
assert env.reset()[0].shape == (4, )
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
obs, rewards, terminals, truncateds, infos = [], [], [], [], []
for action in [0, 0, 1, 1, 2, 2, 0, 3, 3, 0]:
    assert env.can_run(action)
    ob, reward, terminal, truncated, info = env.step(action)
    assert not terminal
    obs.append(ob)
    rewards.append(reward)
    terminals.append(terminal)
    truncateds.append(truncated)
    infos.append(info)
obs = np.stack(obs)
rewards = np.stack(rewards)
terminals = np.stack(terminals)
assert obs.shape == (10, 18, 18)
assert rewards.shape == terminals.shape == (10, )
assert all(rewards == 0) and all(terminals == False)
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
for action in [3]:
    env.step(action)
visible_goal_image = env.get_observation()
assert (hidden_goal_image != visible_goal_image).any()
env.plot()

#%% Test reaching goal when terminate_on_goal is False
for action in [2, 0]:
    ob, reward, terminal, truncated, info = env.step(action)
env.plot()
plt.show()
print(f'r = {reward}, terminal = {terminal}, truncated = {truncated}, info = {info}')
assert env._check_goal() == True
assert reward == 0 and not terminal

#%% Test noop action reaching goal when terminate_on_goal is True
env.terminate_on_goal = True
for action in [0]:
    assert not env.can_run(action)
    ob, reward, terminal, truncated, info = env.step(action)
env.plot()
print(f'r = {reward}, terminal = {terminal}, truncated = {truncated}, info = {info}')
assert env._check_goal() == True
assert reward == 1 and terminal

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

#%% Test loading from saved maze
env = GridworldEnv.from_saved_maze(6, 6, 7, agent_position=(0, 0), goal_position=(5, 5))
expert = GridworldExpert(env)
ob, info = env.reset()
env.plot()
n_steps = 0
while n_steps < 50:
    action = expert.act(ob)
    if not env.can_run(action):
        env.plot()
        plt.show()
    ob, reward, terminal, truncated, info = env.step(action)
    n_steps += 1
    if terminal:
        break
env.plot()
plt.show()
assert terminal == True
assert reward == 1


#%% Test loading from file
env = GridworldEnv.from_file('visgrid/envs/saved/test_3x4.txt')
env.reset()
env.plot()

#%% Test constructing gridworld directly from a grid
grid = Grid.generate_spiral(6, 6)
env = GridworldEnv.from_grid(grid, agent_position=(0, 0), goal_position=(2, 3))
expert = GridworldExpert(env)
ob, info = env.reset()
env.plot()
assert tuple(env.agent.position) == (0, 0)
assert tuple(env.goal.position) == (2, 3)
n_steps = 0
while n_steps < 50:
    action = expert.act(ob)
    if not env.can_run(action):
        env.plot()
        plt.show()
    ob, reward, terminal, truncated, info = env.step(action)
    n_steps += 1
    if terminal:
        break
env.plot()
plt.show()
assert terminal == True
assert reward == 1

#%% Test saving a grid
grid = Grid(6, 6)
grid[1:5, 4:9] = 1
grid[8:, 4:9] = 1
grid.save('visgrid/envs/saved/h_maze_6x6.txt')
env = GridworldEnv.from_grid(grid, goal_position=(0, 5), agent_position=(5, 0))
env.reset()
env.plot()

#%% Test saving a grid
grid = Grid(6, 6)
grid[4:9, :9] = 1
grid.save('visgrid/envs/saved/u_maze_6x6.txt')
env = GridworldEnv.from_grid(grid, goal_position=(0, 0), agent_position=(5, 0))
env.reset()
env.plot()

#%% Test generating ring mazes
grid = Grid.generate_ring(6, 6, width=2)
env = GridworldEnv.from_grid(grid)
invalid_positions = [(2, 2), (2, 3), (3, 2), (3, 3)]
for _ in range(100):
    env.reset()
    assert tuple(env.agent.position) not in invalid_positions
    assert tuple(env.goal.position) not in invalid_positions
env.plot()
