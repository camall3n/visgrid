# import matplotlib as mpl
# mpl.use('Agg') # yapf:disable
import matplotlib.pyplot as plt
import numpy as np

from visgrid.envs import TaxiEnv
from visgrid.sensors import *

from visgrid.agents.expert.taxi_expert import TaxiExpert

#%% Test shapes
env = TaxiEnv(size=5,
              n_passengers=1,
              exploring_starts=False,
              terminate_on_goal=False,
              depot_dropoff_only=False,
              image_observations=False)
state = env.reset()[0]
assert state.shape == (6, )
initial_agent_position = tuple(state[:2])
initial_psgr_position = tuple(state[2:4])
initial_psgr_goal_id = state[-1]
initial_goal_position = tuple(env.depots[env.depot_names[initial_psgr_goal_id]].position)
assert initial_psgr_position != initial_agent_position
assert initial_psgr_position != initial_goal_position

#%% Test a random action sequence
obs, rewards, terminals, truncateds, infos = [], [], [], [], []
for _ in range(100):
    action = env.action_space.sample()
    ob, reward, terminal, truncated, info = env.step(action)
    obs.append(ob)
    rewards.append(reward)
    terminals.append(terminal)
    truncateds.append(truncated)
    infos.append(info)
obs = np.stack(obs)
rewards = np.stack(rewards)
terminals = np.stack(terminals)
assert obs.shape == (100, 6)
assert rewards.shape == terminals.shape == (100, )
assert any(rewards == 0) and any(terminals == False)
state = env.get_state()
agent_positions = [tuple(info['state'][:2]) for info in infos]
goal_position = tuple(env.depots[env.depot_names[state[-1]]].position)
assert not all(agent_pos == initial_agent_position for agent_pos in agent_positions)
assert goal_position == initial_goal_position

#%% Test image observations
env = TaxiEnv(size=5,
              n_passengers=1,
              exploring_starts=False,
              terminate_on_goal=False,
              depot_dropoff_only=False,
              image_observations=True)
ob, info = env.reset()
assert ob.shape == (84, 84, 3)
assert info['state'].shape == (6, )
env.plot()
plt.show()

#%% Test reaching passenger using expert agent
expert = TaxiExpert(env)
while not expert._at(env.agent, env.passengers[0]):
    action = expert.act()
    ob, reward, terminal, truncated, info = env.step(action)
    state = info['state']
    print(state)
agent_position = tuple(state[:2])
psgr_position = tuple(state[2:4])
assert agent_position == psgr_position

#%%
assert env.can_run(env.INTERACT)
ob, reward, terminal, truncated, info = env.step(env.INTERACT)
assert env.passenger is not None
assert tuple(env.agent.position) == tuple(env.passenger.position)
assert env.passenger.in_taxi == True

env.plot()
plt.show()
assert env._check_goal() == False
assert reward == 0
assert not terminal

#%% Test bringing passenger to desired depot using expert
goal_depot = env.depots[env.passenger.color]
while not expert._at(env.passenger, goal_depot):
    action = expert.act()
    ob, reward, terminal, truncated, info = env.step(action)
    state = info['state']
    print(state)
agent_position = tuple(state[:2])
psgr_position = tuple(state[2:4])
assert agent_position == psgr_position
assert agent_position == tuple(goal_depot.position)

#%% Test dropping off passenger when terminate_on_goal is False
assert env.can_run(env.INTERACT)
ob, reward, terminal, truncated, info = env.step(env.INTERACT)
assert env._check_goal() == True
assert reward == 0 and terminal == False
env.plot()
plt.show()
print(f'r = {reward}, terminal = {terminal}, truncated = {truncated}, info = {info}')

#%% Test reset changes agent & goal positions, but never sets them equal
reset_agent_positions = []
reset_goal_positions = []
for _ in range(10):
    env.reset()
    assert not env._check_goal()
    state = env.get_state()
    assert 0 <= state.min() and state.max() < 6
    reset_agent_positions.append(tuple(state[:2]))
    reset_goal_positions.append(tuple(state[2:]))
assert not all([agent_pos == initial_agent_position for agent_pos in reset_agent_positions])
assert not all([goal_pos == initial_goal_position for goal_pos in reset_goal_positions])

#%% test exploring starts
env = TaxiEnv(size=5,
              n_passengers=1,
              exploring_starts=True,
              terminate_on_goal=True,
              depot_dropoff_only=False,
              image_observations=True)
expert = TaxiExpert(env)
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
at_depot = lambda agent_pos: any([tuple(d.position) == agent_pos for d in env.depots.values()])
assert not all([at_depot(agent_pos) for agent_pos in reset_agent_positions])

#%% test terminate on goal
terminal = False
n_steps = 0
while n_steps < 1000:
    action = expert.act()
    ob, reward, terminal, truncated, info = env.step(action)
    n_steps += 1
    print(info['state'])
    if terminal:
        break
env.plot()
plt.show()
assert terminal == True
assert reward == 1

#%% test multiple passengers
env = TaxiEnv(size=5,
              n_passengers=3,
              exploring_starts=False,
              terminate_on_goal=True,
              depot_dropoff_only=False,
              image_observations=True)
ob, info = env.reset()
expert = TaxiExpert(env)
env.plot()
plt.show()
n_steps = 0
while n_steps < 1000:
    action = expert.act()
    ob, reward, terminal, truncated, info = env.step(action)
    n_steps += 1
    if terminal:
        break
env.plot()
assert terminal == True
assert reward == 1

#%% test extended 10x10 environment with 7 passengers
env = TaxiEnv(size=10,
              n_passengers=7,
              exploring_starts=False,
              terminate_on_goal=True,
              depot_dropoff_only=False,
              image_observations=True)
ob, info = env.reset()
expert = TaxiExpert(env)
env.plot()
n_steps = 0
while n_steps < 1000:
    action = expert.act()
    if not env.can_run(action):
        env.plot()
        plt.show()
    ob, reward, terminal, truncated, info = env.step(action)
    n_steps += 1
    if action == env.INTERACT:
        env.plot()
        plt.show()
    if terminal:
        break
env.plot()
plt.show()
assert terminal == True
assert reward == 1
