import pytest

from typing import Tuple

import numpy as np

from visgrid.envs import TaxiEnv
from visgrid.agents.expert import TaxiExpert

#%% Test shapes

@pytest.fixture
def state_env():
    return TaxiEnv(size=5,
                   n_passengers=1,
                   exploring_starts=False,
                   terminate_on_goal=False,
                   depot_dropoff_only=False,
                   image_observations=False)

@pytest.fixture
def initial_state(state_env):
    return state_env.reset()[0]

def test_state_shape(initial_state):
    assert initial_state.shape == (6, )

@pytest.fixture
def initial_agent_position(initial_state):
    return tuple(initial_state[:2])

@pytest.fixture
def initial_psgr_position(initial_state):
    return tuple(initial_state[2:4])

@pytest.fixture
def initial_goal_position(state_env, initial_state):
    initial_psgr_goal_id = initial_state[-1]
    return tuple(state_env.depots[state_env.depot_names[initial_psgr_goal_id]].position)

def test_initial_positions(initial_psgr_position, initial_agent_position, initial_goal_position):
    assert initial_psgr_position != initial_agent_position
    assert initial_psgr_position != initial_goal_position

def test_random_action_sequence(state_env: TaxiEnv, initial_agent_position, initial_goal_position):
    env = state_env
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

@pytest.fixture
def img_env():
    return TaxiEnv(size=5,
                   n_passengers=1,
                   exploring_starts=False,
                   terminate_on_goal=False,
                   depot_dropoff_only=False,
                   image_observations=True)

def test_image_observations(img_env):
    ob, info = img_env.reset()
    assert ob.shape == (84, 84, 3)
    assert info['state'].shape == (6, )

@pytest.fixture
def img_env_expert(img_env):
    return TaxiExpert(img_env)

@pytest.fixture
def img_env_at_passenger(img_env: TaxiEnv, img_env_expert: TaxiExpert):
    while not img_env_expert._at(img_env.agent, img_env.passengers[0]):
        action = img_env_expert.act()
        img_env.step(action)
    return img_env

def test_expert_to_reach_passenger(img_env_at_passenger: TaxiEnv):
    state = img_env_at_passenger.get_state()
    agent_position = tuple(state[:2])
    psgr_position = tuple(state[2:4])
    assert agent_position == psgr_position

def test_can_pickup(img_env_at_passenger: TaxiEnv):
    env = img_env_at_passenger
    assert env.can_run(env.INTERACT)

@pytest.fixture
def img_env_with_passenger(img_env_at_passenger):
    env = img_env_at_passenger
    env.step(env.INTERACT)
    return env

def test_pickup(img_env_with_passenger):
    env = img_env_with_passenger
    assert env.passenger is not None
    assert tuple(env.agent.position) == tuple(env.passenger.position)
    assert env.passenger.in_taxi == True
    assert env._check_goal() == False

@pytest.fixture
def img_env_and_state_at_goal(img_env_with_passenger: TaxiEnv, img_env_expert: TaxiExpert):
    env = img_env_with_passenger
    goal_depot = env.depots[env.passenger.color]
    while not img_env_expert._at(env.passenger, goal_depot):
        action = img_env_expert.act()
        env.step(action)
        state = env.get_state()
    return env, state

def test_at_goal(img_env_and_state_at_goal: Tuple[TaxiEnv, tuple]):
    env, state = img_env_and_state_at_goal
    goal_depot = env.depots[env.passenger.color]
    agent_position = tuple(state[:2])
    psgr_position = tuple(state[2:4])
    assert agent_position == psgr_position
    assert agent_position == tuple(goal_depot.position)

def test_dropoff_without_termination(img_env_and_state_at_goal: Tuple[TaxiEnv, tuple]):
    env, _ = img_env_and_state_at_goal
    assert env.can_run(env.INTERACT)
    ob, reward, terminal, truncated, info = env.step(env.INTERACT)
    assert env._check_goal() == True
    assert reward == 0 and terminal == False

def test_reset_agent_and_goal_unique(img_env_with_passenger: TaxiEnv, initial_agent_position,
                                     initial_goal_position):
    env = img_env_with_passenger
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

@pytest.fixture
def exploring_env():
    env = TaxiEnv(size=5,
                  n_passengers=1,
                  exploring_starts=True,
                  terminate_on_goal=True,
                  depot_dropoff_only=False,
                  image_observations=True)
    env.reset()
    return env

def test_exploring_starts(exploring_env):
    reset_agent_positions = []
    reset_goal_positions = []
    for _ in range(100):
        exploring_env.reset()
        assert not exploring_env._check_goal()
        state = exploring_env.get_state()
        assert 0 <= state.min() and state.max() < 6
        reset_agent_positions.append(tuple(state[:2]))
        reset_goal_positions.append(tuple(state[2:]))
    assert not all([agent_pos == initial_agent_position for agent_pos in reset_agent_positions])
    assert not all([goal_pos == initial_goal_position for goal_pos in reset_goal_positions])
    at_depot = lambda agent_pos: any(
        [tuple(d.position) == agent_pos for d in exploring_env.depots.values()])
    assert not all([at_depot(agent_pos) for agent_pos in reset_agent_positions])

def test_terminate_on_goal(exploring_env):
    expert = TaxiExpert(exploring_env)
    terminal = False
    n_steps = 0
    while n_steps < 1000:
        action = expert.act()
        ob, reward, terminal, truncated, info = exploring_env.step(action)
        n_steps += 1
        if terminal:
            break
    assert terminal == True
    assert reward == 1

def test_multiple_passengers():
    env = TaxiEnv(size=5,
                  n_passengers=3,
                  exploring_starts=False,
                  terminate_on_goal=True,
                  depot_dropoff_only=False,
                  image_observations=True)
    env.reset()
    expert = TaxiExpert(env)
    n_steps = 0
    while n_steps < 1000:
        action = expert.act()
        ob, reward, terminal, truncated, info = env.step(action)
        n_steps += 1
        if terminal:
            break
    assert terminal == True
    assert reward == 1

def test_extended_10x10_environment_with_7_passengers():
    env = TaxiEnv(size=10,
                  n_passengers=7,
                  exploring_starts=False,
                  terminate_on_goal=True,
                  depot_dropoff_only=False,
                  image_observations=True)
    env.reset()
    expert = TaxiExpert(env)
    n_steps = 0
    while n_steps < 1000:
        action = expert.act()
        ob, reward, terminal, truncated, info = env.step(action)
        n_steps += 1
        if terminal:
            break
    assert terminal == True
    assert reward == 1
