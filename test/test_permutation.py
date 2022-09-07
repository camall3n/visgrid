import pytest

import copy

import numpy as np

from visgrid.wrappers.permutation import PermuteFactorsWrapper
from visgrid.envs import TaxiEnv
#%%
def test_permutation_indexing():
    ob = np.arange(1, 6)
    ob0 = np.zeros(5, dtype=int)
    ob1 = np.ones(5, dtype=int)
    obs = [ob, ob0, ob1]

    p = np.array([4, 2, 0, 5, 3, 1])
    q = np.array([0, 1, 2, 3, 4, 5])
    r = np.array([5, 4, 3, 2, 1, 0])
    s = np.array([2, 1, 3, 4, 5, 0])
    t = np.array([2, 4, 5, 1, 3, 0])
    perms = np.stack([p, q, r, s, t])

    p_ob = np.array([2, 2, 2, 5, 0])
    p_ob0 = np.array([4, 0, 5, 2, 2])
    p_ob1 = np.array([2, 1, 4, 1, 4])
    p_obs = [p_ob, p_ob0, p_ob1]

    for ob, p_ob in zip(obs, p_obs):
        assert np.all(perms[np.arange(5), ob] == p_ob)

def test_variable_sizes():
    factor_sizes = (6, 2, 4)
    N = 10
    obs = np.random.randint(factor_sizes, size=(N,3))
    obs[0] = np.array([factor_sizes]) - 1

    perms = [np.random.permutation(fac_sz) for fac_sz in factor_sizes]
    p_obs = np.stack([[p[ob[i]] for i, p in enumerate(perms)] for ob in obs])
    assert tuple(p_obs[0]) == tuple([p[-1] for p in perms])

@pytest.fixture
def env():
    env = TaxiEnv(
        size=5,
        n_passengers=1,
        exploring_starts=True,
        terminate_on_goal=False,
        image_observations=False,
    )
    return env

@pytest.fixture
def w_env(env):
    env = copy.deepcopy(env)
    return PermuteFactorsWrapper(env)

def test_same_initial_state(env, w_env):
    env.reset(seed=1)
    w_env.reset(seed=1)
    assert np.all(env.get_state() == w_env.unwrapped.get_state())

def test_same_subsequent_states(env, w_env):
    env.reset(seed=1)
    w_env.reset(seed=1)
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)
        w_env.step(action)
        assert np.all(env.get_state() == w_env.unwrapped.get_state())

def test_same_reset_states(env, w_env):
    env.reset(seed=1)
    w_env.reset(seed=1)
    for _ in range(100):
        env.reset()
        w_env.reset()
        assert np.all(env.get_state() == w_env.unwrapped.get_state())

def test_same_obs_shape(env, w_env):
    ob = env.reset(seed=1)[0]
    w_ob = w_env.reset(seed=1)[0]
    assert ob.shape == w_ob.shape

def test_same_action_selections(env, w_env):
    env.reset(seed=1)
    w_env.reset(seed=1)
    env.action_space.seed(1)
    w_env.action_space.seed(1)
    for _ in range(100):
        action = env.action_space.sample()
        w_action = w_env.action_space.sample()
        assert action == w_action

def test_same_unchanged_factor_indices(env, w_env):
    env.reset(seed=1)
    w_env.reset(seed=1)
    for ep in range(100):
        ob = env.reset()[0]
        w_ob = w_env.reset()[0]
        prev_ob, prev_w_ob = ob, w_ob
        for step in range(100):
            action = env.action_space.sample()
            ob = env.step(action)[0]
            w_ob = w_env.step(action)[0]
            unchanged = np.where(ob == prev_ob)
            assert np.all(w_ob[unchanged] == prev_w_ob[unchanged])
            prev_ob, prev_w_ob = ob, w_ob

def test_same_obs_range(env, w_env):
    env.reset(seed=1)
    w_env.reset(seed=1)
    obs = []
    w_obs = []
    for ep in range(100):
        obs.append(env.reset()[0])
        w_obs.append(w_env.reset()[0])
        for step in range(100):
            action = env.action_space.sample()
            obs.append(env.step(action)[0])
            w_obs.append(w_env.step(action)[0])
    obs = np.stack(obs)
    w_obs = np.stack(w_obs)

    assert np.min(obs) == np.min(w_obs)
    assert np.max(obs) == np.max(w_obs)

# env = env()
# test_permute_factors(env, w_env(env))
