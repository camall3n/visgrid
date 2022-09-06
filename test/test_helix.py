import pytest
import matplotlib.pyplot as plt
import numpy as np

from visgrid.wrappers.helix import HelixWrapper
from visgrid.envs.point import PointEnv

def get_curves():
    ndim = 5
    n_points = 10000
    n_rotations = 4
    action = np.zeros(ndim)
    action[2] = 1 / n_points

    line_env = PointEnv(ndim)
    helix1_env = HelixWrapper(PointEnv(ndim), rotations_per_unit_z=n_rotations)
    helix2_env = HelixWrapper(HelixWrapper(PointEnv(ndim)),
                              axes_xy=(1, 2),
                              axis_z=0,
                              rotations_per_unit_z=n_rotations)
    helix3_env = HelixWrapper(
        HelixWrapper(HelixWrapper(PointEnv(ndim)),
                     axes_xy=(1, 2),
                     axis_z=0,
                     rotations_per_unit_z=n_rotations),
        axes_xy=(2, 0),
        axis_z=1,
        rotations_per_unit_z=n_rotations / 10,
    )

    def get_points(env):
        env.reset(x=np.array([0.5, 0.0, -0.5, 1.0, 1.0]))
        obs = []
        for _ in range(n_points + 1):
            obs.append(env.step(action)[0])
        obs = np.stack(obs)
        return obs

    line_points = get_points(line_env)
    helix1_points = get_points(helix1_env)
    helix2_points = get_points(helix2_env)
    helix3_points = get_points(helix3_env)

    return line_points, helix1_points, helix2_points, helix3_points

def visualize_points():
    line_points, helix1_points, helix2_points, helix3_points = get_curves()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(*line_points[:, :3].T, 'k', label='0')
    ax.plot3D(*helix1_points[:, :3].T, 'r', label='1')
    ax.plot3D(*helix2_points[:, :3].T, 'b', label='2')
    ax.plot3D(*helix3_points[:, :3].T, 'c', label='3')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title('Increasing numbers of helix transformations')
    plt.show()

@pytest.fixture
def precompute_curves():
    return get_curves()

@pytest.fixture
def precompute_line(precompute_curves):
    return precompute_curves[0]

@pytest.fixture
def precompute_helix1(precompute_curves):
    return precompute_curves[1]

@pytest.fixture
def precompute_helix3(precompute_curves):
    return precompute_curves[3]

def test_1dim_line(precompute_line):
    line_points = precompute_line
    line_std = np.std(line_points, axis=0)
    for i in [0, 1, 3, 4]:
        assert line_std[i] == 0.0
    assert line_std[2] > 0.1

def test_helix1_shape(precompute_helix1):
    helix1_points = precompute_helix1
    assert np.allclose(helix1_points[0, :2], helix1_points[-1, :2])
    helix_delta = helix1_points[-1, 2] - helix1_points[0, 2]
    assert np.isclose(helix_delta, 1.0)

def test_unmodified_dims(precompute_line, precompute_helix3):
    line_points = precompute_line
    helix3_points = precompute_helix3
    assert np.allclose(line_points[:, 3:], np.ones_like(line_points[:, 3:]))
    assert np.allclose(helix3_points[:, 3:], line_points[:, 3:])

if __name__ == '__main__':
    visualize_points()
