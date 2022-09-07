import matplotlib.pyplot as plt
import numpy as np

from visgrid.wrappers.rotation import RotationWrapper
from visgrid.envs.point import PointEnv

def get_curves(axes=None):
    ndim = 3
    n_points = 100
    n_lines = 100
    action = np.zeros(ndim)
    action[-1] = 1 / n_points

    line_env = PointEnv(ndim)
    rot_envs = []
    for i in range(n_lines):
        rot_envs.append(RotationWrapper(PointEnv(ndim), axes=axes))

    def get_points(env):
        env.reset(x=np.array([0.0, 0.0, -0.5]))
        obs = []
        for _ in range(n_points + 1):
            obs.append(env.step(action)[0])
        obs = np.stack(obs)
        return obs

    line_points = get_points(line_env)
    rot_points = []
    for i in range(n_lines):
        rot_points.append(get_points(rot_envs[i]))

    return line_points, rot_points

def visualize_points():
    line_points, rot_points = get_curves()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(*line_points.T, 'r')
    for i in range(len(rot_points)):
        ax.plot3D(*rot_points[i].T, 'k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Random rotation transformations')
    plt.show()

def visualize_points_with_axes():
    line_points, rot_points = get_curves(axes=(1, 2))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(*line_points.T, 'r')
    for i in range(len(rot_points)):
        ax.plot3D(*rot_points[i].T, 'k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Random rotation transformations')
    plt.show()

def test_rotation_wrapper():
    line_points, rot_points = get_curves()
    line_std = np.std(line_points, axis=0)
    assert line_std[0] == 0.0 and line_std[1] == 0.0
    assert line_std[2] > 0.1

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    n_points = len(rot_points)
    for i in range(n_points):
        start = rot_points[i][0]
        end = rot_points[i][1]
        length = np.linalg.norm(end - start)
        for point in rot_points[i][1:-1]:
            seg1 = point - start
            seg2 = end - start
            sim = cosine_similarity(seg1, seg2)
            assert np.isclose(sim, 1.0)

def test_rotation_wrapper_with_axes():
    line_points, rot_points = get_curves(axes=(1, 2))
    line_std = np.std(line_points, axis=0)
    assert line_std[0] == 0.0 and line_std[1] == 0.0
    assert line_std[2] > 0.1

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    n_points = len(rot_points)
    for i in range(n_points):
        start = rot_points[i][0]
        end = rot_points[i][1]
        length = np.linalg.norm(end - start)
        for point in rot_points[i][1:-1]:
            seg1 = point - start
            seg2 = end - start
            sim = cosine_similarity(seg1, seg2)
            assert np.isclose(sim, 1.0)

            assert point[0] == 0

if __name__ == '__main__':
    visualize_points()
    visualize_points_with_axes()
    test_rotation_wrapper_with_axes()
