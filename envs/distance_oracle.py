import numpy as np

from visgrid.agents.expert import gridworld_expert

class DistanceOracle:
    def __init__(self, env):
        self.env = env
        states = np.indices((env._rows, env._cols)).T.reshape(-1, 2)
        for s in states:
            for sp in states:
                # Pre-compute all pairwise distances
                gridworld_expert.GoToGridPosition(env, s, sp)

    def pairwise_distances(self, indices, s0, s1):
        init_states = s0[indices]
        next_states = s1[indices]

        distances = [
            gridworld_expert.GoToGridPosition(self.env, s, sp)[1]
            for s, sp in zip(init_states, next_states)
        ]

        return distances

#%%
if __name__ == '__main__':
    import seeding
    import numpy as np
    import random

    from visgrid.envs import GridWorld, MazeWorld, SpiralWorld
    from visgrid.envs import grid
    import matplotlib.pyplot as plt

    grid.directions[3]

    seeding.seed(0, np, random)
    env = SpiralWorld(rows=6, cols=6)
    env.plot()

    oracle = DistanceOracle(env)

    distances = [v[-1] for k, v in env.saved_directions.items()]

    plt.hist(distances, bins=36)
    plt.show()
