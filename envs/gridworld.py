import random

import numpy as np
import matplotlib.pyplot as plt

from .components import grid
from .components.agent import Agent
from .components.depot import Depot

class GridWorld:
    # Offsets:
    LEFT = np.asarray((0, -1))
    RIGHT = np.asarray((0, 1))
    UP = np.asarray((-1, 0))
    DOWN = np.asarray((1, 0))

    _action_offsets = {
        0: LEFT,
        1: RIGHT,
        2: UP,
        3: DOWN,
    }

    _action_ids = {
        tuple(LEFT): 0,
        tuple(RIGHT): 1,
        tuple(UP): 2,
        tuple(DOWN): 3,
    }

    def __init__(self, rows, cols):
        self.grid = grid.Grid(rows, cols)
        self.agent = Agent()
        self.actions = [i for i in range(4)]
        self.agent.position = np.asarray((0, 0), dtype=int)
        self.goal = None

    @property
    def rows(self):
        return self.grid._rows

    @property
    def cols(self):
        return self.grid._cols

    def reset_agent(self):
        self.agent.position = self.grid.get_random_position()
        at = lambda x, y: np.all(x.position == y.position)
        while (self.goal is not None) and at(self.agent, self.goal):
            self.agent.position = self.grid.get_random_position()

    def reset_goal(self):
        if self.goal is None:
            self.goal = Depot()
        self.goal.position = self.grid.get_random_position()
        self.reset_agent()

    def check_goal(self):
        return np.all(self.agent.position == self.goal.position)

    def step(self, action):
        assert (action in range(4))
        direction = self._action_offsets[action]
        if not self.grid.has_wall(self.agent.position, direction):
            self.agent.position += direction
        s = self.get_state()
        if self.goal:
            at_goal = self.check_goal(s)
            r = 0 if at_goal else -1
            done = True if at_goal else False
        else:
            r = 0
            done = False
        return s, r, done

    def can_run(self, action):
        assert (action in range(4))
        direction = self._action_offsets[action]
        return False if self.grid.has_wall(self.agent.position, direction) else True

    def get_state(self):
        return np.copy(self.agent.position)

    def plot(self, ax=None, draw_bg_grid=True, linewidth_multiplier=1.0, plot_goal=True):
        ax = self.grid.plot(ax,
                            draw_bg_grid=draw_bg_grid,
                            linewidth_multiplier=linewidth_multiplier)
        if self.agent:
            self.agent.plot(ax, linewidth_multiplier=linewidth_multiplier)
        if self.goal and plot_goal:
            self.goal.plot(ax, linewidth_multiplier=linewidth_multiplier)
        return ax

class TestWorld(GridWorld):
    def __init__(self):
        super().__init__(rows=3, cols=4)
        self._grid[1, 4] = 1
        self._grid[2, 3] = 1
        self._grid[3, 2] = 1
        self._grid[5, 4] = 1
        self._grid[4, 7] = 1

        # Should look roughly like this:
        # _______
        #|  _|   |
        #| |    _|
        #|___|___|

class RingWorld(GridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for r in range(self.rows - 2):
            self._grid[2 * r + 3, 2] = 1
            self._grid[2 * r + 3, 2 * self.cols - 2] = 1
        for c in range(self.cols - 2):
            self._grid[2, 2 * c + 3] = 1
            self._grid[2 * self.rows - 2, 2 * c + 3] = 1

class SnakeWorld(GridWorld):
    def __init__(self):
        super().__init__(rows=3, cols=4)
        self._grid[1, 4] = 1
        self._grid[2, 3] = 1
        self._grid[2, 5] = 1
        self._grid[3, 2] = 1
        self._grid[3, 6] = 1
        self._grid[5, 4] = 1

        # Should look roughly like this:
        # _______
        #|  _|_  |
        #| |   | |
        #|___|___|

class MazeWorld(GridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        walls = []
        for row in range(0, self.rows):
            for col in range(0, self.cols):
                #add vertical walls
                self._grid[row * 2 + 2, col * 2 + 1] = 1
                walls.append((row * 2 + 2, col * 2 + 1))

                #add horizontal walls
                self._grid[row * 2 + 1, col * 2 + 2] = 1
                walls.append((row * 2 + 1, col * 2 + 2))

        random.shuffle(walls)

        cells = []
        #add each cell as a set_text
        for row in range(0, self.rows):
            for col in range(0, self.cols):
                cells.append({(row * 2 + 1, col * 2 + 1)})

        #Randomized Kruskal's Algorithm
        for wall in walls:
            if (wall[0] % 2 == 0):

                def neighbor(set):
                    for x in set:
                        if (x[0] == wall[0] + 1 and x[1] == wall[1]):
                            return True
                        if (x[0] == wall[0] - 1 and x[1] == wall[1]):
                            return True
                    return False

                neighbors = list(filter(neighbor, cells))
                if (len(neighbors) == 1):
                    continue
                cellSet = neighbors[0].union(neighbors[1])
                cells.remove(neighbors[0])
                cells.remove(neighbors[1])
                cells.append(cellSet)
                self._grid[wall[0], wall[1]] = 0
            else:

                def neighbor(set):
                    for x in set:
                        if (x[0] == wall[0] and x[1] == wall[1] + 1):
                            return True
                        if (x[0] == wall[0] and x[1] == wall[1] - 1):
                            return True
                    return False

                neighbors = list(filter(neighbor, cells))
                if (len(neighbors) == 1):
                    continue
                cellSet = neighbors[0].union(neighbors[1])
                cells.remove(neighbors[0])
                cells.remove(neighbors[1])
                cells.append(cellSet)
                self._grid[wall[0], wall[1]] = 0

    @classmethod
    def load_maze(cls, rows, cols, seed):
        env = GridWorld(rows=rows, cols=cols)
        maze_file = 'gridworlds/gridworld/mazes/mazes_{rows}x{cols}/seed-{seed:03d}/maze-{seed}.txt'.format(
            rows=rows, cols=cols, seed=seed)
        try:
            env.load(maze_file)
        except IOError as e:
            print()
            print(
                'Could not find standardized {rows}x{cols} maze file for seed {seed}. Maybe it needs to be generated?'
                .format(rows=rows, cols=cols, seed=seed))
            print()
            raise e
        return env

class SpiralWorld(GridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add all walls
        for row in range(0, self.rows):
            for col in range(0, self.cols):
                #add vertical walls
                self._grid[row * 2 + 2, col * 2 + 1] = 1

                #add horizontal walls
                self._grid[row * 2 + 1, col * 2 + 2] = 1

        # Check dimensions to decide on appropriate spiral direction
        if self.cols > self.rows:
            direction = 'cw'
        else:
            direction = 'ccw'

        # Remove walls to build spiral
        for i in range(0, min(self.rows, self.cols)):
            # Create concentric hooks, and connect them after the first to build spiral
            if direction == 'ccw':
                self._grid[(2 * i + 1):-(2 * i + 1), (2 * i + 1)] = 0
                self._grid[-(2 * i + 2), (2 * i + 1):-(2 * i + 1)] = 0
                self._grid[(2 * i + 1):-(2 * i + 1), -(2 * i + 2)] = 0
                self._grid[(2 * i + 1), (2 * i + 3):-(2 * i + 1)] = 0
                if i > 0:
                    self._grid[2 * i, 2 * i + 1] = 0

            else:
                self._grid[(2 * i + 1), (2 * i + 1):-(2 * i + 1)] = 0
                self._grid[(2 * i + 1):-(2 * i + 1), -(2 * i + 2)] = 0
                self._grid[-(2 * i + 2), (2 * i + 1):-(2 * i + 1)] = 0
                self._grid[(2 * i + 3):-(2 * i + 1), (2 * i + 1)] = 0
                if i > 0:
                    self._grid[2 * i + 1, 2 * i] = 0

class LoopWorld(SpiralWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check dimensions to decide on appropriate spiral direction
        if self.cols > self.rows:
            direction = 'cw'
        else:
            direction = 'ccw'

        if direction == 'ccw':
            self._grid[-3, -4] = 0
        else:
            self._grid[-4, -3] = 0
