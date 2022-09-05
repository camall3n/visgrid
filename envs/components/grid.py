import random

import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, rows, cols):
        self._rows = rows
        self._cols = cols

        self.wall_color = 'C0'

        # Add rows and columns for walls between cells
        self._grid = np.ones([rows * 2 + 1, cols * 2 + 1], dtype=int)

        # Reset valid positions and walls
        self._grid[1:-1:2, 1:-1] = 0
        self._grid[1:-1, 1:-1:2] = 0

    def __getitem__(self, key):
        return self._grid[key]

    def __setitem__(self, key, value):
        self._grid[key] = value

    def __delitem__(self, _):
        raise NotImplementedError(
            f'Deleting items from a {self.__classname__} instance is not supported')

    @property
    def shape(self):
        return self._grid.shape

    def get_random_position(self):
        cells = self._grid[:, 1::2][1::2, :]
        nonzero_indices = np.stack(np.nonzero(1 - cells)).T
        return random.choice(nonzero_indices)

    def has_wall(self, position, offset):
        row, col = position
        d_row, d_col = offset
        wall_row = 2 * row + 1 + d_row
        wall_col = 2 * col + 1 + d_col
        return self[wall_row, wall_col]

    def render(self, cell_width, wall_width) -> np.ndarray:
        grid = self._grid
        for row_or_col_axis in [0, 1]:
            slices = np.split(grid, grid.shape[row_or_col_axis], axis=row_or_col_axis)
            walls = slices[0::2]
            cells = slices[1::2]
            walls = [np.repeat(wall, wall_width, axis=row_or_col_axis) for wall in walls]
            cells = [np.repeat(cell, cell_width, axis=row_or_col_axis) for cell in cells]
            slices = [item for pair in zip(walls, cells) for item in pair] + [walls[-1]]
            grid = np.concatenate(slices, axis=row_or_col_axis).astype(float)
        return grid

    def save(self, filename):
        np.savetxt(filename, self._grid.astype(int), fmt='%1d')

    @classmethod
    def from_file(self, filename):
        grid = np.loadtxt(filename, dtype=int)
        r, c = grid.shape
        self._rows = r // 2
        self._cols = c // 2
        self._grid = grid

    @classmethod
    def generate_ring(cls, rows, cols):
        grid = cls(rows, cols)
        for r in range(rows - 2):
            grid[2 * r + 3, 2] = 1
            grid[2 * r + 3, 2 * cols - 2] = 1
        for c in range(cols - 2):
            grid[2, 2 * c + 3] = 1
            grid[2 * rows - 2, 2 * c + 3] = 1
        return grid

    @classmethod
    def generate_spiral(cls, rows, cols):
        grid = cls(rows, cols)

        # Add all walls
        for row in range(0, rows):
            for col in range(0, cols):
                #add vertical walls
                grid[row * 2 + 2, col * 2 + 1] = 1

                #add horizontal walls
                grid[row * 2 + 1, col * 2 + 2] = 1

        # Check dimensions to decide on appropriate spiral direction
        if cols > rows:
            direction = 'cw'
        else:
            direction = 'ccw'

        # Remove walls to build spiral
        for i in range(0, min(rows, cols)):
            # Create concentric hooks, and connect them after the first to build spiral
            if direction == 'ccw':
                grid[(2 * i + 1):-(2 * i + 1), (2 * i + 1)] = 0
                grid[-(2 * i + 2), (2 * i + 1):-(2 * i + 1)] = 0
                grid[(2 * i + 1):-(2 * i + 1), -(2 * i + 2)] = 0
                grid[(2 * i + 1), (2 * i + 3):-(2 * i + 1)] = 0
                if i > 0:
                    grid[2 * i, 2 * i + 1] = 0

            else:
                grid[(2 * i + 1), (2 * i + 1):-(2 * i + 1)] = 0
                grid[(2 * i + 1):-(2 * i + 1), -(2 * i + 2)] = 0
                grid[-(2 * i + 2), (2 * i + 1):-(2 * i + 1)] = 0
                grid[(2 * i + 3):-(2 * i + 1), (2 * i + 1)] = 0
                if i > 0:
                    grid[2 * i + 1, 2 * i] = 0

        return grid

    @classmethod
    def generate_spiral_with_shortcut(cls, rows, cols):
        grid = cls.generate_spiral(rows, cols)

        # Check dimensions to decide on appropriate shortcut location
        if cols <= rows:
            grid[-3, -4] = 0
        else:
            grid[-4, -3] = 0

        return grid

    @classmethod
    def generate_maze(cls, rows, cols):
        grid = cls(rows, cols)
        walls = []
        for row in range(0, rows):
            for col in range(0, cols):
                #add vertical walls
                grid[row * 2 + 2, col * 2 + 1] = 1
                walls.append((row * 2 + 2, col * 2 + 1))

                #add horizontal walls
                grid[row * 2 + 1, col * 2 + 2] = 1
                walls.append((row * 2 + 1, col * 2 + 2))

        random.shuffle(walls)

        cells = []
        #add each cell as a set_text
        for row in range(0, rows):
            for col in range(0, cols):
                cells.append({(row * 2 + 1, col * 2 + 1)})

        #Randomized Kruskal's Algorithm
        for wall in walls:

            def neighbor(set):
                for x in set:
                    if (wall[0] % 2 == 0):
                        if (x[0] == wall[0] + 1 and x[1] == wall[1]):
                            return True
                        if (x[0] == wall[0] - 1 and x[1] == wall[1]):
                            return True
                    else:
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
            grid[wall[0], wall[1]] = 0

        return grid