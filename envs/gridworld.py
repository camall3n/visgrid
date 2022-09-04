import warnings
from typing import Tuple

from cv2 import resize, INTER_AREA, INTER_LINEAR
import matplotlib.pyplot as plt
import numpy as np

from .components.grid import Grid
from .components.agent import Agent
from .components.depot import Depot
from .. import utils
from ..sensors import Sensor

class GridworldEnv:
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

    _default_dimensions = {
        'wall_width': 4,
        'cell_width': 24,
        'character_width': 16,
        'depot_width': 8,
        'border_widths': (0, 0),
        'img_shape': (64, 64),
    }

    def __init__(self,
                 rows: int,
                 cols: int,
                 exploring_starts: bool = True,
                 terminate_on_goal: bool = True,
                 fixed_goal: bool = True,
                 hidden_goal: bool = False,
                 agent_position: Tuple = None,
                 goal_position: Tuple = None,
                 image_observations: bool = True,
                 sensor: Sensor = None,
                 dimensions: dict = None):
        self.grid = Grid(rows, cols)
        self.exploring_starts = exploring_starts
        self.fixed_goal = fixed_goal
        self.hidden_goal = hidden_goal
        self.terminate_on_goal = terminate_on_goal
        self.image_observations = image_observations
        self.sensor = sensor if sensor is not None else Sensor()
        self.dimensions = dimensions if dimensions is not None else self._default_dimensions
        self.actions = [i for i in range(4)]
        self._initialize_agent(agent_position)
        self._initialize_depots(goal_position)

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------

    def _initialize_agent(self, position=None):
        if position is None:
            position = self.grid.get_random_position()
        self.agent = Agent(position)
        self._agent_initial_position = self.agent.position.copy()

    def _initialize_depots(self, position=None):
        if position is None:
            position = self.grid.get_random_position()
        self.goal = Depot(position, color='red')
        self.depots = {'red': self.goal}

    # ------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------

    @classmethod
    def from_grid(cls, grid: np.ndarray, dimensions: dict = None):
        env = cls(grid.rows, grid.cols, dimensions)
        env.grid = grid
        return env

    @classmethod
    def from_file(cls, filename: str, dimensions: dict = None):
        try:
            grid = Grid.from_file(filename)
        except IOError as e:
            print(f'Grid file not found: {filename}')
            raise e
        return cls.from_grid(grid, dimensions)

    @classmethod
    def from_saved_maze(cls, rows: int, cols: int, seed: int, dimensions: dict = None):
        maze_file = f'visgrid/envs/saved/mazes_{rows}x{cols}/seed-{seed:03d}/maze-{seed}.txt'
        return cls.from_file(maze_file, dimensions)

    # ------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------

    @property
    def rows(self):
        return self.grid._rows

    @property
    def cols(self):
        return self.grid._cols

    # ------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------

    def _reset(self):
        if not self.fixed_goal:
            self.goal.position = self.grid.get_random_position()

        if self.exploring_starts:
            while True:
                self.agent.position = self.grid.get_random_position()
                if not (self.terminate_on_goal and self._check_goal()):
                    break
        else:
            self.agent.position = self._agent_initial_position

    def reset(self):
        self._reset()
        ob = self.get_observation(self.get_state())
        return ob

    def step(self, action):
        """
        Execute action if it can run, then return the corresponding effects
        """
        if self.can_run(action):
            self._step(action)

        state = self.get_state()
        if self.terminate_on_goal and self._check_goal(state):
            done = True
        else:
            done = False
        reward = 1 if done else 0
        info = {'state': state}
        ob = self.get_observation(state)
        return ob, reward, done, info

    def _step(self, action):
        """
        Execute action without checking if it can run
        """
        offset = self._action_offsets[action]
        self.agent.position += offset

    def can_run(self, action):
        assert (action in range(4))
        direction = self._action_offsets[action]
        return False if self.grid.has_wall(self.agent.position, direction) else True

    def get_state(self):
        row, col = self.agent.position
        goal_row, goal_col = self.goal.position
        state = [row, col, goal_row, goal_col]
        return np.asarray(state, dtype=int)

    def set_state(self, state):
        row, col, *remaining = state
        self.agent.position = row, col
        if remaining:
            goal_row, goal_col = remaining
            self.goal.position = goal_row, goal_col

    def get_observation(self, state=None):
        if state is None:
            state = self.get_state()
        if self.image_observations:
            state = self.render(state)
        elif self.hidden_goal:
            state = state[:2]
        return self.sensor(state)

    def _check_goal(self, state=None):
        if state is None:
            state = self.get_state()
        for depot in self.depots:
            if np.all(state[:2] == depot.position):
                return True
        return False

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------

    def plot(self):
        ob = self.render()
        plt.imshow(ob)

    def render(self, state=None) -> np.ndarray:
        current_state = self.get_state()
        if state is not None:
            self.set_state(state)
        return self._render()

    def _render(self) -> np.ndarray:
        objects = self._render_objects()
        foreground = sum(objects.values())
        background = np.ones_like(foreground) * utils.get_rgb('white')
        fg_indices = np.any(foreground > 0, axis=-1)
        background[fg_indices, :] = 0

        content = background + foreground
        frame = self._render_frame(content)
        image = self._render_composite_image(frame, content)

        desired_shape = self.dimensions['img_shape']
        current_shape = image.shape[:2]
        if current_shape != desired_shape:
            warnings.warn(f'Resizing image from {current_shape} to desired {desired_shape}',
                          RuntimeWarning)
            if np.prod(desired_shape) < np.prod(current_shape):
                interp_mode = INTER_AREA  # downscale
            else:
                interp_mode = INTER_LINEAR  # upscale
            image = resize(image, desired_shape, interpolation=interp_mode)
            image = np.clip(image, 0, 1)
        return image

    def _render_composite_image(self, frame: np.ndarray, content: np.ndarray) -> np.ndarray:
        """Generate a border to reflect the current in_taxi status"""
        pad_top_left, pad_bot_right = self.dimensions['border_widths']
        H, W, *_ = frame.shape
        frame[pad_top_left:(H - pad_bot_right), pad_top_left:(W - pad_bot_right), :] = content
        return frame

    def _render_frame(self, content) -> np.ndarray:
        frame = np.ones_like(content) * utils.get_rgb('white')
        pad_top_left, pad_bot_right = self.dimensions['border_widths']
        pad_width = ((pad_top_left, pad_bot_right), (pad_top_left, pad_bot_right), (0, 0))
        frame = np.pad(frame, pad_width=pad_width, mode='edge')
        return frame

    def _render_objects(self) -> dict:
        walls = self.grid.render(cell_width=self.dimensions['cell_width'],
                                 wall_width=self.dimensions['wall_width'])
        walls = utils.to_rgb(walls, 'dimgray') / 8

        depot_patches = np.zeros_like(walls)
        for depot in self.depots.values():
            patch = self._render_depot_patch(depot.color)
            self._add_patch(depot_patches, patch, depot.position)

        agent_patches = np.zeros_like(walls)
        patch = self._render_character_patch()
        self._add_patch(agent_patches, patch, self.agent.position)

        objects = {
            'walls': walls,
            'depots': depot_patches,
            'agent': agent_patches,
        }
        if self.hidden_goal:
            del objects['depots']

        return objects

    def _render_depot_patch(self, color) -> np.ndarray:
        """Generate a patch representing a depot"""
        cell_width = self.dimensions['cell_width']
        depot_width = self.dimensions['depot_width']
        assert depot_width <= cell_width // 2
        sw_patch = np.tri(cell_width // 2, k=(depot_width - cell_width // 2))
        nw_patch = np.flipud(sw_patch)
        ne_patch = np.fliplr(nw_patch)
        se_patch = np.fliplr(sw_patch)

        patch = np.block([[nw_patch, ne_patch], [sw_patch, se_patch]])

        # add center row / column for odd widths
        if cell_width % 2 == 1:
            patch = np.insert(patch, cell_width // 2, 0, axis=0)
            patch = np.insert(patch, cell_width // 2, 0, axis=1)

        patch[0, :] = 1
        patch[:, 0] = 1
        patch[-1, :] = 1
        patch[:, -1] = 1

        patch = utils.to_rgb(patch, color)
        return patch

    def _render_character_patch(self, color='red') -> np.ndarray:
        """Generate a patch representing a character"""
        cell_width = self.dimensions['cell_width']
        agent_width = self.dimensions['character_width']
        assert agent_width <= cell_width
        sw_bg = np.tri(cell_width // 2, k=(cell_width // 2 - agent_width // 2 - 2))
        nw_bg = np.flipud(sw_bg)
        ne_bg = np.fliplr(nw_bg)
        se_bg = np.fliplr(sw_bg)

        bg = np.block([[nw_bg, ne_bg], [sw_bg, se_bg]])

        # add center row / column for odd widths
        if cell_width % 2 == 1:
            bg = np.insert(bg, cell_width // 2, 0, axis=0)
            bg = np.insert(bg, cell_width // 2, 0, axis=1)

        # crop edges to a circle and invert
        excess = (cell_width - agent_width) // 2
        bg[:excess, :] = 1
        bg[:, :excess] = 1
        bg[-excess:, :] = 1
        bg[:, -excess:] = 1
        patch = (1 - bg)

        patch = utils.to_rgb(patch, color)
        return patch

    def _add_patch(self, image: np.ndarray, patch: np.ndarray, position) -> np.ndarray:
        cell_width = self.dimensions['cell_width']
        wall_width = self.dimensions['wall_width']
        row, col = self._cell_start(position, cell_width, wall_width)
        image[row:row + cell_width, col:col + cell_width, :] = patch
        return image

    @staticmethod
    def _cell_start(position, cell_width: int, wall_width: int):
        """Compute <row, col> indices of top-left pixel of cell at given position"""
        row, col = position
        row_start = wall_width + row * (cell_width + wall_width)
        col_start = wall_width + col * (cell_width + wall_width)
        return (row_start, col_start)
