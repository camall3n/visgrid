import warnings
from typing import Tuple, Optional

from cv2 import resize, INTER_AREA, INTER_LINEAR
import gym
from gym import spaces
from gym.core import ObsType, ActType
import matplotlib.pyplot as plt
import numpy as np

from .components import Grid, Agent, Depot
from .. import utils
from ..wrappers.sensors import Sensor

class GridworldEnv(gym.Env):
    # Offsets:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    _action_ids = {
        (0, -1): LEFT,
        (0, 1): RIGHT,
        (-1, 0): UP,
        (1, 0): DOWN,
    }
    _action_offsets = {action_id: np.array(offset) for offset, action_id in _action_ids.items()}

    dimensions_6x6_to_64x64 = {
        'wall_width': 1,
        'cell_width': 9,
        'character_width': 5,
        'depot_width': 2,
        'border_widths': (2, 1),
        'img_shape': (64, 64),
    }
    _default_dimensions = dimensions_6x6_to_64x64

    def __init__(self,
                 rows: int,
                 cols: int,
                 grid: Grid = None,
                 exploring_starts: bool = True,
                 terminate_on_goal: bool = True,
                 fixed_goal: bool = True,
                 hidden_goal: bool = False,
                 agent_position: Tuple = None,
                 goal_position: Tuple = None,
                 image_observations: bool = True,
                 sensor: Sensor = None,
                 dimensions: dict = None):
        """
        Visual gridworld environment

        rows, cols: the shape of the gridworld
        exploring_starts:
            True: agent starting position is sampled uniformly from all non-goal positions
                   (ignored if goal_position is provided)
            False: agent returns to its initial position on reset
                   (automatically enabled if goal_position is provided)
        terminate_on_goal:
            True: reaching the goal produces a terminal state and a reward
            False: the goal has no special significance and the episode simply continues
        fixed_goal:
            True: the goal stays the same after each episode
                   (automatically enabled if goal_position is provided)
            False: the goal resets to a randomly chosen location after each episode
                   (ignored if goal_position is provided)
        hidden_goal:
            True: goal information is included in the observations
            False: goal information is removed from the observations
        agent_position: position for the agent (disables exploring_starts)
        goal_position: position for the goal (disables fixed_goal)
        image_observations:
            True: Observations are images
            False: Observations use internal state vector
        sensor: (deprecated) an operation (or chain of operations) to apply after generating
            each observation
        dimensions: dictionary of size information for rendering
        """
        self.grid = Grid(rows, cols) if grid is None else grid
        assert rows == self.grid._rows and cols == self.grid._cols
        self.exploring_starts = exploring_starts if agent_position is None else False
        self.fixed_goal = fixed_goal if goal_position is None else True
        self.hidden_goal = hidden_goal
        self.terminate_on_goal = terminate_on_goal
        self.image_observations = image_observations
        self.sensor = sensor if sensor is not None else Sensor()
        self.dimensions = dimensions if dimensions is not None else self._default_dimensions

        self._initialize_agent(agent_position)
        self._initialize_depots(goal_position)

        self.action_space = spaces.Discrete(4)
        self._initialize_state_space()
        self._initialize_obs_space()

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------

    def _initialize_state_space(self):
        factor_sizes = (self.rows, self.cols, self.rows, self.cols)
        self.state_space = spaces.MultiDiscrete(factor_sizes, dtype=int)

    def _initialize_obs_space(self):
        if self.image_observations:
            img_shape = self.dimensions['img_shape'] + (3, )
            self.observation_space = spaces.Box(0.0, 1.0, img_shape, dtype=np.float32)
        else:
            obs_shape = self.state_space.shape
            if self.hidden_goal:
                obs_shape = obs_shape[:2]
            self.observation_space = spaces.MultiDiscrete(obs_shape, dtype=int)

    def _initialize_agent(self, position=None):
        if position is None:
            position = self.random_grid_position()
        self.agent = Agent(position)
        self._agent_initial_position = self.agent.position.copy()

    def _initialize_depots(self, position=None):
        if position is None:
            position = self.random_grid_position()
        self.goal = Depot(position, color='red')
        self.depots = {'red': self.goal}

    # ------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------

    @classmethod
    def from_grid(cls, grid: np.ndarray, *args, **kw):
        env = cls(grid._rows, grid._cols, grid=grid, *args, **kw)
        return env

    @classmethod
    def from_file(cls, filename: str, *args, **kw):
        try:
            grid = Grid.from_file(filename)
        except IOError as e:
            print(f'Grid file not found: {filename}')
            raise e
        return cls.from_grid(grid, *args, **kw)

    @classmethod
    def from_saved_maze(cls, rows: int, cols: int, seed: int, *args, **kw):
        maze_file = f'visgrid/envs/saved/mazes_{rows}x{cols}/seed-{seed:03d}/maze-{seed}.txt'
        return cls.from_file(maze_file, *args, **kw)

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
    # Helpers
    # ------------------------------------------------------------

    def random_grid_position(self):
        return self.grid.get_random_position(self.np_random)

    # ------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------

    def _reset(self):
        if not self.fixed_goal:
            self.goal.position = self.random_grid_position()

        if self.exploring_starts:
            while True:
                self.agent.position = self.random_grid_position()
                if not (self.terminate_on_goal and self._check_goal()):
                    break
        else:
            self.agent.position = self._agent_initial_position

    def reset(self, seed: Optional[int] = None) -> Tuple[ObsType, dict]:
        """
        Reset the environment. Must be called before calling step().

        returns: observation, info
        """
        super().reset(seed=seed)
        self._cached_state = None
        self._cached_render = None
        self._reset()
        state = self.get_state()
        ob = self.get_observation(state)
        info = self._get_info(state)
        return ob, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """
        Execute action if it can run, then return the corresponding effects

        returns: observation, reward, terminated, truncated, info
        """
        self._cached_state = None
        self._cached_render = None
        if self.can_run(action):
            self._step(action)

        state = self.get_state()
        if self.terminate_on_goal and self._check_goal(state):
            terminated = True
        else:
            terminated = False
        reward = 1 if terminated else 0
        ob = self.get_observation(state)
        info = self._get_info(state)
        return ob, reward, terminated, False, info

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
            state = self._render(state)
        elif self.hidden_goal:
            state = state[:2]
        return self.sensor(state)

    def _get_info(self, state=None):
        if state is None:
            state = self.get_state()
        info = {
            'state': state,
        }
        return info

    def _check_goal(self, state=None):
        if state is None:
            state = self.get_state()
        for depot in self.depots.values():
            if np.all(state[:2] == depot.position):
                return True
        return False

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------

    def plot(self):
        ob = self.get_observation()
        plt.imshow(ob)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def _render(self, state: Optional[Tuple] = None) -> np.ndarray:
        current_state = self.get_state()
        try:
            if state is not None:
                self.set_state(state)
            # only render observation once per step
            if (self._cached_state is None) or (state != self._cached_state).any():
                self._cached_state = state
                self._cached_render = self._do_render()
            return self._cached_render
        finally:
            self.set_state(current_state)

    def _do_render(self) -> np.ndarray:
        objects = self._render_objects()
        foreground = sum(objects.values())
        background = np.ones_like(foreground) * utils.get_rgb('white')
        fg_indices = np.any(foreground > 0, axis=-1)
        background[fg_indices, :] = 0

        content = background + foreground
        frame = self._render_frame(content)
        image = self._render_composite_image(frame, content)

        image = self._resize_if_necessary(image, self.dimensions['img_shape'])
        return image

    def _resize_if_necessary(self, image, desired_shape):
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
        walls = utils.to_rgb(walls, 'almost black')

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
