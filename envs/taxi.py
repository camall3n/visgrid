import copy
import random

import numpy as np
import gym
from gym import spaces

from .gridworld import GridworldEnv
from .components.passenger import Passenger
from .components.depot import Depot
from .. import utils
from ..sensors import Sensor

class TaxiEnv(GridworldEnv):
    dimensions_64x64 = {
        'wall_width': 1,
        'cell_width': 11,
        'character_width': 7,
        'depot_width': 2,
        'border_widths': (2, 1),
        'dash_widths': (4, 4),
        'img_shape': (64, 64),
    }
    dimensions_84x84 = {
        'wall_width': 2,
        'cell_width': 13,
        'character_width': 9,
        'depot_width': 3,
        'border_widths': (4, 3),
        'dash_widths': (6, 6),
        'img_shape': (84, 84),
    }
    _default_dimensions = dimensions_84x84

    def __init__(self,
                 size: int = 5,
                 n_passengers: int = 1,
                 exploring_starts: bool = True,
                 terminate_on_goal: bool = True,
                 depot_dropoff_only: bool = False,
                 image_observations: bool = True,
                 sensor: Sensor = None,
                 dimensions: dict = None):
        """
        Visual taxi environment

        Original 5x5 taxi environment adapted from:
            Dietterich, G. Thomas. "Hierarchical Reinforcement Learning
            with the MAXQ Value Function Decomposition", JAIR, 2000

        Extended 10x10 version adapted from:
            Diuk, Cohen, & Littman. "An Object-Oriented Representation
            for Efficient Reinforcement Learning", ICML, 2008

        size: {5, 10}
        n_passengers: {0..3} for size 5; {0..7} for size 10
        exploring_starts:
            True: initial state is sampled from a balanced distribution over the
                  entire state space.
            False: initial taxi/passenger positions are at random unique depots
        terminate_on_goal:
            True: reaching the goal produces a terminal state and a reward
            False: the goal has no special significance and the episode simply continues
        depot_dropoff_only:
            True: passengers can only be dropped off at (vacant) depots
            False: passengers can be dropped off anywhere in the grid
        image_observations:
            True: Observations are images
            False: Observations use internal state vector
        sensor: (deprecated) an operation (or chain of operations) to apply after generating
            each observation
        dimensions: dictionary of size information for rendering
        """
        self.size = size
        self.n_passengers = n_passengers
        self.depot_dropoff_only = depot_dropoff_only
        super().__init__(rows=size,
                         cols=size,
                         exploring_starts=exploring_starts,
                         terminate_on_goal=terminate_on_goal,
                         fixed_goal=False,
                         hidden_goal=False,
                         image_observations=image_observations,
                         sensor=sensor,
                         dimensions=dimensions)

        self.goal = None
        self._initialize_walls()
        self._initialize_passengers()

        self.action_space = spaces.Discrete(5)
        self._initialize_state_space()
        self._initialize_obs_space()

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------

    def _initialize_state_space(self):
        taxi_state_shape = (self.rows, self.cols)
        psgr_state_shape = (self.rows, self.cols, 2, len(self.depots))
        state_shape = taxi_state_shape + (psgr_state_shape * self.n_passengers)
        self.state_space = spaces.MultiDiscrete(state_shape, dtype=int)

    def _initialize_obs_space(self):
        if self.image_observations:
            img_shape = self.dimensions['img_shape'] + (3, )
            self.observation_space = spaces.Box(0.0, 1.0, img_shape, dtype=np.float32)
        else:
            obs_shape = self.state_space.shape
            self.observation_space = spaces.MultiDiscrete(obs_shape, dtype=int)

    def _initialize_depots(self):
        if self.size == 5:
            self.depot_locs = {# yapf: disable
                'red':    (0, 0),
                'yellow': (4, 0),
                'blue':   (4, 3),
                'green':  (0, 4),
            }# yapf: enable
        elif self.size == 10:
            self.depot_locs = {# yapf: disable
                'red':     (0, 0),
                'blue':    (8, 0),
                'green':   (9, 4),
                'yellow':  (0, 5),
                'gray':    (3, 3),
                'magenta': (4, 6),
                'cyan':    (0, 8),
                'orange':  (9, 9),
            }# yapf: enable
        else:
            raise NotImplementedError(
                f'Invalid size ({self.size}) provided for {self.__classname__}'
                'Valid options: {5, 10}.')

        self.depot_names = sorted(self.depot_locs.keys())
        self.depot_ids = {name: id_ for id_, name in enumerate(self.depot_names)}
        self.depots = dict()
        for name in self.depot_names:
            self.depots[name] = Depot(color=name)
            self.depots[name].position = self.depot_locs[name]

    def _initialize_walls(self):
        if self.size == 5:
            self.grid[1:4, 4] = 1
            self.grid[7:10, 2] = 1
            self.grid[7:10, 6] = 1
        elif self.size == 10:
            self.grid[1:8, 6] = 1
            self.grid[13:20, 2] = 1
            self.grid[13:20, 8] = 1
            self.grid[5:12, 12] = 1
            self.grid[1:8, 16] = 1
            self.grid[13:20, 16] = 1
        else:
            raise NotImplementedError(
                f'Invalid size ({self.size}) provided for {self.__classname__}'
                'Valid options: {5, 10}.')

    def _initialize_passengers(self):
        self.passenger = None
        max_passengers = len(self.depots) - 1
        if not (0 <= self.n_passengers <= max_passengers):
            raise ValueError(
                f"'n_passengers' ({self.n_passengers}) must be between 0 and {max_passengers}")

        self.passengers = [Passenger(color=c) for c in self.depot_names][:self.n_passengers]

    # ------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------

    def _reset(self):
        if self.exploring_starts:
            self._reset_exploring_start()
        else:
            self._reset_classic_start()

    def _reset_classic_start(self):
        # Place passengers randomly at unique depots
        starting_depots = copy.deepcopy(self.depot_names)
        random.shuffle(starting_depots)
        for i, p in enumerate(self.passengers):
            p.position = self.depots[starting_depots[i]].position
            p.in_taxi = False

        # Place taxi at a different unique depot
        self.agent.position = self.depots[starting_depots[-1]].position

        # Generate list of goal depots for the passengers
        goal_depots = copy.deepcopy(self.depot_names)
        N = self.n_passengers
        while True:
            random.shuffle(goal_depots)
            # Shuffle goal depots until no passenger is at their corresponding goal depot
            start_and_goal_depots = zip(starting_depots[:N], goal_depots[:N])
            if not any([start == goal for start, goal in start_and_goal_depots]):
                break

        # Update passenger colors to match their goal depots
        for p, g in zip(self.passengers, goal_depots[:N]):
            p.color = g

    def _reset_exploring_start(self):
        while True:
            # Fully randomize agent position
            self.agent.position = self.grid.get_random_position()
            self.passenger = None

            # Fully randomize passenger locations (without overlap)
            passenger_locs = np.stack(
                [self.grid.get_random_position() for _ in range(self.n_passengers)], axis=0)
            while len(np.unique(passenger_locs, axis=0)) < len(passenger_locs):
                passenger_locs = np.stack(
                    [self.grid.get_random_position() for _ in range(self.n_passengers)], axis=0)
            for i, p in enumerate(self.passengers):
                p.position = passenger_locs[i]

            # Randomly decide whether to move the taxi to a passenger
            if random.random() > 0.5:
                # If so, randomly choose which passenger
                p = random.choice(self.passengers)
                self.agent.position = p.position

                # Randomly decide if that passenger should be *in* the taxi
                if random.random() > 0.5:
                    p.in_taxi = True
                    self.passenger = p

            s = self.get_state()
            # Repeat until we aren't at a goal state (or stop if we don't care)
            if not (self.terminate_on_goal and self._check_goal(s)):
                break

    def _step(self, action):
        """
        Execute action without checking if it can run
        """
        if action < 4:
            super()._step(action)
            if self.passenger is not None:
                self.passenger.position = self.agent.position
        else:  # Interact
            if self.passenger is None:
                # pick up
                for p in self.passengers:
                    if (self.agent.position == p.position).all():
                        p.in_taxi = True
                        self.passenger = p
                        break  # max one passenger per taxi
            else:
                # dropoff
                dropoff_clear = True
                for p in (p for p in self.passengers if p is not self.passenger):
                    if (p.position == self.passenger.position).all():
                        dropoff_clear = False
                        break
                if dropoff_clear:
                    self.passenger.in_taxi = False
                    self.passenger = None

    def can_run(self, action):
        assert action in range(5)
        if action < 4:
            # movement
            offset = self._action_offsets[action]
            if self.grid.has_wall(self.agent.position, offset):
                return False
            elif self.passenger is None:
                return True
            else:
                # ensure movement won't cause passengers to overlap
                next_position = (self.agent.position + offset)
                for p in self.passengers:
                    if (p is not self.passenger) and (next_position == p.position).all():
                        return False
                return True
        else:
            if self.passenger is None:
                # pickup; can only pick up a passenger if one is here
                for p in self.passengers:
                    if (self.agent.position == p.position).all():
                        return True
                return False
            else:
                # dropoff
                if not self.depot_dropoff_only:
                    return True
                elif any([(depot.position == self.agent.position).all()
                          for depot in self.depots.values()]):
                    return True
                return False

    def get_state(self) -> np.ndarray:
        state = []
        row, col = self.agent.position
        state.extend([row, col])
        for p in self.passengers:
            row, col = p.position
            goal_depot_id = self.depot_ids[p.color]
            state.extend([row, col, p.in_taxi, goal_depot_id])
        return np.asarray(state, dtype=int)

    def set_state(self, state: np.ndarray):
        row, col, *remaining = state
        self.agent.position = row, col
        self.passenger = None
        self.passengers = []
        while remaining:
            row, col, in_taxi, goal_depot_id, *remaining = remaining
            color = self.depot_names[goal_depot_id]
            p = Passenger((row, col), color)
            p.in_taxi = in_taxi
            self.passengers.append(p)

    def get_goal_state(self) -> np.ndarray:
        state = []
        # omit taxi position from goal state
        for p in self.passengers:
            goal_depot_name = p.color
            goal_depot_id = self.depot_ids[goal_depot_name]
            goal_row, goal_col = self.depots[goal_depot_name].position
            in_taxi = False
            state.extend([goal_row, goal_col, in_taxi, goal_depot_id])
        return np.asarray(state, dtype=int)

    def _check_goal(self, state):
        goal = self.get_goal_state()
        if np.all(state[2:] == goal):  # ignore taxi, check passenger positions
            return True
        else:
            return False

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------

    def _render_objects(self) -> dict:
        objects = super()._render_objects()
        del objects['agent']

        passenger_patches = np.zeros_like(objects['walls'])
        for p in self.passengers:
            patch = self._render_passenger_patch(p.in_taxi, p.color)
            self._add_patch(passenger_patches, patch, p.position)

        taxi_patches = np.zeros_like(objects['walls'])
        patch = self._render_taxi_patch()
        self._add_patch(taxi_patches, patch, self.agent.position)

        objects.update({
            'taxi': taxi_patches,
            'passengers': passenger_patches,
        })

        return objects

    def _render_frame(self, content):
        """Generate a border to reflect the current in_taxi status"""
        in_taxi = (self.passenger is not None)
        img_shape = self.dimensions['img_shape']
        dash_widths = self.dimensions['dash_widths']
        if in_taxi:
            # pad with dashes to HxW
            dw_r, dw_c = dash_widths
            n_repeats = (img_shape[0] // (2 * dw_r), img_shape[1] // (2 * dw_c))
            image = np.tile(
                np.block([
                    [np.ones((dw_r, dw_c)), np.zeros((dw_r, dw_c))],
                    [np.zeros((dw_r, dw_c)), np.ones((dw_r, dw_c))],
                ]), n_repeats)

            # convert to color HxWx3
            image = np.tile(np.expand_dims(image, -1), (1, 1, 3))
            image = image * utils.get_rgb(self.passenger.color)
        else:
            # pad with white to HxWx3
            image = np.ones(img_shape + (3, ))

        pad_top_left, pad_bot_right = self.dimensions['border_widths']
        pad_width = ((pad_top_left, pad_bot_right), (pad_top_left, pad_bot_right), (0, 0))
        assert image.shape == np.pad(content, pad_width=pad_width).shape

        return image

    def _render_passenger_patch(self, in_taxi, color):
        """Generate a patch representing a passenger, along with any associated marks"""
        cell_width = self.dimensions['cell_width']

        patch = self._render_character_patch(color='white')

        # add marks relating to 'in_taxi'
        center = cell_width // 2
        if in_taxi:
            marks = np.zeros_like(patch[:, :, 0])
            marks[center, :] = 1
            marks[:, center] = 1
        else:
            marks = np.eye(cell_width, dtype=int) | np.fliplr(np.eye(cell_width, dtype=int))
        marks[(patch == 0).max(axis=-1)] = 0

        patch = utils.to_rgb(patch, color)
        patch[marks > 0, :] = utils.get_rgb('dimgray') / 4

        return patch

    def _render_taxi_patch(self):
        """Generate a patch representing a taxi"""
        depot = self._render_depot_patch(color='white')
        passenger = self._render_character_patch('white')
        patch = np.ones_like(depot) - (depot + passenger)

        # crop edges
        patch[0, :, :] = 0
        patch[:, 0, :] = 0
        patch[-1, :, :] = 0
        patch[:, -1, :] = 0

        patch = patch * utils.get_rgb('dimgray') / 4

        return patch
