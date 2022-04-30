import copy
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import scipy.ndimage
import random
from ..gridworld.grid import BaseGrid
from ..gridworld.gridworld import GridWorld
from ..gridworld.objects.passenger import Passenger
from ..gridworld.objects.depot import Depot
from visgrid import utils
import pdb

class TaxiGrid5x5(BaseGrid):
    depot_locs = {# yapf: disable
        'red':    (0, 0),
        'yellow': (4, 0),
        'blue':   (4, 3),
        'green':  (0, 4),
    }# yapf: enable
    depot_names = depot_locs.keys()

    def __init__(self):
        super().__init__(rows=5, cols=5)
        self._grid[1:4, 4] = 1
        self._grid[7:10, 2] = 1
        self._grid[7:10, 6] = 1

class TaxiGrid10x10(BaseGrid):
    depot_locs = {# yapf: disable
        'red':     (0, 0),
        'blue':    (8, 0),
        'green':   (9, 4),
        'yellow':  (0, 5),
        'gray':    (3, 3),
        'magenta': (4, 6),
        'cyan':    (0, 8),
        'orange':  (9, 9),
    }# yapf: enable
    depot_names = depot_locs.keys()

    def __init__(self):
        super().__init__(rows=10, cols=10)
        self._grid[1:8, 6] = 1
        self._grid[13:20, 2] = 1
        self._grid[13:20, 8] = 1
        self._grid[5:12, 12] = 1
        self._grid[1:8, 16] = 1
        self._grid[13:20, 16] = 1

class BaseTaxi(GridWorld):
    def __init__(self):
        super().__init__()
        self.actions.append(4)  # Add interact action
        self.passenger = None
        self.goal = None
        self.wall_color = 'black'
        self.grayscale = False

        # Place depots
        self.depots = dict()
        for name in self.depot_names:
            self.depots[name] = Depot(color=name)
            self.depots[name].position = self.depot_locs[name]

    def reset(self, goal=True, explore=False):
        self.passenger = None
        # Place passengers and taxi
        start_depots = list(self.depot_names)
        passenger_colors = copy.deepcopy(start_depots)
        random.shuffle(start_depots)
        random.shuffle(passenger_colors)
        for i, p in enumerate(self.passengers):
            p.position = self.depots[start_depots[i]].position
            p.color = passenger_colors[i]
            p.goal = p.color
            p.intaxi = False
        self.agent.position = self.depots[start_depots[-1]].position

        if goal:
            # Generate goal condition
            goal_depots = list(self.depots.keys())
            random.shuffle(goal_depots)
            N = len(self.passengers)
            while N > 0 and all([g == s for g, s in zip(goal_depots[:N], start_depots[:N])]):
                random.shuffle(goal_depots)
            for p, g in zip(self.passengers, goal_depots[:N]):
                p.color = g
                p.goal = g
            self.passenger_goals = dict([(p.goal, g)
                                         for p, g in zip(self.passengers, goal_depots[:N])])
            self.goal = TaxiGoal(self.passenger_goals)
        else:
            self.goal = None

        if explore:
            # Fully randomize agent position
            self.agent.position = self.get_random_position()
            
            #self.agent.position = self.depots[self.passengers[0].color].position
            #pdb.set_trace()
            
             
            # Fully randomize passenger positions (without overlap)
            if self.passengers:
                passenger_positions = np.stack(
                    [self.get_random_position() for _ in range(len(self.passengers))], axis=0)
                while len(np.unique(passenger_positions, axis=0)) < len(passenger_positions):
                    passenger_positions = np.stack(
                        [self.get_random_position() for _ in range(len(self.passengers))], axis=0)
                for i, p in enumerate(self.passengers):
                    p.position = passenger_positions[i]

                # Randomly decide if one passenger should be at the taxi
                if random.random() > 0.5:
                    # If so, randomly choose which passenger
                    p = random.choice(self.passengers)
                    p.position = self.agent.position

                    # Randomly decide if that passenger should be *in* the taxi
                    if random.random() > 0.5:
                        p.intaxi = True
                        self.passenger = p

        return self.get_state()

    def plot(self, ax=None, goal_ax=None, draw_bg_grid=True, linewidth_multiplier=1.0):
        ax = super().plot(ax,
                          draw_bg_grid=draw_bg_grid,
                          linewidth_multiplier=linewidth_multiplier,
                          plot_goal=False)
        for _, depot in self.depots.items():
            depot.plot(ax, linewidth_multiplier=linewidth_multiplier)
        for p in (p for p in self.passengers if not p.intaxi):
            p.plot(ax, linewidth_multiplier=linewidth_multiplier)
        for p in (p for p in self.passengers if p.intaxi):
            p.plot(ax, linewidth_multiplier=linewidth_multiplier)

    def render(self):
        wall_width = 2
        cell_width = 13
        passenger_width = 9
        depot_width = 3

        walls = expand_grid(self._grid, cell_width, wall_width)
        walls = to_rgb(walls) * get_rgb('dimgray') / 8

        passengers = np.zeros_like(walls)
        for p in self.passengers:
            patch, marks = passenger_patch(cell_width, passenger_width, p.intaxi)
            color_patch = to_rgb(patch) * get_rgb(p.color)
            color_patch[marks > 0, :] = get_rgb('dimgray') / 4
            row, col = cell_start(p.position, cell_width, wall_width)
            passengers[row:row + cell_width, col:col + cell_width, :] = color_patch

        depots = np.zeros_like(walls)
        for depot in self.depots.values():
            patch = depot_patch(cell_width, depot_width)
            color_patch = to_rgb(patch) * get_rgb(depot.color)
            row, col = cell_start(depot.position, cell_width, wall_width)
            depots[row:row + cell_width, col:col + cell_width, :] = color_patch

        taxis = np.zeros_like(walls)
        patch = taxi_patch(cell_width, depot_width, passenger_width)
        color_patch = to_rgb(patch) * get_rgb('dimgray') / 4
        row, col = cell_start(self.agent.position, cell_width, wall_width)
        taxis[row:row + cell_width, col:col + cell_width, :] = color_patch

        # compute foreground
        objects = passengers + depots + walls + taxis
        fg = np.any(objects > 0, axis=-1)

        # compute background
        bg = np.ones_like(walls) * get_rgb('white')
        bg[fg, :] = 0

        # construct border
        in_taxi = (self.passenger is not None)
        border_color = get_rgb('white' if self.grayscale or not in_taxi else self.passenger.color)
        image = generate_border(in_taxi, border_color)

        # insert content on top of border
        content = bg + objects
        image[4:-3, 4:-3, :] = content

        if self.grayscale:
            image = np.mean(image, axis=-1, keepdims=True)

        return image

    def step(self, action):
        pickup = False; bad_dropoff = False; good_dropoff = False; bad_pickup = False

        if action < 4:
            super().step(action)
            for p in self.passengers:
                if p.intaxi:
                    p.position = self.agent.position
                    break  # max one passenger per taxi
        elif action == 4:  # Interact
            if self.passenger is None:
                # pick up?
                for p in self.passengers:
                    if (self.agent.position == p.position).all():
                        p.intaxi = True
                        self.passenger = p
                        pickup = True
                        break  # max one passenger per taxi
		
                if not pickup:
                    bad_pickup = True
            else:
                # drop off?
                dropoff_clear = True
                for p in (p for p in self.passengers if p is not self.passenger):
                    if (p.position == self.passenger.position).all():
                        dropoff_clear = False
                        bad_dropoff = True
                        break

                if dropoff_clear:
                    self.passenger.intaxi = False
                    
		    #drop of current passenger at right depot
                    depot_position = self.depots[self.passenger.goal].position

                    if np.all(depot_position==self.passenger.position):
                        good_dropoff = True
                    else:
                        bad_dropoff = True

                    self.passenger = None
		    
		    

        s = self.get_state()
        if (self.goal is None) or not self.check_goal(s):
            done = False
        else:
            done = True


        #r = 0.0 if not done else 1.0
        
        if pickup:
            r = +5.0
        elif action==4 and (bad_dropoff or bad_pickup):
           r = -10.0
        elif action==4 and good_dropoff:
            r = +20.0
        else:
            r = -1.0

        #if done:
        #    r = +100
        if r==0:
            pdb.set_trace()
            print(s) 
        
        return s, r, done

    def get_state(self):
        state = []
        row, col = self.agent.position
        state.extend([row, col])
        for p in self.passengers:
            row, col = p.position
            intaxi = p.intaxi
            state.extend([row, col, intaxi])
        return np.asarray(state, dtype=int)

    def get_goal_state(self):
        state = []
        for p in self.passengers:
            goal_name = p.goal
            row, col = self.depots[goal_name].position
            intaxi = False
            state.extend([row, col, intaxi])
        return np.asarray(state, dtype=int)

    def check_goal(self, state):
        goal = self.get_goal_state()
        if np.all(state[2:] == goal):  # ignore taxi, check passenger positions
            return True
        else:
            return False

class TaxiGoal(BaseGrid):
    def __init__(self, passenger_goals):
        super().__init__(rows=1, cols=1 + len(passenger_goals))
        self._grid[:, :] = 0  # Clear walls

        colors = [color for passenger, color in passenger_goals.items()]
        self.depots = dict([(color, Depot(color=color)) for color in colors])
        for i, color in enumerate(colors):
            self.depots[color].position = (0, 1 + i)

        self.passengers = [Passenger(color=name) for name in list(passenger_goals.keys())]
        for p in self.passengers:
            p.position = self.depots[passenger_goals[p.color]].position

class Taxi5x5(BaseTaxi, TaxiGrid5x5):
    name = 'Taxi5x5'

    def __init__(self, n_passengers=1):
        super().__init__()
        assert 0 <= n_passengers <= 3, "'n_passengers' must be between 0 and 7"
        self.passengers = [Passenger(color='gray') for _ in range(n_passengers)]

class VisTaxi5x5(Taxi5x5):
    def __init__(self, grayscale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grayscale = grayscale

        if self.grayscale and len(self.passengers) == 1:
            for depot in self.depots.values():
                depot.color = 'gray'

    def reset(self, goal=False, explore=False):
        super().reset(goal=goal, explore=explore)
        if self.grayscale and len(self.passengers) == 1:
            self.passengers[0].color = 'gray'
        return self.render()

    def step(self, action):
        _, r, done = super().step(action)
        ob = self.render()
        return ob, r, done

class Taxi10x10(BaseTaxi, TaxiGrid10x10):
    name = 'Taxi10x10'

    def __init__(self, n_passengers=1):
        super().__init__()
        assert 0 <= n_passengers <= 7, "'n_passengers' must be between 0 and 7"
        self.passengers = [Passenger(color='gray') for _ in range(n_passengers)]
        self.reset()

def get_rgb(colorname):
    good_color = utils.get_good_color(colorname)
    color_tuple = colors.hex2color(colors.get_named_colors_mapping()[good_color])
    return np.asarray(color_tuple)

def to_rgb(array):
    """Add a channel dimension with 3 entries"""
    return np.tile(array[:, :, np.newaxis], (1, 1, 3))

def cell_start(position, cell_width, wall_width):
    """Compute <row, col> indices of top-left pixel of cell at given position"""
    row, col = position
    row_start = wall_width + row * (cell_width + wall_width)
    col_start = wall_width + col * (cell_width + wall_width)
    return (row_start, col_start)

def expand_grid(grid, cell_width, wall_width):
    """Expand the built-in maze grid using the provided width information"""
    for row_or_col_axis in [0, 1]:
        slices = np.split(grid, grid.shape[row_or_col_axis], axis=row_or_col_axis)
        walls = slices[0::2]
        cells = slices[1::2]
        walls = [np.repeat(wall, wall_width, axis=row_or_col_axis) for wall in walls]
        cells = [np.repeat(cell, cell_width, axis=row_or_col_axis) for cell in cells]
        slices = [item for pair in zip(walls, cells) for item in pair] + [walls[-1]]
        grid = np.concatenate(slices, axis=row_or_col_axis).astype(float)
    return grid

def passenger_patch(cell_width, passenger_width, in_taxi):
    """Generate a patch representing a passenger, along with any associated marks"""
    assert passenger_width <= cell_width
    sw_bg = np.tri(cell_width // 2, k=(cell_width // 2 - passenger_width // 2 - 2), dtype=int)
    nw_bg = np.flipud(sw_bg)
    ne_bg = np.fliplr(nw_bg)
    se_bg = np.fliplr(sw_bg)

    bg = np.block([[nw_bg, ne_bg], [sw_bg, se_bg]])

    # add center row / column for odd widths
    if cell_width % 2 == 1:
        bg = np.insert(bg, cell_width // 2, 0, axis=0)
        bg = np.insert(bg, cell_width // 2, 0, axis=1)

    # crop edges to a circle and invert
    excess = (cell_width - passenger_width) // 2
    bg[:excess, :] = 1
    bg[:, :excess] = 1
    bg[-excess:, :] = 1
    bg[:, -excess:] = 1
    patch = (1 - bg)

    # add marks relating to 'in_taxi'
    center = cell_width // 2
    if in_taxi:
        marks = np.zeros_like(patch)
        marks[center, :] = 1
        marks[:, center] = 1
    else:
        marks = np.eye(cell_width, dtype=int) | np.fliplr(np.eye(cell_width, dtype=int))
    marks[patch == 0] = 0

    return patch, marks

def depot_patch(cell_width, depot_width):
    """Generate a patch representing a depot"""
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

    return patch

def taxi_patch(cell_width, depot_width, passenger_width):
    """Generate a patch representing a taxi"""
    depot = depot_patch(cell_width, depot_width)
    passenger, _ = passenger_patch(cell_width, passenger_width, False)
    taxi_patch = 1 - (depot + passenger)

    # crop edges
    taxi_patch[0, :] = 0
    taxi_patch[:, 0] = 0
    taxi_patch[-1, :] = 0
    taxi_patch[:, -1] = 0

    return taxi_patch

def generate_border(in_taxi, color=None):
    """Generate a border to reflect the current in_taxi status"""
    if in_taxi:
        # pad with dashes to 84x84
        image = np.tile(
            np.block([[np.ones((6, 6)), np.zeros((6, 6))], [np.zeros((6, 6)),
                                                            np.ones((6, 6))]]), (7, 7))
        image = np.tile(np.expand_dims(image, -1), (1, 1, 3))
        image = image * color
    else:
        # pad with white to 84x84
        image = np.ones((84, 84, 3))

    return image
