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
        # make a large 1280x1280 plot
        fig, ax = plt.subplots(figsize=(2, 2), dpi=640)
        ax.axis('off')
        ax.margins(0)
        fig.tight_layout(pad=0)

        # draw with thick lines & without background grid markers
        self.plot(ax, draw_bg_grid=False, linewidth_multiplier=4)

        # extract image buffer from pyplot
        fig.canvas.draw()
        image_content = np.frombuffer(fig.canvas.tostring_rgb(),
                                      dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] +
                                                             (3, ))

        # prevent figure from actually displaying
        plt.close(fig)

        # downsample to 80x80
        image_content = scipy.ndimage.zoom(image_content, zoom=(1 / 16, 1 / 16, 1), order=1)

        if self.passenger:
            # pad with black/white dashes to 84x84
            if self.grayscale:
                dash_color = 255 * np.ones(3)
            else:
                dash_color = get_rgb(self.passenger.get_good_color(self.passenger.color))
            image = np.tile(
                np.block([[np.ones((6, 6)), np.zeros((6, 6))], [np.zeros((6, 6)),
                                                                np.ones((6, 6))]]), (7, 7))
            image = np.tile(np.expand_dims(image, -1), (1, 1, 3)).astype(image_content.dtype)
            image = 255 * image * dash_color
        else:
            # pad with white to 84x84
            image = 255 * np.ones((84, 84, 3), dtype=image_content.dtype)

        image[2:-2, 2:-2, :] = image_content

        # flip image vertically
        image = image[::-1, :, :]

        if self.grayscale:
            image = np.mean(image, axis=-1, keepdims=True)

        return image

    def step(self, action):
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
                        break  # max one passenger per taxi
            else:
                # drop off?
                dropoff_clear = True
                for p in (p for p in self.passengers if p is not self.passenger):
                    if (p.position == self.passenger.position).all():
                        dropoff_clear = False
                        break
                if dropoff_clear:
                    self.passenger.intaxi = False
                    self.passenger = None
        s = self.get_state()
        if (self.goal is not None) or not self.check_goal(s):
            done = False
        else:
            done = True
        r = -1.0 if not done else 1000
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
    return colors.hex2color(colors.get_named_colors_mapping()[colorname])
