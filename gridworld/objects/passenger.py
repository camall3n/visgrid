import numpy as np
import matplotlib.pyplot as plt
from .basesprite import BaseSprite, pos2xy

class Passenger(BaseSprite):
    def __init__(self, position=(0, 0), color='gray'):
        self.position = np.asarray(position)
        self.color = color
        self.intaxi = False

    def plot(self, ax, linewidth_multiplier=1.0):
        x, y = pos2xy(self.position) + (0.5, 0.5)
        outline_color = self.get_good_color(self.color)
        fill_color = outline_color if self.intaxi else 'white'

        fill = plt.Circle((x, y),
                          0.2,
                          color=fill_color,
                          fill=self.intaxi,
                          linewidth=1 * linewidth_multiplier)
        ax.add_patch(fill)
        outline = plt.Circle((x, y),
                             0.2,
                             color=outline_color,
                             fill=False,
                             linewidth=1 * linewidth_multiplier)
        ax.add_patch(outline)
