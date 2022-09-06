import copy
import numpy as np

class BaseSprite:
    def __init__(self, position=(0, 0)):
        self.position = np.asarray(position)  # row, col

    def __setattr__(self, name, value):
        if name == 'position':
            value = copy.deepcopy(np.asarray(value, dtype=int))
        super().__setattr__(name, value)
