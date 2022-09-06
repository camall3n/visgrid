import numpy as np
from .basesprite import BaseSprite

class Depot(BaseSprite):
    def __init__(self, position=(0, 0), color: str = 'red'):
        self.position = np.asarray(position)
        self.color = color
