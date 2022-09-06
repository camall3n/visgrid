import numpy as np
from .basesprite import BaseSprite

class Passenger(BaseSprite):
    def __init__(self, position=(0, 0), color: str = None):
        self.position = np.asarray(position)
        self.color = color
        self.in_taxi = False
