import numpy as np
from .basesprite import BaseSprite

class Agent(BaseSprite):
    def __init__(self, position=(0, 0)):
        self.position = np.asarray(position)
