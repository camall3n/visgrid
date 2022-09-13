import random
import os

import matplotlib.pyplot as plt
import numpy as np
import seeding
from tqdm import tqdm

from visgrid.envs import GridworldEnv
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, NoiseWrapper, ClipWrapper

env = GridworldEnv(10, 10, hidden_goal=True, dimensions=GridworldEnv.dimensions_onehot)
env = ClipWrapper(NoiseWrapper(InvertWrapper(GrayscaleWrapper(env))))
ob, _ = env.reset()
env.plot(ob)
print(ob.shape)
