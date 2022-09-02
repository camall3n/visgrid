# %matplotlib inline
import random
from visgrid.agents.expert.taxi_expert import *
from visgrid.envs.taxi import *

random.seed(0)
w = Taxi10x10()
w.plot()

#%%
run_skill(w, 'gray')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'magenta')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%

assert w.check_goal(w.get_state())
