# %matplotlib inline
import random
from visgrid.envs.taxi import *
from visgrid.agents.expert.taxi_expert import *

random.seed(0)
w = Taxi5x5(n_passengers=3)
w.plot()

#%%
run_skill(w, 'yellow')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'green')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'red')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'yellow')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'blue')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'red')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'yellow')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'blue')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'green')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'yellow')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
assert w.check_goal(w.get_state())
