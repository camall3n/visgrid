# %matplotlib inline
import random
from visgrid.envs.taxi import *
from visgrid.agents.expert.taxi_expert import TaxiExpert

random.seed(6)
env = TaxiEnv(n_passengers=3, exploring_starts=False, terminate_on_goal=False)
env.reset()
expert = TaxiExpert(env)
env.plot()

#%%
expert.run_skill('green')
env.plot()

#%%
expert.run_skill('interact')
env.plot()

#%%
expert.run_skill('blue')
env.plot()

#%%
expert.run_skill('interact')
env.plot()

#%%
expert.run_skill('yellow')
env.plot()

#%%
expert.run_skill('interact')
env.plot()

#%%
expert.run_skill('green')
env.plot()

#%%
expert.run_skill('interact')
env.plot()

#%%
expert.run_skill('red')
env.plot()

#%%
expert.run_skill('interact')
env.plot()

#%%
expert.run_skill('yellow')
env.plot()

#%%
expert.run_skill('interact')
env.plot()

#%%
expert.run_skill('blue')
env.plot()

#%%
expert.run_skill('interact')
env.plot()

#%%
expert.run_skill('red')
env.plot()

#%%
expert.run_skill('interact')
env.plot()


#%%
assert env._check_goal(env.get_state())
