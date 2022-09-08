from visgrid.envs.gridworld import GridworldEnv
from visgrid.wrappers.transforms import wrap_gridworld

env = GridworldEnv(rows=6,
                   cols=6,
                   exploring_starts=False,
                   terminate_on_goal=False,
                   fixed_goal=True,
                   hidden_goal=True,
                   rendering=True,
                   dimensions=GridworldEnv.dimensions_6x6_to_18x18)
env = wrap_gridworld(env)
ob = env.reset()[0]
env.plot(ob)
