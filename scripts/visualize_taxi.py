from visgrid.envs import TaxiEnv

for seed in range(10):
    env = TaxiEnv(size=5,
                  n_passengers=1,
                  exploring_starts=True,
                  terminate_on_goal=True,
                  should_render=True,
                  dimensions=TaxiEnv.dimensions_5x5_to_48x48)
    ob = env.reset(seed=seed)[0]
    env.plot(ob)
print(ob.shape)
