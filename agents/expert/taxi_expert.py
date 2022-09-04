import numpy as np
from .gridworld_expert import GridworldExpert

class TaxiExpert(GridworldExpert):
    def __init__(self, env):
        super().__init__(env)
        self.skills = ['interact'] + list(env.depot_names)

    def run_skill(self, name):
        if name in self.env.depot_locs.keys():
            skill = lambda x: self.GoToDepot(x, name)
        else:
            skill = lambda _: self.Interact()
        while True:
            can_run, a, term = skill(self.env.agent.position)
            print(can_run, a, term)
            assert (can_run or term)
            if can_run:
                self.env.step(a)
            if term:
                break

    def skill_policy(self, env, skill_name):
        if skill_name in env.depot_locs.keys():
            skill = lambda env, x: self.GoToDepot(x, skill_name)
        elif skill_name == 'interact':
            skill = lambda env, x: self.Interact()
        else:
            raise ValueError('Invalid skill name' + str(skill_name))
        return skill(env, env.agent.position)

    def GoToDepot(self, start, depotname):
        depot = self.env.depot_locs[depotname]
        return self.GoToGridPosition(start=start, target=depot)[0]

    def Interact(self):
        # Check relevant state variables to see if skill can run
        agent_pos = self.env.agent.position
        at_depot = any(np.all(loc == agent_pos) for _, loc in self.env.depot_locs.items())
        at_passenger = any(np.all(p.position == agent_pos) for p in self.env.passengers)
        crowded = (self.env.passenger is not None and any(
            np.all(p.position == agent_pos)
            for p in self.env.passengers if p != self.env.passenger))

        if at_depot and at_passenger and not crowded:
            can_run = True
            action = 4
        else:
            # Nothing to do
            can_run = False
            action = None

        terminate = True  # skill always terminates in one step
        return (can_run, action, terminate)
