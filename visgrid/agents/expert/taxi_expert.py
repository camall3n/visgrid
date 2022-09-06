import numpy as np

from .gridworld_expert import GridworldExpert

class TaxiExpert(GridworldExpert):
    def __init__(self, env):
        super().__init__(env)
        self.excluded_depots = set()

    def act(self, observation=None):
        if self.env.passenger is not None:
            goal_depot = self._goal_depot(self.env.passenger)
            if self._at(self.env.agent, goal_depot):
                self.excluded_depots.clear()
                return self.env.INTERACT
            elif self._depot_is_open(goal_depot):
                return self._next_step_towards(goal_depot.position)
            else:
                backup_depot = self._nearest_open_depot()
                if self._at(self.env.agent, backup_depot):
                    self.excluded_depots.clear()
                    return self.env.INTERACT
                else:
                    return self._next_step_towards(backup_depot.position)
        else:
            p = self._nearest_fare(with_open_goal_depot=True)
            if p is None:
                p = self._nearest_fare()
            if p is None:
                # no fares remaining!
                return self.env.action_space.sample()
            if self._at(self.env.agent, p):
                self._exclude_current_depot_if_any()
                return self.env.INTERACT
            else:
                return self._next_step_towards(p.position)

    def _exclude_current_depot_if_any(self):
        for other_depot in self.env.depots.values():
            if self._at(self.env.agent, other_depot):
                # already at another incorrect depot, but we
                # don't just want to return here in one step
                self.excluded_depots.add(other_depot.color)

    def _at(self, a: np.ndarray, b: np.ndarray):
        return (a.position == b.position).all()

    def _goal_depot(self, passenger):
        return self.env.depots[passenger.color]

    def _depot_is_open(self, depot):
        for p in self.env.passengers:
            if self._at(p, depot):
                if p is self.env.passenger:
                    return True
                return False
        return True

    def _nearest_open_depot(self):
        """
        Get the nearest unoccupied depot, excluding those that we already came from
        """
        nearest_depot = None
        nearest_distance = np.inf
        for d in self.env.depots.values():
            if not self._depot_is_open(d):
                continue
            if d.color in self.excluded_depots:
                continue
            distance = self._get_distance(self.env.agent.position, d.position)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_depot = d
        return nearest_depot

    def _nearest_fare(self, with_open_goal_depot=False):
        """
        Get the nearest passenger who still hasn't reached their goal depot

        exceptions: a list of passengers to ignore
        """
        nearest_fare = None
        nearest_distance = np.inf
        for p in self.env.passengers:
            psgr_goal_depot = self._goal_depot(p)
            if self._at(p, psgr_goal_depot):
                continue
            if with_open_goal_depot and not self._depot_is_open(psgr_goal_depot):
                continue
            distance = self._get_distance(self.env.agent.position, p.position)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_fare = p
        return nearest_fare
