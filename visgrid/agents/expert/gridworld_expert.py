from collections import defaultdict
from typing import Tuple
import warnings

import numpy as np

from ...utils import manhattan_dist

class GridworldExpert:
    def __init__(self, env):
        self.env = env
        self.saved_routes = {}

    def act(self, observation=None):
        agent_position = tuple(self.env.agent.position)
        goal_position = tuple(self.env.goal.position)
        if agent_position == goal_position:
            # nothing to do!
            return self.env.action_space.sample()
        else:
            return self._next_step_towards(goal_position)

    def store(self, _):
        pass

    def update(self):
        return 0.0

    def _next_step_towards(self, position):
        start = tuple(self.env.agent.position)
        target = tuple(position)
        self._update_routes(start, target)
        direction, _ = self.saved_routes.get((start, target), (None, None))
        if direction is None:
            warnings.warn(
                f'Could not find path from {start} to {target}; taking a random action...',
                RuntimeWarning)
            action = self.env.action_space.sample()
        else:
            action = self.env._action_ids[direction]
        return action

    def _get_distance(self, start, target):
        start = tuple(start)
        target = tuple(target)
        self._update_routes(start, target)
        direction, distance = self.saved_routes.get((start, target), (None, None))
        return distance

    def _update_routes(self, start: Tuple[int], target: Tuple[int]):
        """
        Update saved routes from start to target
        """
        # Cache results to save on repeated calls
        direction, distance = self.saved_routes.get((start, target), (None, None))
        action = self.env._action_ids[direction] if direction else None
        if (start, target) not in self.saved_routes or (action is not None
                                                        and not self.env.can_run(action)):
            path = self._find_astar_path(start, target)
            if path is not None:
                if path:
                    for i, (next_, current) in enumerate(reversed(list(zip(path[:-1], path[1:])))):
                        direction = tuple(np.asarray(next_) - current)
                        distance = len(path) - 1 - i
                        self.saved_routes[(current, target)] = direction, distance
                else:
                    self.saved_routes[(start, target)] = None, 0

    def _find_astar_path(self, start, target):
        """
        Run A* search to find a path from start to target
        """
        if all(np.asarray(start) == target):
            return []
        # Use A* to search for a path in gridworld from start to target
        closed_set = set()
        open_set = set()
        came_from = dict()
        gScore = defaultdict(lambda x: np.inf)
        fScore = defaultdict(lambda x: np.inf)

        open_set.add(tuple(start))
        gScore[start] = 0
        fScore[start] = manhattan_dist(start, target)

        while open_set:
            current = min(open_set, key=lambda x: fScore[x])
            if all(np.asarray(current) == target):
                return self._reconstruct_path(came_from, current)

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_gScore = gScore[current] + 1

                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_gScore >= gScore[neighbor]:
                    continue

                came_from[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + manhattan_dist(neighbor, target)

        return None

    def _get_neighbors(self, pos):
        neighbors = []
        for action, direction in self.env._action_offsets.items():
            if pos == tuple(self.env.agent.position):
                # if we're *at* this position, check with the env directly
                can_run = self.env.can_run(action)
            else:
                # otherwise, just check based on wall information
                can_run = not self.env.grid.has_wall(pos, direction)
            if can_run:
                neighbor = tuple(np.asarray(pos) + direction)
                neighbors.append(neighbor)
        return neighbors

    def _reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path
