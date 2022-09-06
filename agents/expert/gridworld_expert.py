from collections import defaultdict
import numpy as np
from ...utils import manhattan_dist

class GridworldExpert:
    def __init__(self, env):
        self.env = env
        self.saved_directions = {}

    def GetDistance(self, start, target):
        _, distance = self.GoToGridPosition(start, target)
        return distance

    def GoToGridPosition(self, start, target):
        start = tuple(start)
        target = tuple(target)
        # Cache results to save on repeated calls
        direction, distance = self.saved_directions.get((start, target), (None, None))
        action = self.env._action_ids[direction] if direction else None
        if (start, target) not in self.saved_directions or (action is not None
                                                            and not self.env.can_run(action)):
            path = self._GridAStarPath(start, target)
            if path is not None:
                if path:
                    for i, (next_, current) in enumerate(reversed(list(zip(path[:-1], path[1:])))):
                        direction = tuple(np.asarray(next_) - current)
                        self.saved_directions[(current, target)] = direction, len(path) - 1 - i
                else:
                    self.saved_directions[(start, target)] = None, 0
        direction, distance = self.saved_directions.get((start, target), (None, None))
        action = self.env._action_ids[direction] if direction else None
        can_run = True if (action is not None and self.env.can_run(action)) else False
        terminate = True if (start == target) else False
        return (can_run, action, terminate), distance

    def _GridAStarPath(self, start, target):
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
