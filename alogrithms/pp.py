import numpy as np
import time as timer

from .utils import a_star, compute_heuristics, get_sum_of_cost


class PPSolver:
    def __init__(self, grid_map, starts, goals, max_timestep, runtime_limit):
        self.grid_map = grid_map
        self.starts = starts
        self.goals = goals
        self.num_agents = len(starts)
        self.max_timestep = max_timestep
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(grid_map, goal))
        self.runtime_limit = runtime_limit
        self.num_generated = 0

    def get_location(self, path, time):
        if time < 0:
            return path[0]
        elif time < len(path):
            return path[time]
        else:
            return path[-1]

    def find_solution(self):
        result, constraints = [], []
        max_path_len = 0
        num_free_cells = len(self.grid_map) * len (self.grid_map[0]) - sum(map(sum, self.grid_map))
        self.start_time = timer.time()
        for i in range(self.num_agents):
            if timer.time() - self.start_time > self.runtime_limit:
                return None
            path = a_star(self.grid_map, self.starts[i], self.goals[i], self.heuristics[i], i, constraints)
            if path is None or len(path) >= max_path_len + num_free_cells:
                return None
            result.append(path)
            if len(path) > max_path_len:
                max_path_len = len(path)
            for t, loc in enumerate(path):
                for j in range(i + 1, self.num_agents):
                    constraints.append({'agent': j, 'loc': [loc], 'timestep': t, 'positive': False})
                    if t <= len(path) - 2:
                        next_loc = path[t + 1]
                        constraints.append({'agent': j, 'loc': [next_loc, loc], 'timestep': t + 1, 'positive': False})
            if i < self.num_agents - 1:
                for j in range(i + 1):
                    for t in range(len(result[j]), max_path_len + num_free_cells + 1):
                        constraints.append({'agent': i + 1, 'loc': [self.goals[j]], 'timestep': t, 'positive': False})
        return result