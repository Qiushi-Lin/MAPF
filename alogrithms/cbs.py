import numpy as np
import time as timer
import heapq
import random
import copy
from collections import OrderedDict

from .utils import a_star, compute_heuristics, get_sum_of_cost


class CBSSolver:
    def __init__(self, grid_map, starts, goals, max_timestep, runtime_limit, disjoint=False):
        self.grid_map = grid_map
        self.starts = starts
        self.goals = goals
        self.num_agents = len(starts)
        self.max_timestep = max_timestep
        self.stack = []
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(grid_map, goal))
        self.runtime_limit = runtime_limit
        self.runtime = 0.0
        self.disjoint = disjoint
        self.num_generated = 0

    def get_location(self, path, time):
        if time < 0:
            return path[0]
        elif time < len(path):
            return path[time]
        else:
            return path[-1]

    def detect_collision(self, path1, path2):
        for t in range(max(len(path1), len(path2))):
            pos1 = path1[min(len(path1) - 1, t)]
            pos2 = path2[min(len(path2) - 1, t)]
            if pos1 == pos2:
                return {'loc': [pos1], 'timestep': t}
            if t + 1 < min(len(path1), len(path2)):
                if path1[t] == path2[t + 1] and path1[t + 1] == path2[t]:
                    return {'loc': [path1[t], path1[t + 1]], 'timestep': t + 1}
        return None

    def detect_collisions(self, paths):
        collisions = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                c = self.detect_collision(paths[i], paths[j])
                if c:
                    collisions.append({'a1': i, 'a2': j, **c})
        return collisions
    
    def standard_splitting(self, collision):
        constraints = []
        # vertex collision
        if len(collision['loc']) == 1:
            constraints.append({'agent': collision['a1'],
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': False})
            constraints.append({'agent': collision['a2'],
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': False})
        # edge collision
        if len(collision['loc']) == 2:
            constraints.append({'agent': collision['a1'],
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': False})
            constraints.append({'agent': collision['a2'],
                'loc': collision['loc'][::-1], # reverse
                'timestep': collision['timestep'],
                'positive': False})
        return constraints

    def disjoint_splitting(self, collision):
        constraints = []
        agent  = collision['a1'] if random.randint(0, 1) else collision['a2']
        # vertex collision
        if len(collision['loc']) == 1:
            constraints.append({'agent': agent,
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': False})
            constraints.append({'agent': agent,
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': True})
        # edge collision
        if len(collision['loc']) == 2:
            loc = collision['loc'] if agent == collision['a1'] else collision['loc'][::-1] # reverse
            constraints.append({'agent': agent,
                'loc': loc,
                'timestep': collision['timestep'],
                'positive': False})
            constraints.append({'agent': agent,
                'loc': loc,
                'timestep': collision['timestep'],
                'positive': True})
        return constraints

    def paths_violate_constraint(self, constraint, paths):
        assert constraint['positive'] is True
        rst = []
        for i in range(len(paths)):
            if i == constraint['agent']:
                continue
            curr = self.get_location(paths[i], constraint['timestep'])
            prev = self.get_location(paths[i], constraint['timestep'] - 1)
            if len(constraint['loc']) == 1:  # vertex constraint
                if constraint['loc'][0] == curr:
                    rst.append(i)
            else:  # edge constraint
                if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                        or constraint['loc'] == [curr, prev]:
                    rst.append(i)
        return rst

    def push_node(self, node):
        heapq.heappush(self.stack, (node['cost'], len(node['collisions']), self.num_generated, node))
        self.num_generated += 1

    def pop_node(self):
        _, _, _, node = heapq.heappop(self.stack)
        return node

    def find_solution(self):
        self.start_time = timer.time()

        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_agents):  # Find initial path for each agent
            path = a_star(self.grid_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                return None
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = self.detect_collisions(root['paths'])
        self.push_node(root)

        while self.stack:
            if timer.time() - self.start_time > self.runtime_limit:
                return None
            P = self.pop_node()
            if not P['collisions']:
                return P['paths']
            collision = copy.deepcopy(P['collisions'][0])
            for c in copy.deepcopy(P['collisions']):
                if c['timestep'] < collision['timestep']:
                    collision = c
            constraints = self.standard_splitting(collision) if not self.disjoint \
                    else self.disjoint_splitting(collision)
            for constraint in constraints:
                Q = {}
                Q['constraints'] = copy.deepcopy(P['constraints']) + [constraint]
                Q['paths'] = copy.deepcopy(P['paths'])
                # vanilla CBS just needs to replan one agent
                agents = [constraint['agent']]
                if constraint['positive']:
                    agents = agents + self.paths_violate_constraint(constraint, Q['paths'])
                    # add negative constraints for all other agents
                    for a_i in range(self.num_agents):
                        if a_i != constraint['agent']:
                            if len(constraint['loc']) == 1:
                                Q['constraints'].append({'agent': a_i,
                                    'loc': constraint['loc'],
                                    'timestep': constraint['timestep'],
                                    'positive': False})
                            if len(constraint['loc']) == 2:
                                Q['constraints'].append({'agent': a_i,
                                    'loc': constraint['loc'][::-1],
                                    'timestep': constraint['timestep'],
                                    'positive': False})
                                Q['constraints'].append({'agent': a_i,
                                    'loc': [constraint['loc'][0]],
                                    'timestep': constraint['timestep'] - 1,
                                    'positive': False})
                                Q['constraints'].append({'agent': a_i,
                                    'loc': [constraint['loc'][1]],
                                    'timestep': constraint['timestep'],
                                    'positive': False})
                # replan agents
                path_existence = True
                for a_i in agents:
                    path = a_star(self.grid_map, self.starts[a_i], self.goals[a_i],\
                            self.heuristics[a_i], a_i, Q['constraints'])
                    if path:
                        Q['paths'][a_i] = path
                    else:
                        path_existence = False
                        break
                if path_existence:
                    Q['collisions'] = self.detect_collisions(Q['paths'])
                    Q['cost'] = get_sum_of_cost(Q['paths'])
                    self.push_node(Q)
        return None