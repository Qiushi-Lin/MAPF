import numpy as np
import time as timer
import copy
from collections import OrderedDict

from .utils import a_star, compute_heuristics, get_sum_of_cost


class PBSSolver:
    def __init__(self, grid_map, starts, goals, max_timestep, runtime_limit):
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

    def detect_collision(self, path1, path2):
        max_path_len = max(len(path1), len(path2))
        for t in range(max_path_len):
            pos1 = path1[min(len(path1) - 1, t)]
            pos2 = path2[min(len(path2) - 1, t)]
            if pos1 == pos2:
                return {'loc': [pos1], 'timestep': t}
            if t + 1 < min(len(path1), len(path2)):
                if path1[t] == path2[t + 1] and path1[t + 1] == path2[t]:
                    return {'loc': [path1[t], path1[t + 1]], 'timestep': t + 1}

    def detect_collisions(self, paths):
        collisions = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                c = self.detect_collision(paths[i], paths[j])
                if c:
                    collisions.append({'a1': i, 'a2': j, **c})
        return collisions

    def topological_ordering(self, graph_matrix):
        num_nodes = len(graph_matrix)
        rst = []
        visited = [False for _ in range(num_nodes)]
        while len(rst) < num_nodes:
            in_degree = np.sum(graph_matrix, axis=0)
            n_i  = -1
            for i, d in enumerate(in_degree):
                if d == 0 and not visited[i]:
                    n_i = i
                    break
            if n_i < 0:
                return None
            rst.append(n_i)
            visited[n_i] = True
            for i in range(num_nodes):
                graph_matrix[n_i][i] = 0
        return rst

    def update_plan(self, node, a_i):
        paths = copy.deepcopy(node['paths'])
        orderings = node['orderings']
        agents = OrderedDict()
        agents[a_i] = 0
        counter = 0
        for a_k, a_j in orderings:
            if a_k == a_i and a_j not in agents:
                counter += 1
                agents[a_j] = counter
        graph_matrix = np.zeros((counter + 1, counter + 1))
        for a_k, a_j in orderings:
            if a_k in agents and a_j in agents:
                graph_matrix[agents[a_k]][agents[a_j]] = 1
        partial_ordering = self.topological_ordering(graph_matrix)
        if not partial_ordering:
            return None
        partial_ordering = [list(agents.keys())[i] for i in partial_ordering]
        for a_j in partial_ordering:
            if a_j != a_i:
                flag = False
                for a_k, a_l in orderings:
                    collision = self.detect_collision(paths[a_k], paths[a_l])
                    if a_l == a_j and collision:
                        flag = True
                if not flag:
                    continue
            constraints = []
            for a_k, a_l in orderings:
                if a_l == a_j:
                    path = paths[a_k]
                    for t, loc in enumerate(path):
                        constraints.append({'agent': a_j, 'loc': [loc], 'timestep': t, 'positive': False})
                        if t <= len(path) - 2:
                            next_loc = path[t + 1]
                            constraints.append({'agent': a_j, 'loc': [next_loc, loc], 'timestep': t + 1, 'positive': False})
                    for t in range(len(path), self.max_timestep):
                        constraints.append({'agent': a_j, 'loc': [loc], 'timestep': t, 'positive': False})
            new_path = a_star(self.grid_map, self.starts[a_j], self.goals[a_j], self.heuristics[a_j], a_j, constraints)
            if new_path:
                paths[a_j] = new_path
            else:
                return None
        return paths

    def find_solution(self):
        self.stack = []
        root = {'cost': 0,
                'orderings': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_agents):
            path = a_star(self.grid_map, self.starts[i], self.goals[i], self.heuristics[i], i, [])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)
        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = self.detect_collisions(root['paths'])
        self.stack.append(root)
        self.start_time = timer.time()
        while self.stack:
            P = self.stack.pop()
            if not P['collisions']:
                return P['paths']
            if timer.time() - self.start_time > self.runtime_limit:
                return None
            collision = copy.deepcopy(P['collisions'][0])
            for c in copy.deepcopy(P['collisions']):
                if c['timestep'] < collision['timestep']:
                    collision = c
            orderings = [(collision['a1'], collision['a2']), (collision['a2'], collision['a1'])]
            nodes = []
            for ordering in orderings:
                Q = {}
                Q['orderings'] = copy.deepcopy(P['orderings']) + [ordering]
                Q['paths'] = copy.deepcopy(P['paths'])
                paths = self.update_plan(Q, ordering[1])
                if paths:
                    Q['paths'] = copy.deepcopy(paths)
                    Q['collisions'] = self.detect_collisions(Q['paths'])
                    Q['cost'] = get_sum_of_cost(Q['paths'])
                    nodes.append(Q)
            if len(nodes) > 1 and nodes[0]['cost'] > nodes[1]['cost']:
                nodes.reverse()
            self.stack += nodes
        return None