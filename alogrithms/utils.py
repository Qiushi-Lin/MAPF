import heapq

# config
import yaml
config = yaml.safe_load(open("./config.yaml", 'r'))
FREE_SPACE, OBSTACLE = config['grid_map']['FREE_SPACE'], config['grid_map']['OBSTACLE'] 


def move(loc, d):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[d][0], loc[1] + directions[d][1]


def compute_heuristics(grid_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for d in range(4):
            child_loc = move(loc, d)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(grid_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(grid_map[0]):
               continue
            if grid_map[child_loc[0]][child_loc[1]] == OBSTACLE:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def detect_collision(path1, path2):
    for t in range(max(len(path1), len(path2))):
        pos1 = path1[min(len(path1) - 1, t)]
        pos2 = path2[min(len(path2) - 1, t)]
        if pos1 == pos2:
            return {'loc': [pos1], 'timestep': t}
        if t + 1 < min(len(path1), len(path2)):
            if path1[t] == path2[t + 1] and path1[t + 1] == path2[t]:
                return {'loc': [path1[t], path1[t + 1]], 'timestep': t + 1}
    return None


def detect_collisions(paths):
    collisions = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            c = detect_collision(paths[i], paths[j])
            if c:
                collisions.append({'a1': i, 'a2': j, **c})
    return collisions


def is_valid(starts, goals, paths):
    if len(detect_collisions(paths)):
        return False
    for i in range(len(paths)):
        if starts[i] != paths[i][0] or goals[i] != paths[i][-1]:
            return False
    return True


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    if next_time not in constraint_table:
        return False
    constraints = constraint_table[next_time]
    for c in constraints:
        if c['positive'] == False: # negative constraint
            if len(c['loc']) == 1 and next_loc == c['loc'][0]:
                return True
            if len(c['loc']) == 2 and curr_loc == c['loc'][0] and next_loc == c['loc'][1]:
                return True
        else: # positive constraint
            if len(c['loc']) == 1 and next_loc != c['loc'][0]:
                return True
            if len(c['loc']) == 2 and (curr_loc != c['loc'][0] or next_loc != c['loc'][1]):
                return True
    return False


def build_constraint_table(constraints, agent):
    constraint_table = {}
    for c in constraints:
        if c['agent'] == agent:
            if c['timestep'] not in constraint_table:
                constraint_table[c['timestep']] = []
            constraint_table[c['timestep']].append(c)
    return constraint_table


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def a_star(grid_map, start_loc, goal_loc, h_values, agent, constraints):
    open_list = []
    closed_list = dict()
    constraint_table = build_constraint_table(constraints, agent)
    earliest_goal_timestep = 0
    for t, constraints in constraint_table.items():
        for c in constraints:
            if c['positive'] == False and [goal_loc] == c['loc'] and t + 1 > earliest_goal_timestep:
                earliest_goal_timestep = t + 1
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep': 0}
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root

    while len(open_list) > 0:
        curr = pop_node(open_list)
        if curr['loc'] == goal_loc and curr['timestep'] >= earliest_goal_timestep:
            return get_path(curr)
        for d in range(5):
            child_loc = curr['loc']
            if d < 4:
                child_loc = move(curr['loc'], d)
            if child_loc[0] < 0 or child_loc[0] >= len(grid_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(grid_map[0]):
                continue
            if grid_map[child_loc[0]][child_loc[1]] == OBSTACLE:
                continue
            if is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraint_table):
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'timestep': curr['timestep'] + 1}
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)
    return None


def detect_collision(path1, path2):
    for t in range(max(len(path1), len(path2))):
        pos1, pos2 = path1[min(len(path1) - 1, t)], path2[min(len(path2) - 1, t)]
        if pos1 == pos2:
            return {'loc': [pos1], 'timestep': t}
        if t + 1 < min(len(path1), len(path2)):
            if path1[t] == path2[t + 1] and path1[t + 1] == path2[t]:
                return {'loc': [path1[t], path1[t + 1]], 'timestep': t + 1}
    return None


def detect_collisions(paths):
    collisions = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            c = detect_collision(paths[i], paths[j])
            if c:
                collisions.append({'a1': i, 'a2': j, **c})
    return collisions


def is_valid(starts, goals, paths):
    if len(detect_collisions(paths)):
        return False
    for i in range(len(paths)):
        if tuple(starts[i]) != paths[i][0] or tuple(goals[i]) != paths[i][-1]:
            return False
    return True


def get_sum_of_cost(paths):
    soc = 0
    for path in paths:
        soc += len(path) - 1
    return soc