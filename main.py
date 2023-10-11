import numpy as np
from tqdm import tqdm
import pickle
import argparse

from alogrithms.utils import is_valid, get_sum_of_cost
from alogrithms import CBSSolver, PBSSolver, PPSolver

# config
import yaml
config = yaml.safe_load(open("./config.yaml", 'r'))
FREE_SPACE, OBSTACLE = config['grid_map']['FREE_SPACE'], config['grid_map']['OBSTACLE'] 


def get_solver(solver, grid_map, starts, goals, max_timestep, runtime_limit):
    if solver == "CBS":
        return CBSSolver(grid_map, starts, goals, max_timestep, runtime_limit, disjoint=False)
    if solver == "CBS-D":
        return CBSSolver(grid_map, starts, goals, max_timestep, runtime_limit, disjoint=True)
    if solver == "PBS":
        return PBSSolver(grid_map, starts, goals, max_timestep, runtime_limit)
    if solver == "PP":
        return PPSolver(grid_map, starts, goals, max_timestep, runtime_limit)
    raise AssertionError("solver not implemented")


def main(args):
    print(f"Calling the {args.solver} solver ...")
    runtime_limit = config['runtime_limit']
    num_instances = config['num_instances_per_test']
    for map_name, num_agents in config['test_settings']:
        file_name = f"./benchmarks/test_set/{map_name}_{num_agents}agents.pth"
        with open(file_name, 'rb') as f:
            instances = pickle.load(f)
        print(f"Testing instances for {map_name} with {num_agents} agents ...")
        success, avg_step = 0.0, 0.0
        max_timestep = config['max_timesteps'][map_name]
        for grid_map, starts, goals in tqdm(instances[0: num_instances]):
            solver = get_solver(args.solver, grid_map, starts, goals, max_timestep, runtime_limit)
            paths = solver.find_solution()
            if paths and is_valid(starts, goals, paths):
                success += 1
                avg_step += get_sum_of_cost(paths)
            else:
                avg_step += max_timestep * num_agents
        with open(f"./log/results.csv", 'a+') as f:
            height, width = np.shape(grid_map)
            num_obstacles = sum([row.count(OBSTACLE) for row in grid_map])
            f.write(f"{args.solver},{runtime_limit},{num_instances},{map_name},{height * width},{num_obstacles}," +\
                f"{num_agents},{success / num_instances},{avg_step / (num_instances * num_agents)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="CBS")
    args = parser.parse_args()
    main(args)