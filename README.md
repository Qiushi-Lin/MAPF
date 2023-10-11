# MAPF

The repository aims to reproduce some commonly used search-based solvers for Multi-Agent Path Finding (MAPF) in Python.

## Algorithms supported:

- [Conflict-Based Search (CBS)](https://www.sciencedirect.com/science/article/pii/S0004370214001386)
- [Conflict-Based Search with Disjoint Splitting (CBS-D)](https://ojs.aaai.org/index.php/ICAPS/article/view/3487)
- [Priority-Based Search (PBS)](https://ojs.aaai.org/index.php/AAAI/article/view/4758)
- [Prioritized Planning (PP)](https://ojs.aaai.org/index.php/AIIDE/article/view/18726)

## Benchmarks supported:

[MAPF Benchmark Sets from Nathan Sturtevant's Moving AI Lab](https://movingai.com/benchmarks/mapf/index.html)
- random-32-32-10.map
- random-64-64-10.map
- den312d.map
- warehouse-10-20-10-2-1.map

## Creating Instances

Generate the test set used for evaluation.
```
cd benchmarks
python create_test.py
```

## Testing

``python main.py --solver  solver_name``

# Contact

Email: qiushi_lin@sfu.ca