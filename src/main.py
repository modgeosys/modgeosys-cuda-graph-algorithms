"""Usage example(s)."""

import pickle
from audioop import reverse

import numpy as np
import cupy as cp

from modgeosys.graph.cuda.steiner import use_gpu, manhattan_distance, euclidean_distance, steiner_tree_approximation, purge_unreachable_subgraphs


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with open('/home/kweller/graph.pickle', 'rb') as pickled_sample_larger_graph_file:
        graph = pickle.load(pickled_sample_larger_graph_file)

    # Example usage
    # if use_gpu:
    #     nodes = cp.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)], dtype=cp.float32)
    # else:
    #     nodes = np.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)], dtype=np.float32)

    # terminals = [0, 1, 2, 3, 4]
    # edges = [
    #     (0, 1, {'weight': 1}),
    #     (1, 2, {'weight': 2}),
    #     (2, 3, {'weight': 3}),
    #     (3, 4, {'weight': 4}),
    #     (4, 0, {'weight': 5}),
    #     (0, 2, {'weight': 1.5}),
    #     (1, 3, {'weight': 2.5})
    # ]
    # required_currents = {
    #     (0, 1): 15,
    #     (1, 2): 25,
    #     (2, 3): 20,
    #     (3, 4): 10,
    #     (4, 0): 30,
    #     (0, 2): 20,
    #     (1, 3): 15
    # }

    # Cable properties (example)
    # cable_types = [
    #     {'capacity': 10, 'cost': 1},
    #     {'capacity': 20, 'cost': 1.8},
    #     {'capacity': 30, 'cost': 2.5}
    # ]

    steiner_tree_manhattan = steiner_tree_approximation(graph, manhattan_distance, manhattan_distance, use_gpu)
    # steiner_tree_euclidean = steiner_tree_approximation(graph, euclidean_distance, euclidean_distance, use_gpu)

    print('Steiner tree (Manhattan distance): ', steiner_tree_manhattan.edges(data=True))
    print('Length of Steiner tree (Manhattan distance): ', len(steiner_tree_manhattan.edges(data=True)))
    # print('Steiner tree (Euclidean distance): ', steiner_tree_euclidean.edges(data=True))
