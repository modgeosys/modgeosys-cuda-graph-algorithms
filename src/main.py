"""Usage example(s)."""

import pickle

from modgeosys.graph.cuda.steiner import manhattan_distance, euclidean_distance, approximate_steiner_minimal_tree, is_gpu_available


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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

    with open('/home/kweller/graph.pickle', 'rb') as pickled_sample_larger_graph_file:
        graph = pickle.load(pickled_sample_larger_graph_file)

    use_gpu = False # is_gpu_available()
    steiner_minimal_tree_manhattan = approximate_steiner_minimal_tree(graph, manhattan_distance, use_gpu)
    # steiner_minimal_tree_euclidean = approximate_steiner_minimal_tree(graph, euclidean_distance, )

    print('Steiner minimal tree (Manhattan distance): ', steiner_minimal_tree_manhattan.edges(data=True))
    print('Length of Steiner minimal tree (Manhattan distance): ', len(steiner_minimal_tree_manhattan.edges(data=True)))
    # print('Steiner minimal tree (Euclidean distance): ', steiner_minimal_tree_euclidean.edges(data=True))
