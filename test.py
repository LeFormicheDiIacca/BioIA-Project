import copy
from itertools import combinations
from meshgraph import meshgraph_generator, plot_graph, cost_assignment
from cost_functions import test_cost_assignment
from aco_routing import ACO

mesh_graph, pos_to_node = meshgraph_generator(8,5,5)
edges_metadata = dict()
cost_assignment(mesh_graph, edges_metadata, test_cost_assignment, print_assignment=False)

k = 3
plot_mesh_changes = False
keep_key_nodes_neighbors = False
final_graph = copy.deepcopy(mesh_graph)
key_nodes = [1,22]
all_key_node_pairs = list(combinations(key_nodes, 2))

if keep_key_nodes_neighbors:
    to_sum = []
    for node in key_nodes:
        to_sum += list(mesh_graph[node].keys())
    key_nodes += to_sum
#Source and dest should be int as the node id
#The library uses source and dest as str. Should we uniform the thing and use strings as ids?
#No GPU parallelism for now

aco = ACO(mesh_graph, ant_max_steps=100, num_iterations=100, ant_random_spawn=True)
color_list = ["red", "blue", "green", "yellow", "cyan", "magenta"]
path_list = []

#TODO: Se no path, ripristinare i vicini dei nodi chiave.
for pair in all_key_node_pairs:
    for path_id in range(k):
        aco_path, aco_cost = aco.find_shortest_path(
            source=pair[0],
            destination=pair[1],
            num_ants=200,
        )

        print(aco_path)
        print(aco_cost)
        path_list.append((aco_path, color_list[path_id]))

        aco_path = [x for x in aco_path if x not in key_nodes]
        mesh_graph.remove_nodes_from(aco_path)
        if plot_mesh_changes:
            plot_graph(mesh_graph)

plot_graph(final_graph, path_list)