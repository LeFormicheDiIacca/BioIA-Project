import copy
from itertools import combinations

import inspyred
from meshgraph import meshgraph_generator, plot_graph, cost_assignment
from cost_functions import test_cost_assignment
from aco_routing import ACO

mesh_graph, pos_to_node, node_to_pos = meshgraph_generator(8,10,10)
edges_metadata = dict()
cost_assignment(mesh_graph, edges_metadata, test_cost_assignment, print_assignment=False)

k = 1
plot_mesh_changes = False
keep_key_nodes_neighbors = False
final_graph = copy.deepcopy(mesh_graph)
key_nodes = [1,22, 55]
all_key_node_pairs = list(combinations(key_nodes, 2))

if keep_key_nodes_neighbors:
    to_sum = []
    for node in key_nodes:
        to_sum += list(mesh_graph[node].keys())
    key_nodes += to_sum

"""complete_graph = nx.Graph()
for(source, target) in all_key_node_pairs:
    path = nx.shortest_path(final_graph, source, target, weight='cost')
    path_cost = 0
    for i in range(len(path)-1):
        path_cost += final_graph[path[i]][path[i+1]]['cost']
    complete_graph.add_edge(source, target, weight=path_cost)
    print(path)

plot_graph(complete_graph)"""

#Source and dest should be int as the node id
#The library uses source and dest as str. Should we uniform the thing and use strings as ids?
#No GPU parallelism for now

aco = ACO(mesh_graph, ant_max_steps=100, num_iterations=20, ant_random_spawn=True, alpha= 1, beta= 3)



color_list = ["red", "blue", "green", "yellow", "cyan", "magenta"]
path_list = []
idx = 0
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
        path_list.append((aco_path, color_list[path_id+idx]))

        aco_path = [x for x in aco_path if x not in key_nodes]
        mesh_graph.remove_nodes_from(aco_path)
        if plot_mesh_changes:
            plot_graph(mesh_graph)
    idx += k

plot_graph(final_graph, path_list, key_nodes)