from meshgraph import meshgraph_generator, plot_graph
from support_functions import cost_assignment, test_cost_assignment
from aco_routing import ACO

mesh_graph, pos_to_node = meshgraph_generator(8,5,5)
edges_metadata = dict()
cost_assignment(mesh_graph, edges_metadata, test_cost_assignment, print_assignment=True)
plot_graph(mesh_graph)
aco = ACO(mesh_graph, ant_max_steps=100, num_iterations=100, ant_random_spawn=True)

#Source and dest should be int as the node id
#The library uses source and dest as str. Should we uniform the thing and use strings as ids?
#No GPU parallelism for now
aco_path, aco_cost = aco.find_shortest_path(
    source=1,
    destination=22,
    num_ants=100,
)
print(aco_path)
print(aco_cost)