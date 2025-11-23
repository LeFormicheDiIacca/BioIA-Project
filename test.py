from ACO.ACO_simulator import ACO_simulator
from cost_functions import test_cost_assignment
from meshgraph import MeshGraph

mesh_graph = MeshGraph(8,10,10)
edges_metadata = dict()
mesh_graph.cost_assignment(edges_metadata, test_cost_assignment, print_assignment=False)
key_nodes = {1, 44, 59, 81}
aco = ACO_simulator(mesh_graph, key_nodes, 1,5,0.7,200, 100)
path, path_cost = aco.simulation()
print(path)
print(path_cost)
#for edge in mesh_graph.edges():
#    print(f"{edge[0]}->{edge[1]}\nMetadata:{mesh_graph[edge[0]][edge[1]]}")
mesh_graph.plot_graph([(path, "red")], key_nodes = key_nodes)

path, path_cost = aco.TwoOptHeuristic(path)
print(path)
print(path_cost)
#for edge in mesh_graph.edges():
#    print(f"{edge[0]}->{edge[1]}\nMetadata:{mesh_graph[edge[0]][edge[1]]}")
mesh_graph.plot_graph([(path, "red")], key_nodes = key_nodes)

#Ants still stupids as fuck
"""
TODO:
    1. Add possibility to see what each ant is doing
    2. Parallelization
    3. Add a path optimization system in order to improve the ants result
    4. Should local update be done when the ants finish with their path calculation?
        In sequential ants computation this causes no problem but in could in parallel
        computation. No actual idea
"""