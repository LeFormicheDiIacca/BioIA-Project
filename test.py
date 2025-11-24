from ACO.ACO_simulator import ACO_simulator
from cost_functions import test_cost_assignment
from meshgraph import MeshGraph

if __name__ == '__main__':
    mesh_graph = MeshGraph(8,10,10)
    edges_metadata = dict()
    mesh_graph.cost_assignment(edges_metadata, test_cost_assignment, print_assignment=False)
    key_nodes = {1, 44, 59, 81}
    aco = ACO_simulator(mesh_graph, key_nodes, 1,2,0.1,25, 1000)
    path, path_cost = aco.simulation()
    print(path)
    print(path_cost)
    #for edge in mesh_graph.edges():
    #    print(f"{edge[0]}->{edge[1]}\nMetadata:{mesh_graph[edge[0]][edge[1]]}")
    mesh_graph.plot_graph([(path, "red")], key_nodes = key_nodes)
#Ants still stupids as fuck
"""
TODO:
    1. Add possibility to see what each ant is doing
    2. Need to fine tune the ACO hyperparameters and we are done
    3. Should local update be done when the ants finish with their path calculation?
        Shared memory and let's go for now. Glory to C and the Ants🫡
"""