import networkx as nx
import matplotlib.pyplot as plt

def meshgraph_generator(n_neighbours: int, n_row: int = 10, n_col: int = 10):
    if n_neighbours not in [4,8]:
        raise ValueError('Invalid number of neighbours. Must be 4 or 8')
    if n_row < 3 or n_col < 3:
        raise ValueError('Minimum mesh size must be 3x3')

    mesh_graph = nx.Graph()
    n_nodes = n_row * n_col
    pos_to_node = {}
    x,y = 0,0
    for i in range(n_nodes):
        label = f"N{i}"
        mesh_graph.add_node(i)
        mesh_graph.nodes[i]["label"] = label
        #Saving grid pos in a separate dictionary for faster retrieval
        pos_to_node[x,y] = i
        y += 1
        if y == n_row:
            x += 1
            y = 0

    #Connecting nodes
    for i in range(1,n_row-1):
        for j in range(1,n_col-1):
            #By default a 4 neighbors grid
            mesh_graph.add_edge(pos_to_node[(i, j)], pos_to_node[(i-1, j)])
            mesh_graph.add_edge(pos_to_node[(i, j)], pos_to_node[(i+1, j)])
            mesh_graph.add_edge(pos_to_node[(i, j)], pos_to_node[(i, j-1)])
            mesh_graph.add_edge(pos_to_node[(i, j)], pos_to_node[(i, j+1)])

            #Expand to 8 if needed
            if n_neighbours >= 8:
                mesh_graph.add_edge(pos_to_node[(i, j)], pos_to_node[(i-1, j-1)])
                mesh_graph.add_edge(pos_to_node[(i, j)], pos_to_node[(i-1, j+1)])
                mesh_graph.add_edge(pos_to_node[(i, j)], pos_to_node[(i+1, j-1)])
                mesh_graph.add_edge(pos_to_node[(i, j)], pos_to_node[(i+1, j+1)])

    #Edge Cases
    for i in range(n_col-1):
        mesh_graph.add_edge(pos_to_node[(0,i)], pos_to_node[(0,i+1)])
    for i in range(n_col-1):
        mesh_graph.add_edge(pos_to_node[(n_row-1,i)], pos_to_node[(n_row-1,i+1)])
    for i in range(n_row-1):
        mesh_graph.add_edge(pos_to_node[(i,0)], pos_to_node[(i+1,0)])
    for i in range(n_row-1):
        mesh_graph.add_edge(pos_to_node[(i,n_col-1)], pos_to_node[(i+1,n_col-1)])
    #Expand edge cases
    if n_neighbours >= 8:
        mesh_graph.add_edge(pos_to_node[(0, 1)], pos_to_node[(1,0)])
        mesh_graph.add_edge(pos_to_node[(0, n_col-2)], pos_to_node[(1,n_col-1)])
        mesh_graph.add_edge(pos_to_node[(n_row-2, 0)], pos_to_node[(n_row-1,1)])
        mesh_graph.add_edge(pos_to_node[(n_row-2, n_col-1)], pos_to_node[(n_row-1,n_col-2)])


    return mesh_graph, pos_to_node

test = True
if test:
    graph, _ = meshgraph_generator(n_neighbours=8, n_row=5, n_col=5)
    labels = nx.get_node_attributes(graph, 'label')
    pos = nx.spring_layout(graph, iterations=10000)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos, labels=labels)
    plt.show()