import matplotlib.pyplot as plt
import networkx as nx
import math


def draw_graph_with_path(graph, path, title="Graph Visualization with Shortest Path"):
    """
    Draws a NetworkX graph using node tuples as positions, highlighting
    a specific path in red.

    Args:
        graph (nx.Graph or nx.DiGraph): The NetworkX graph object.
        path (list): An ordered list of node identifiers representing the path.
        title (str): The title for the matplotlib plot.
    """
    plt.figure(figsize=(9, 7))

    # Use the tuple coordinates as the position for drawing
    # This dictionary maps the node identifier (e.g., (0,0)) to its position (0,0)
    pos = {node: node for node in graph.nodes()}

    # Draw all edges in grey first
    all_edges = graph.edges()
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=all_edges,
        edge_color='gray',
        arrows=False,
    )

    node_costs = [graph.nodes[node].get('elevation', 0) for node in graph.nodes()]
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_costs,
        cmap='magma', 
        node_size=10, 
    )

    water_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_water')]
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=water_nodes,
        node_color='lightblue',
        node_size=10,
    )

    # Highlight the edges that are part of the shortest path
    # Create a list of (start_node, end_node) tuples for the path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=path_edges,
        edge_color='red',
        width=2.5,
    )

    # pos = {node: node  for node in graph.nodes() if node in path }
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=path,
        node_color='red',
        node_size=16, 
    )


    # Set plot parameters
    plt.title(title)
    plt.axis('off') # Hide the standard axes
    plt.show()


from TerrainGraph.terraingraph import create_graph

def heuristic(G, u, v):
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# 2. Weight Function: Distance + Elevation Penalty + Water Avoidance
def weight_func(graph, u, v, edge_attr):
    # Check for water (avoidance)
    if graph.nodes[u].get('is_water') or graph.nodes[v].get('is_water'):
        return float(100000000000) # Impossible to cross water
        
    # 2D Distance
    dist = heuristic(graph, u,v)
    
    # Elevation Difference
    elev_diff = abs(graph.nodes[u]['elevation'] - graph.nodes[v]['elevation'])
    
    # Cost = Distance + (Elevation Change * Penalty Factor)
    # Increase the multiplier (e.g. 10) to make the path flatter but longer
    return dist + (elev_diff * 5.0)

if __name__ == '__main__':
    res = 1000

    print("Generating Graph...")
    G = create_graph("../TerrainGraph/trentino.tif", "TerrainGraph/trentino_alto_adige.pbf", resolution=res)

    print("Finding best path...")
    # aco = ACO(G, ant_max_steps=1000, num_iterations=100, ant_random_spawn=True)
    # path, cost = aco.find_shortest_path( source=(0,0), destination=(res-1,res-1), num_ants=1000)
    # path = nx.dijkstra_path(G, source= 0, target= res -1 , weight='length')

    # ant_colony_parameters = {"alpha": 1, "beta": 2, "rho": 0.1, "ant_number": 5, "max_iterations": 10, "max_no_updates": 50, "n_best_ants": 5, "average_cycle_lenght": 3600}
    # key_nodes = {1, 200, 900, 547}
    #
    # aco = ACO_simulator(G, **ant_colony_parameters)
    # path = aco.simulation()

    print("Plotting graph...")
    # G.plot_graph(paths=[path], paths_colors=["blue"])
    G.plot_graph( paths_colors=["blue"])
    # draw_graph_with_path(G,path)
