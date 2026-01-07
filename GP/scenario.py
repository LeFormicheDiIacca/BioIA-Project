from TerrainGraph.terraingraph import create_graph
import random
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
import numpy as np

# two nodes are directly connected IF:
#   - their edge number is +-1 (i.e., vertically connected)
#   - their edge number is +-res (i.e., horizontally connected)
#   - their edge number is +- res+1 (i.e., diagonally, on the SW-NE axis)
#   - their edge number is +- res-1 (i.e., diagonally, on the NW-SE axis)

# I want my function to loop through all nodes AND find n lists of tuples with:
# -  two connected water nodes and a non-connected node that is linearly reachable only going through water
#   (if no two connected water nodes, a whatever water node a non connected one)
# -  a high point and a low point that are not directly connected 
# -  two points that are actually at the extremes of the graph

def generate_scenarios(runs, graph, res):
    if res%2 != 0:
        res -=1 # it will exclude the last row on the left but it will work, at least
        print(f"Even number needed: a resolution of {res} will be considered")
    
    res = int(res)
    half = res//2

    # let's divide the graph into quadrants to ensure points are distant enough
    x, y = np.indices((res, res))

    # Mask for the SW quadrant (Quad 1)
    mask1 = (x < half) & (y < half)
    mask2 = (x >= half) & (y < half) 
    mask3 = (x >= half) & (y>= half)
    mask4 = (x < half) & (y >= half)

    quad1_ind = np.where(mask1.flatten())[0]
    quad2_ind = np.where(mask2.flatten())[0]
    quad3_ind = np.where(mask3.flatten())[0]
    quad4_ind = np.where(mask4.flatten())[0]

    quad1 = set(quad1_ind.tolist())
    quad2 = set(quad2_ind.tolist())
    quad3 = set(quad3_ind.tolist())
    quad4 = set(quad4_ind.tolist())

    taken = set()
    categories = {
    "water": set(),
    "low": set(),
    "high": set()}

    for node, data in graph.nodes(data=True):
        if data.get("is_water"):
            categories["water"].add(node)
        elif data.get("elevation", 0) < 600:
            categories["low"].add(node)
        else:
            categories["high"].add(node)

    # now I set the start and finish nodes for each type of route 
    elevation_couples = list() # high vs low
    water_couples = list() # easiest path crosses water
    distant_couples = list() # nodes that are far apart

    low = list(categories["low"])
    high = list(categories["high"])
    water = list(categories["water"])
    #random shuffle
    random.shuffle(low)
    random.shuffle(high)
    random.shuffle(water)
    # I need as many scenario couples as the number of separate runs n
    for _ in range(runs): 

        # pick a low node and a high node
        if not low or not high: break
        start = low.pop()
        while start in taken and low:
            start = low.pop()
        taken.add(start)
        finish = high.pop()
        while finish in taken and high:
            finish = high.pop()
        taken.add(finish)
        elev_tuple = [start,finish]
        random.shuffle(elev_tuple)
        elevation_couples.append(tuple(elev_tuple))
        
        #pick a node nearby a water node and another one on the same line
        wat1 = water.pop()
        if wat1 in quad1 or wat1 in quad4:
            while start in taken and water:
                start = wat1 - res
            taken.add(start)
            while finish in taken and water:
                finish = start + res*res//2
            taken.add(finish)
        else:
            while start in taken and water:
                start = wat1 + res
            taken.add(start)
            while finish in taken and water:
                finish = start - res*res//2
            taken.add(finish)
        couple = [start,finish]
        random.shuffle(couple)
        water_couples.append(tuple(couple))
        
        # pick two nodes that are on the furthermost edges

        quad1_left = quad1_ind[:res//10].tolist()
        quad4_left = quad4_ind[:res//10].tolist()
        quad2_right = quad2_ind[-res//10:].tolist()
        quad3_right = quad2_ind[-res//10:].tolist()

        random.shuffle(quad1_left)
        random.shuffle(quad4_left)
        random.shuffle(quad2_right)
        random.shuffle(quad3_right)

        r = random.random()

        if r< 0.5:
            while start in taken and quad1_left:
                start = quad1_left.pop()
            taken.add(start)
            while finish in taken and quad3_right:
                finish = quad3_right.pop()    
            taken.add(finish) 
        else:
            while start in taken and quad4_left:
                start = quad4_left.pop()
            taken.add(start)
            while finish in taken and quad2_right:
                finish = quad2_right.pop()    
            taken.add(finish) 
        couple = [start, finish]
        random.shuffle(couple)
        distant_couples.append(tuple(couple))
    scenarios = [water_couples, elevation_couples, distant_couples]
    #visualize_scenarios(graph, scenarios, runs, dpi = 200)
    return scenarios

def visualize_scenarios(graph,scenario, runs,
            draw_labels = False,
            figsize= (100,100),
            dpi=100
    ):
        plt.figure(figsize=figsize, dpi=dpi)
        labels = nx.get_node_attributes(graph, 'label')
        pos = graph.node_to_pos
        nx.draw_networkx_edges(graph, pos, edge_color="gray")
        node_costs = [graph.nodes[node].get('elevation', 0) for node in graph.nodes()]
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_costs,
            cmap='Greys', 
            node_size=10,
        )
        if graph.key_nodes is not None:
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=graph.key_nodes,
                node_color="green",
                node_size=300,
            )
        if draw_labels:
            nx.draw_networkx_labels(graph, graph.node_to_pos, labels=labels)
        water_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_water')]
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=water_nodes,
            node_color='lightblue',
            node_size=10,
        )
        colors = mpl.colormaps["Reds"].resampled(len(scenario)*runs)(range(len(scenario)*runs))
        k = 0
        for i in range(len(scenario)):
            for j in range(len(scenario[i])):
                nx.draw_networkx_nodes(
                    graph, pos,
                    nodelist = list(scenario[i][j]),
                    node_color=[colors[k]],
                    node_size=10,
                )
                k +=1
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    res = 100
    tif_path = "../TerrainGraph/trentino.tif"
    osm_path = "../TerrainGraph/trentino_alto_adige.pbf"
    graph = create_graph(tif_path=tif_path, osm_pbf_path=osm_path, resolution=res)
    runs = 5
    scenarios = generate_scenarios(runs,graph,res= res)
    visualize_scenarios(graph, scenarios, dpi = 200, runs = runs)

     
        
            





