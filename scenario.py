from terraingraph import create_graph
import random
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl

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
    # let's divide the graph into quadrants to ensure points are distant enough

    x1 = [num for num in range(0,res*(res//2 -1) +1,res)]
    x2 = [num for num in range(res*(res)//2, res*(res-1)+1, res)]
    quad1 = []
    quad2 = []
    quad3 = []
    quad4 = []
    for i in range(len(x1)):
        for j in range(0, res//2):
            quad1.append(x1[i]+j) # first quadrant, SW
            quad2.append(x2[i]+j) # second quadrant, SE
            quad3.append(x2[i]+res//2+j) # third quadrant, NE
            quad4.append(x1[i]+res//2+j) # fourth quadrant, NW
    water = list()
    taken = list()
    high = list()
    low = list()
    # first for robustness I give priority to water nodes that are close to each other (more difficult to avoid)
    for u,v in graph.edges():
        node_u = graph.nodes[u]
        node_v = graph.nodes[v]
        if node_u["is_water"] and node_v["is_water"]:
            water.append((u,v))
            taken.append(u)
            taken.append(v)
    for w in graph.nodes():
        if w not in taken:
            # if I don't have enough coupled water points, I consider the single ones
            if len(water) < runs:
                if graph.nodes[w]["is_water"]:
                    water.append((w,))
                    taken.append(w)
            # I separate the remaining nodes based on elevation (don't care for water nodes)
            elif graph.nodes[w]["elevation"] < 600:
                low.append(w)
            else:
                high.append(w)  
    # now I set the start and finish nodes for each type of route 
    elevation_couples = list() # high vs low
    water_couples = list() # easiest path crosses water
    distant_couples = list() # nodes that are far apart
    # I need as many scenario couples as the number of separate runs n
    for _ in range(runs): 
        low = [x for x in low if x not in taken]
        low_node = random.choice(low)
        taken.append(low_node)
        high_nodes = [n for n in graph.nodes() if (graph.nodes[n]["elevation"] - graph.nodes[low_node]["elevation"]) > 1000]
        high_node = random.choice(high_nodes)
        while high_node in taken:
            high_node = random.choice(high_nodes)
                # introduce stochastic ordering of items vs overfitting
        p = random.random()
        if p < 0.5:
            elevation_couples.append((low_node, high_node))
        else:
            elevation_couples.append((high_node, low_node))
        water_choice = random.choice(water)[0] # number between 0 and res^2-1 (included)
        q = random.random()
        if water_choice in quad1 or water_choice in quad4:
            n = water_choice -res
            while n in taken:
                n -= res
                if n <0:
                    n = random.choice(water)[0] - res
                if n > res*res-1:
                    n = random.choice(water)[0] - res
            taken.append(n)
            f = n + res*(res//2)
            while f in taken:
                f += res
            taken.append(f)
            if q <0.5:
                water_couples.append((n, f))
            else:
                water_couples.append((f, n))
        else:
            n = water_choice + res
            while n in taken:
                n += res
                if n <0:
                    n = random.choice(water)[0] + res
                if n > res*res-1:
                    n = random.choice(water)[0] + res 
            taken.append(n)
            f = n - res*(res//2)
            while f in taken:
                f -= res
            taken.append(f)
            if q < 0.5:
                water_couples.append((n, f))
            else:
                water_couples.append((f, n))
        r = random.random()
        if r<0.5:
            quad1_left = quad1[:res*res//16] # to ensure max distance, I want to pick numbers on the leftmost part of the quadrant
            quad3_right = quad3[-res*res//16:] # on the rightmost part of the quadrant
            s = random.random()
            if s < 0.5:
                distant_couples.append((random.choice(quad1_left), random.choice(quad3_right)))
            else:
                distant_couples.append((random.choice(quad3_right), random.choice(quad1_left)))
        else:
            quad4_left = quad4[:res*res//16]
            quad2_right = quad2[-res*res//16:]
            t = random.random()
            if t < 0.5:
                distant_couples.append((random.choice(quad2_right), random.choice(quad4_left)))
            else:
                distant_couples.append((random.choice(quad4_left), random.choice(quad2_right)))
    scenarios = [water_couples, elevation_couples, distant_couples]
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
    tif_path = "trentino.tif"
    osm_path = "trentino_alto_adige.pbf"
    graph = create_graph(tif_path=tif_path, osm_pbf_path=osm_path, resolution=res)
    runs = 5
    scenarios = generate_scenarios(runs,graph,res= res)
    visualize_scenarios(graph, scenarios, dpi = 200, runs = runs)

     
        
            





